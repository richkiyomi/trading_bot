"""
Automated Iron Condor Scanner (Local)

Goals (based on user spec):
- 14 DTE target
- Use delta filter on short legs to approximate ~70–75% probability of profit
- Enforce risk box:
  - Min credit per contract: $100 (i.e. >= $1.00 option credit)
  - Min total potential profit per trade: $700
  - Max loss per trade: $1,000
  - R/R at least ~70/30 in our favor (max_profit / max_loss >= 0.7)
- Only one iron condor open at any time (checked via DB status)

This is a LOCAL script for testing / scanning. It:
- Scans a fixed symbol list
- Applies delta + risk filters
- Proposes the best candidate + recommended quantity
- Optionally submits an MLEG order using existing helper
"""

import os
from math import ceil, floor
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetStatus
from alpaca.trading.requests import GetOptionContractsRequest

from models import get_db_session, IronCondor, CondorStatus

# Reuse core utilities from the existing scanner
from iron_condor_scanner import (
    get_days_to_expiration,
    get_option_price,
    calculate_iron_condor,
    place_iron_condor_mleg_order,
    estimate_spot_price,
)


# Environment / client -------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
ENV = os.getenv("APCA_ENV", "live")
paper = ENV.lower() != "live"

client = TradingClient(api_key=API_KEY, secret_key=SECRET_KEY, paper=paper)


# Configuration --------------------------------------------------------------

TARGET_DTE = 14
DTE_TOLERANCE = 3

# Approximate 70–80% PoP via short-leg deltas.
# Start with a slightly wider band to account for greeks noise / availability.
CALL_DELTA_MIN = 0.20
CALL_DELTA_MAX = 0.35
PUT_DELTA_MIN = 0.20  # use abs(delta) for puts
PUT_DELTA_MAX = 0.35

# Risk / reward box (per trade)
MIN_CREDIT_PER_CONTRACT_DOLLARS = 100.0  # => >= $1.00 credit
MIN_TOTAL_PROFIT_DOLLARS = 700.0
MAX_TOTAL_LOSS_DOLLARS = 1000.0
MIN_PROFIT_TO_LOSS_RATIO = 0.7  # max_profit / max_loss >= 0.7

# Alpaca doesn't support VIX options as an underlying symbol; use equity/ETF list only.
SYMBOLS = ["MSFT", "AAPL", "IWM", "SPY", "QQQ"]


# Helpers --------------------------------------------------------------------

def get_delta(contract) -> Optional[float]:
    """
    Try to extract delta from an Alpaca OptionContract.
    We handle both direct `delta` attribute and `greeks.delta` if present.
    """
    try:
        if hasattr(contract, "delta") and contract.delta is not None:
            return float(contract.delta)
        greeks = getattr(contract, "greeks", None)
        if greeks is not None and getattr(greeks, "delta", None) is not None:
            return float(greeks.delta)
    except Exception:
        pass
    return None


def get_option_contracts_for_dte(symbol: str) -> Dict:
    """
    Very similar to iron_condor_scanner.get_option_contracts, but
    hard-wired to our 14 DTE target and returned here so this file
    is self-contained for scanning logic.
    """
    today = date.today()
    min_expiration = today + timedelta(days=TARGET_DTE - DTE_TOLERANCE)
    max_expiration = today + timedelta(days=TARGET_DTE + DTE_TOLERANCE)

    request = GetOptionContractsRequest(
        underlying_symbols=[symbol.upper()],
        status=AssetStatus.ACTIVE,
        expiration_date_gte=min_expiration,
        expiration_date_lte=max_expiration,
        limit=10000,
    )

    response = client.get_option_contracts(request)
    all_contracts = list(response.option_contracts) if response.option_contracts else []

    # Pagination
    page_token = getattr(response, "next_page_token", None)
    while page_token:
        request.page_token = page_token
        next_response = client.get_option_contracts(request)
        if next_response.option_contracts:
            all_contracts.extend(next_response.option_contracts)
        page_token = getattr(next_response, "next_page_token", None)

    if not all_contracts:
        return {"calls": [], "puts": [], "expiration_date": None}

    # Group expirations and choose the one closest to TARGET_DTE
    expirations: Dict[date, int] = {}
    for c in all_contracts:
        if not c.tradable:
            continue
        d = c.expiration_date
        expirations[d] = get_days_to_expiration(d)

    if not expirations:
        return {"calls": [], "puts": [], "expiration_date": None}

    target_exp = today + timedelta(days=TARGET_DTE)
    best_exp = min(expirations.keys(), key=lambda d: abs((d - target_exp).days))

    calls = [c for c in all_contracts if c.tradable and c.expiration_date == best_exp and c.type.value == "call"]
    puts = [p for p in all_contracts if p.tradable and p.expiration_date == best_exp and p.type.value == "put"]

    calls.sort(key=lambda x: x.strike_price)
    puts.sort(key=lambda x: x.strike_price)

    return {"calls": calls, "puts": puts, "expiration_date": best_exp}


def has_open_condor() -> bool:
    """Check DB for any condor with PENDING_OPEN or OPEN status."""
    session = get_db_session()
    try:
        count = (
            session.query(IronCondor)
            .filter(IronCondor.status.in_([CondorStatus.PENDING_OPEN, CondorStatus.OPEN]))
            .count()
        )
        return count > 0
    finally:
        session.close()


def scan_symbol_with_delta_and_risk(symbol: str) -> List[Dict]:
    """
    Scan a single symbol for iron condor candidates that:
    - Use 14 DTE options
    - Have short legs with delta in our preferred range
    - Satisfy our risk box at a per-condor level (before sizing)
    """
    print(f"\n=== Scanning {symbol} for 14 DTE condors with delta filters ===")
    try:
        contracts = get_option_contracts_for_dte(symbol)
    except APIError as e:
        print(f"❌ API error for {symbol}: {e}")
        return []
    except Exception as e:
        print(f"❌ Error for {symbol}: {e}")
        return []

    calls = contracts["calls"]
    puts = contracts["puts"]
    expiration_date = contracts["expiration_date"]

    if not expiration_date or not calls or not puts:
        print(f"⚠️  No suitable contracts for {symbol}")
        return []

    # Try to estimate spot from strikes (used for fallback when greeks missing)
    estimated_spot = estimate_spot_price(calls + puts)
    if estimated_spot:
        print(f"  Estimated spot from strikes: {estimated_spot:.2f}")
    else:
        print("  Could not estimate spot from strikes.")

    # Filter candidate short legs by delta (preferred, when greeks exist)
    total_call_with_delta = 0
    total_put_with_delta = 0

    call_shorts = []
    for c in calls:
        d = get_delta(c)
        if d is None:
            continue
        total_call_with_delta += 1
        if CALL_DELTA_MIN <= d <= CALL_DELTA_MAX:
            call_shorts.append((c, d))

    put_shorts = []
    for p in puts:
        d = get_delta(p)
        if d is None:
            continue
        total_put_with_delta += 1
        if PUT_DELTA_MIN <= abs(d) <= PUT_DELTA_MAX:
            put_shorts.append((p, d))

    print(
        f"  Contracts with usable delta -> calls: {total_call_with_delta}, puts: {total_put_with_delta}"
    )

    # Fallback: if no greeks, approximate "delta region" via moneyness bands around spot
    if (total_call_with_delta == 0 or total_put_with_delta == 0) and estimated_spot:
        print("  No greeks available; falling back to moneyness bands for short strikes.")
        call_shorts = []
        put_shorts = []

        # Treat ~5–15% OTM as a proxy for ~0.15–0.30 delta
        call_lower = estimated_spot * 1.05
        call_upper = estimated_spot * 1.15
        put_lower = estimated_spot * 0.85
        put_upper = estimated_spot * 0.95

        for c in calls:
            if call_lower <= c.strike_price <= call_upper:
                call_shorts.append((c, None))

        for p in puts:
            if put_lower <= p.strike_price <= put_upper:
                put_shorts.append((p, None))

        print(
            f"  Fallback short call candidates (5–15% OTM): {len(call_shorts)}, "
            f"short put candidates (5–15% OTM): {len(put_shorts)}"
        )

    if not call_shorts or not put_shorts:
        print(f"⚠️  No short legs within chosen region for {symbol}")
        return []

    print(f"  Short call candidates (delta in [{CALL_DELTA_MIN:.2f}, {CALL_DELTA_MAX:.2f}]): {len(call_shorts)}")
    print(f"  Short put candidates (|delta| in [{PUT_DELTA_MIN:.2f}, {PUT_DELTA_MAX:.2f}]): {len(put_shorts)}")

    condors: List[Dict] = []
    # Debug counters to understand why candidates are filtered out
    total_pairs = 0
    price_ok = 0
    min_credit_ok = 0
    ratio_ok = 0

    # For simplicity, choose long legs further OTM than the short legs on each side
    for call_short, call_delta in call_shorts:
        for put_short, put_delta in put_shorts:
            # Ensure put short is below call short in strike
            if put_short.strike_price >= call_short.strike_price:
                continue

            # Choose long call as the nearest strike above short call
            call_longs = [c for c in calls if c.strike_price > call_short.strike_price]
            if not call_longs:
                continue
            call_long = call_longs[0]

            # Choose long put as the nearest strike below short put
            put_longs = [p for p in puts if p.strike_price < put_short.strike_price]
            if not put_longs:
                continue
            put_long = put_longs[-1]

            # Prices
            call_short_price = get_option_price(call_short)
            call_long_price = get_option_price(call_long)
            put_short_price = get_option_price(put_short)
            put_long_price = get_option_price(put_long)

            total_pairs += 1

            if not all([call_short_price, call_long_price, put_short_price, put_long_price]):
                continue
            price_ok += 1

            metrics = calculate_iron_condor(
                call_short.strike_price,
                call_long.strike_price,
                put_short.strike_price,
                put_long.strike_price,
                call_short_price,
                call_long_price,
                put_short_price,
                put_long_price,
            )

            credit_per_condor = metrics["max_profit"] * 100.0  # dollars
            loss_per_condor = metrics["max_loss"] * 100.0  # dollars

            # Per-condor filters
            if credit_per_condor < MIN_CREDIT_PER_CONTRACT_DOLLARS:
                continue
            min_credit_ok += 1
            if loss_per_condor <= 0:
                continue
            profit_to_loss_ratio = credit_per_condor / loss_per_condor
            if profit_to_loss_ratio < MIN_PROFIT_TO_LOSS_RATIO:
                continue
            ratio_ok += 1

            condors.append(
                {
                    "symbol": symbol.upper(),
                    "expiration_date": expiration_date,
                    "call_short_strike": call_short.strike_price,
                    "call_long_strike": call_long.strike_price,
                    "put_short_strike": put_short.strike_price,
                    "put_long_strike": put_long.strike_price,
                    "call_short_symbol": call_short.symbol,
                    "call_long_symbol": call_long.symbol,
                    "put_short_symbol": put_short.symbol,
                    "put_long_symbol": put_long.symbol,
                    "call_short_delta": call_delta,
                    "put_short_delta": put_delta,
                    "call_short_price": call_short_price,
                    "call_long_price": call_long_price,
                    "put_short_price": put_short_price,
                    "put_long_price": put_long_price,
                    **metrics,
                    "profit_to_loss_ratio": profit_to_loss_ratio,
                    "credit_dollars": credit_per_condor,
                    "loss_dollars": loss_per_condor,
                }
            )

    print(
        f"  Debug: pairs={total_pairs}, price_ok={price_ok}, "
        f"min_credit_ok={min_credit_ok}, ratio_ok={ratio_ok}, final={len(condors)}"
    )

    # Sort by score from original metrics (if multiple candidates)
    condors.sort(key=lambda c: c["score"], reverse=True)
    return condors


def choose_best_condor(all_condors: List[Dict]) -> Optional[Dict]:
    """Pick the single best condor across all symbols (by score)."""
    if not all_condors:
        return None
    return max(all_condors, key=lambda c: c["score"])


def compute_recommended_quantity(condor: Dict) -> Optional[int]:
    """
    Given a condor candidate, compute the recommended quantity that:
    - Achieves >= MIN_TOTAL_PROFIT_DOLLARS
    - Keeps loss <= MAX_TOTAL_LOSS_DOLLARS
    """
    credit_per_condor = condor["credit_dollars"]
    loss_per_condor = condor["loss_dollars"]

    # Minimum contracts to hit target profit
    min_qty_for_profit = ceil(MIN_TOTAL_PROFIT_DOLLARS / credit_per_condor)

    # Maximum contracts allowed by risk cap
    max_qty_by_risk = floor(MAX_TOTAL_LOSS_DOLLARS / loss_per_condor)

    if max_qty_by_risk <= 0:
        return None

    qty = max(min_qty_for_profit, 1)
    if qty > max_qty_by_risk:
        return None
    return qty


def main():
    print("=" * 100)
    print("LOCAL AUTO SCANNER - 14 DTE Iron Condors with Delta & Risk Filters")
    print("=" * 100)

    # Enforce "only one condor at a time"
    if has_open_condor():
        print("\n⚠️  Existing iron condor found in DB (PENDING_OPEN or OPEN).")
        print("    No new trades will be proposed.")
        return

    all_condors: List[Dict] = []

    for symbol in SYMBOLS:
        condors = scan_symbol_with_delta_and_risk(symbol)
        if condors:
            print(f"  → {len(condors)} candidates for {symbol}")
            all_condors.extend(condors)

    if not all_condors:
        print("\n❌ No condors met delta + per-condor risk filters.")
        return

    best = choose_best_condor(all_condors)
    if not best:
        print("\n❌ Could not select a best condor.")
        return

    qty = compute_recommended_quantity(best)
    if qty is None:
        print("\n⚠️  Best condor does not allow a quantity that meets BOTH:")
        print(f"    - Min profit ${MIN_TOTAL_PROFIT_DOLLARS:,.0f}")
        print(f"    - Max loss ${MAX_TOTAL_LOSS_DOLLARS:,.0f}")
        return

    print("\n=== RECOMMENDED CONDOR ===")
    dte = get_days_to_expiration(best["expiration_date"])
    print(f"Symbol: {best['symbol']}")
    print(f"Expiration: {best['expiration_date']} ({dte} DTE)")
    print(f"Put spread:  {best['put_short_strike']} / {best['put_long_strike']}")
    print(f"Call spread: {best['call_short_strike']} / {best['call_long_strike']}")
    if best["put_short_delta"] is not None:
        print(f"Short put delta:  {best['put_short_delta']:.3f}")
    if best["call_short_delta"] is not None:
        print(f"Short call delta: {best['call_short_delta']:.3f}")
    print(f"Credit per condor: ${best['credit_dollars']:.2f}")
    print(f"Max loss per condor: ${best['loss_dollars']:.2f}")
    print(f"Profit/Loss ratio: {best['profit_to_loss_ratio']:.2f} (>= {MIN_PROFIT_TO_LOSS_RATIO:.2f} required)")

    total_credit = best["credit_dollars"] * qty
    total_loss = best["loss_dollars"] * qty

    print("\n=== SIZING ===")
    print(f"Recommended quantity: {qty} condor(s)")
    print(f"Total potential profit (credit): ${total_credit:,.2f}")
    print(f"Total potential max loss:        ${total_loss:,.2f}")

    # Ask user whether to place order now (still local, with full visibility)
    confirm = input("\nPlace this condor now as an MLEG order? (yes/no): ").strip().lower()
    if confirm not in ("yes", "y"):
        print("No order placed.")
        return

    try:
        place_iron_condor_mleg_order(best, qty)
    except Exception as e:
        print(f"❌ Error placing order: {e}")


if __name__ == "__main__":
    main()


