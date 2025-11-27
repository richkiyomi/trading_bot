"""
Iron Condor Scanner - Optimized for after-hours/pre-market analysis
Uses close prices (stale data) - suitable for planning trades outside market hours
"""
import os
import sys
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest, MarketOrderRequest, LimitOrderRequest, OptionLegRequest
from alpaca.trading.enums import ContractType, AssetStatus, OrderClass, OrderType, TimeInForce
from alpaca.common.exceptions import APIError
from models import (
    get_db_session, IronCondor, IronCondorLeg, Order,
    CondorStatus, LegRole, OrderSide, OrderType as DBOrderType, TimeInForce as DBTimeInForce
)
from alpaca.trading.enums import OrderType as AlpacaOrderType

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
ENV = os.getenv("APCA_ENV", "live")
paper = ENV.lower() != "live"

# Scanner configuration (works with close prices)
MIN_CREDIT_RATIO = 0.20  # Credit must be ≥ 20% of wing width
MIN_WING_WIDTH = 5.0     # Minimum $5 wing width
STRIKE_RANGE_PCT = 0.30  # Consider strikes within ±30% of estimated spot

# Initialize TradingClient
client = TradingClient(api_key=API_KEY, secret_key=SECRET_KEY, paper=paper)

def get_days_to_expiration(expiration_date: date) -> int:
    """Calculate days to expiration."""
    return (expiration_date - date.today()).days

def is_market_open() -> bool:
    """Check if the market is currently open."""
    try:
        clock = client.get_clock()
        return clock.is_open
    except Exception as e:
        print(f"⚠️  Could not check market status: {e}")
        # Default to False (market closed) to be safe
        return False

def get_option_contracts(symbol: str, target_dte: int = 30, dte_tolerance: int = 5) -> Dict:
    """Get option contracts filtered by DTE."""
    try:
        print(f"Fetching option contracts for {symbol}...")
        
        today = date.today()
        min_expiration = today + timedelta(days=target_dte - dte_tolerance)
        max_expiration = today + timedelta(days=target_dte + dte_tolerance)
        
        request = GetOptionContractsRequest(
            underlying_symbols=[symbol.upper()],
            status=AssetStatus.ACTIVE,
            expiration_date_gte=min_expiration,
            expiration_date_lte=max_expiration,
            limit=10000
        )
        
        response = client.get_option_contracts(request)
        all_contracts = list(response.option_contracts) if response.option_contracts else []
        
        # Handle pagination
        page_token = response.next_page_token if hasattr(response, 'next_page_token') else None
        while page_token:
            request.page_token = page_token
            next_response = client.get_option_contracts(request)
            if next_response.option_contracts:
                all_contracts.extend(next_response.option_contracts)
            page_token = next_response.next_page_token if hasattr(next_response, 'next_page_token') else None
        
        # Filter and organize
        expiration_dates = {}
        for contract in all_contracts:
            if contract.tradable:
                exp_date = contract.expiration_date
                dte = get_days_to_expiration(exp_date)
                if exp_date not in expiration_dates:
                    expiration_dates[exp_date] = {'dte': dte, 'count': 0}
                expiration_dates[exp_date]['count'] += 1
        
        if expiration_dates:
            print(f"📅 Available expiration dates:")
            for exp_date, info in sorted(expiration_dates.items()):
                print(f"   {exp_date.strftime('%Y-%m-%d')}: {info['dte']} DTE ({info['count']} contracts)")
            
            # Use closest to target DTE
            target_exp = today + timedelta(days=target_dte)
            best_exp = min(expiration_dates.keys(), 
                         key=lambda d: abs((d - target_exp).days))
            actual_dte = expiration_dates[best_exp]['dte']
            
            calls = [c for c in all_contracts 
                    if c.tradable and c.expiration_date == best_exp and c.type == ContractType.CALL]
            puts = [p for p in all_contracts 
                   if p.tradable and p.expiration_date == best_exp and p.type == ContractType.PUT]
            
            calls.sort(key=lambda x: x.strike_price)
            puts.sort(key=lambda x: x.strike_price)
            
            print(f"✅ Found {len(calls)} calls and {len(puts)} puts for {best_exp.strftime('%Y-%m-%d')} ({actual_dte} DTE)")
            
            return {'calls': calls, 'puts': puts, 'expiration_date': best_exp}
        
        return {'calls': [], 'puts': [], 'expiration_date': None}
        
    except APIError as e:
        print(f"❌ API Error: {e.message}")
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

def get_option_price(contract) -> float:
    """Get option price (using close_price - stale but available after hours)."""
    if contract.close_price:
        return float(contract.close_price)
    return 0.0

def estimate_spot_price(contracts: List) -> Optional[float]:
    """
    Estimate current spot price from option contracts.
    Uses ATM options to infer spot (rough estimate for after-hours).
    """
    if not contracts:
        return None
    
    # Find ATM strikes (where call and put prices are closest)
    strikes = sorted(set(c.strike_price for c in contracts))
    if len(strikes) < 2:
        return strikes[0] if strikes else None
    
    # Use middle strike as rough estimate
    mid_strike = strikes[len(strikes) // 2]
    return mid_strike

def calculate_iron_condor(
    call_short_strike: float, call_long_strike: float,
    put_short_strike: float, put_long_strike: float,
    call_short_price: float, call_long_price: float,
    put_short_price: float, put_long_price: float
) -> Dict:
    """Calculate iron condor metrics with credit ratio and scoring."""
    # Net credit
    call_credit = call_short_price - call_long_price
    put_credit = put_short_price - put_long_price
    credit_received = call_credit + put_credit
    
    # Spread widths
    call_width = call_long_strike - call_short_strike
    put_width = put_short_strike - put_long_strike
    total_wing = call_width + put_width
    
    # Max loss and profit
    max_loss = total_wing - credit_received
    max_profit = credit_received
    
    # Credit ratio (key metric for quality)
    credit_ratio = (credit_received / total_wing) if total_wing > 0 else 0
    
    # Risk/Reward ratio
    rr_ratio = (max_loss / max_profit) if max_profit > 0 else float('inf')
    
    # Profit zone width
    profit_zone_width = call_short_strike - put_short_strike
    
    # Scoring: rewards high credit ratio, credit amount, penalizes poor R/R
    score = (credit_ratio * 100) + (credit_received * 10) - (min(rr_ratio, 10) * 5)
    
    return {
        'credit_received': credit_received,
        'max_loss': max_loss,
        'max_profit': max_profit,
        'call_width': call_width,
        'put_width': put_width,
        'total_wing': total_wing,
        'credit_ratio': credit_ratio,
        'rr_ratio': rr_ratio,
        'profit_zone_width': profit_zone_width,
        'score': score,
        'call_credit': call_credit,
        'put_credit': put_credit
    }

def scan_iron_condors(symbol: str, target_dte: int = 30, max_results: int = 50) -> List[Dict]:
    """Scan for iron condor opportunities."""
    contracts = get_option_contracts(symbol, target_dte)
    
    if not contracts['calls'] or not contracts['puts']:
        print(f"⚠️  No suitable contracts found for {symbol}")
        return []
    
    calls = contracts['calls']
    puts = contracts['puts']
    expiration_date = contracts['expiration_date']
    
    # Estimate spot price from strikes
    all_contracts = calls + puts
    estimated_spot = estimate_spot_price(all_contracts)
    if estimated_spot:
        print(f"📍 Estimated spot: ${estimated_spot:.2f} (from option strikes)")
    
    print(f"\n🔍 Scanning for iron condors...")
    
    iron_condors = []
    
    # Filter strikes within reasonable range
    if estimated_spot:
        lower_bound = estimated_spot * (1 - STRIKE_RANGE_PCT)
        upper_bound = estimated_spot * (1 + STRIKE_RANGE_PCT)
        calls = [c for c in calls if lower_bound <= c.strike_price <= upper_bound]
        puts = [p for p in puts if lower_bound <= p.strike_price <= upper_bound]
        print(f"   Filtered to {len(calls)} calls and {len(puts)} puts within ±{int(STRIKE_RANGE_PCT*100)}% of spot")
    
    # Generate combinations
    for call_short in calls:
        for call_long in calls:
            if call_long.strike_price <= call_short.strike_price:
                continue
            
            for put_short in puts:
                if put_short.strike_price >= call_short.strike_price:
                    continue
                
                for put_long in puts:
                    if put_long.strike_price >= put_short.strike_price:
                        continue
                    
                    # Get prices
                    call_short_price = get_option_price(call_short)
                    call_long_price = get_option_price(call_long)
                    put_short_price = get_option_price(put_short)
                    put_long_price = get_option_price(put_long)
                    
                    if not all([call_short_price, call_long_price, put_short_price, put_long_price]):
                        continue
                    
                    # Calculate metrics
                    metrics = calculate_iron_condor(
                        call_short.strike_price, call_long.strike_price,
                        put_short.strike_price, put_long.strike_price,
                        call_short_price, call_long_price,
                        put_short_price, put_long_price
                    )
                    
                    # Quality filters
                    if metrics['credit_received'] <= 0:
                        continue
                    if metrics['total_wing'] < MIN_WING_WIDTH:
                        continue
                    if metrics['credit_ratio'] < MIN_CREDIT_RATIO:
                        continue
                    
                    condor = {
                        'symbol': symbol.upper(),
                        'expiration_date': expiration_date,
                        'estimated_spot': estimated_spot,
                        'call_short_strike': call_short.strike_price,
                        'call_long_strike': call_long.strike_price,
                        'put_short_strike': put_short.strike_price,
                        'put_long_strike': put_long.strike_price,
                        'call_short_symbol': call_short.symbol,  # Store contract symbols for order placement
                        'call_long_symbol': call_long.symbol,
                        'put_short_symbol': put_short.symbol,
                        'put_long_symbol': put_long.symbol,
                        'call_short_price': call_short_price,
                        'call_long_price': call_long_price,
                        'put_short_price': put_short_price,
                        'put_long_price': put_long_price,
                        **metrics
                    }
                    
                    iron_condors.append(condor)
    
    # Sort by score (best first)
    iron_condors.sort(key=lambda x: x['score'], reverse=True)
    return iron_condors[:max_results]

def filter_top_rr_by_symbol(iron_condors: List[Dict], top_n: int = 2) -> List[Dict]:
    """
    Filter to top N iron condors by R/R ratio for each symbol.
    Lower R/R ratio is better (less risk per unit of reward).
    
    Args:
        iron_condors: List of iron condor dictionaries
        top_n: Number of top condors to keep per symbol (default 2)
        
    Returns:
        Filtered list with top N condors per symbol
    """
    from collections import defaultdict
    
    # Group by symbol
    by_symbol = defaultdict(list)
    for condor in iron_condors:
        by_symbol[condor['symbol']].append(condor)
    
    # For each symbol, sort by R/R ratio (ascending - lower is better)
    # and take top N
    filtered = []
    for symbol, condors in by_symbol.items():
        # Sort by R/R ratio (lower is better), handling infinity
        sorted_condors = sorted(condors, 
                               key=lambda x: (x['rr_ratio'] if x['rr_ratio'] != float('inf') else 999999))
        filtered.extend(sorted_condors[:top_n])
    
    # Re-sort by score for final display
    filtered.sort(key=lambda x: x['score'], reverse=True)
    return filtered

def filter_iron_condors(iron_condors: List[Dict], filters: Dict) -> List[Dict]:
    """Apply filters to iron condors."""
    filtered = iron_condors.copy()
    
    if filters.get('min_credit') is not None:
        filtered = [c for c in filtered if c['credit_received'] >= filters['min_credit']]
    
    if filters.get('max_loss') is not None:
        filtered = [c for c in filtered if c['max_loss'] <= filters['max_loss']]
    
    if filters.get('max_rr_ratio') is not None:
        filtered = [c for c in filtered 
                    if c['max_profit'] > 0 and c['rr_ratio'] <= filters['max_rr_ratio']]
    
    if filters.get('min_credit_ratio') is not None:
        filtered = [c for c in filtered if c['credit_ratio'] >= filters['min_credit_ratio']]
    
    if filters.get('min_profit_zone_width') is not None:
        filtered = [c for c in filtered 
                    if c['profit_zone_width'] >= filters['min_profit_zone_width']]
    
    if filters.get('symbols') is not None and filters['symbols']:
        filtered = [c for c in filtered if c['symbol'] in filters['symbols']]
    
    return filtered

def get_filter_inputs() -> Dict:
    """Prompt for filter criteria."""
    filters = {}
    print("\n" + "="*100)
    print("FILTER OPTIONS (Press Enter to skip)")
    print("="*100)
    
    min_credit = input("Minimum credit received ($): ").strip()
    if min_credit:
        try:
            filters['min_credit'] = float(min_credit)
        except ValueError:
            pass
    
    max_loss = input("Maximum loss allowed ($): ").strip()
    if max_loss:
        try:
            filters['max_loss'] = float(max_loss)
        except ValueError:
            pass
    
    max_rr = input("Maximum R/R ratio (e.g., 2.0): ").strip()
    if max_rr:
        try:
            filters['max_rr_ratio'] = float(max_rr)
        except ValueError:
            pass
    
    min_credit_ratio = input("Minimum credit ratio (e.g., 0.25 for 25%): ").strip()
    if min_credit_ratio:
        try:
            filters['min_credit_ratio'] = float(min_credit_ratio)
        except ValueError:
            pass
    
    min_profit_zone = input("Minimum profit zone width ($): ").strip()
    if min_profit_zone:
        try:
            filters['min_profit_zone_width'] = float(min_profit_zone)
        except ValueError:
            pass
    
    symbol_filter = input("Filter by symbols (comma-separated): ").strip()
    if symbol_filter:
        filters['symbols'] = [s.strip().upper() for s in symbol_filter.split(',') if s.strip()]
    
    return filters

def display_iron_condors_table(all_results: List[Dict]):
    """Display results in table format."""
    if not all_results:
        print("\n❌ No iron condors found")
        return
    
    print(f"\n{'='*160}")
    print("IRON CONDOR SCAN RESULTS - AFTER-HOURS ANALYSIS")
    print("⚠️  Using close prices (stale data) - verify with live prices before trading")
    print(f"{'='*160}\n")
    
    header = (f"{'Rank':<5} {'Symbol':<8} {'Exp':<12} {'DTE':<4} {'Call Spread':<20} {'Put Spread':<20} "
              f"{'Credit':<8} {'MaxL':<8} {'Cred%':<7} {'R/R':<7} {'Score':<7} {'Profit Zone':<12}")
    print(header)
    print("-" * 160)
    
    for i, condor in enumerate(all_results, 1):
        exp_str = condor['expiration_date'].strftime('%Y-%m-%d')
        dte = get_days_to_expiration(condor['expiration_date'])
        call_spread = f"${condor['call_short_strike']:.0f}/${condor['call_long_strike']:.0f}"
        put_spread = f"${condor['put_short_strike']:.0f}/${condor['put_long_strike']:.0f}"
        credit = f"${condor['credit_received']:.2f}"
        max_loss = f"${condor['max_loss']:.2f}"
        cred_pct = f"{condor['credit_ratio']:.1%}"
        rr = f"{condor['rr_ratio']:.2f}:1" if condor['rr_ratio'] != float('inf') else "N/A"
        score = f"{condor['score']:.1f}"
        profit_zone = f"${condor['put_short_strike']:.0f}-${condor['call_short_strike']:.0f}"
        
        row = (f"{i:<5} {condor['symbol']:<8} {exp_str:<12} {dte:<4} {call_spread:<20} {put_spread:<20} "
               f"{credit:<8} {max_loss:<8} {cred_pct:<7} {rr:<7} {score:<7} {profit_zone:<12}")
        print(row)
    
    print("-" * 160)
    print(f"\nTotal: {len(all_results)} iron condors | Symbols: {len(set(c['symbol'] for c in all_results))}")

def place_iron_condor_mleg_order(condor: Dict, quantity: int):
    """
    Place an iron condor as a multi-leg (MLEG) order via Alpaca.
    MLEG orders can be placed outside market hours.
    
    Args:
        condor: Iron condor dictionary with contract symbols
        quantity: Number of condors (contracts per leg)
    """
    try:
        print("\n" + "="*100)
        print("PLACING IRON CONDOR MLEG ORDER")
        print("="*100)
        print(f"Symbol: {condor['symbol']}")
        print(f"Expiration: {condor['expiration_date'].strftime('%Y-%m-%d')}")
        print(f"Quantity: {quantity} condor(s)")
        print(f"Target Credit: ${condor['credit_received']:.2f} per condor")
        print(f"\nStructure:")
        print(f"  PUT:  BUY  {condor['put_long_symbol']} @ ${condor['put_long_strike']:.0f}")
        print(f"        SELL {condor['put_short_symbol']} @ ${condor['put_short_strike']:.0f}")
        print(f"  CALL: SELL {condor['call_short_symbol']} @ ${condor['call_short_strike']:.0f}")
        print(f"        BUY  {condor['call_long_symbol']} @ ${condor['call_long_strike']:.0f}")
        
        # Build the 4 legs for MLEG order
        # Order: Put Short, Put Long, Call Short, Call Long
        legs = [
            OptionLegRequest(
                symbol=condor['put_short_symbol'],
                side="sell",
                ratio_qty=1.0
            ),
            OptionLegRequest(
                symbol=condor['put_long_symbol'],
                side="buy",
                ratio_qty=1.0
            ),
            OptionLegRequest(
                symbol=condor['call_short_symbol'],
                side="sell",
                ratio_qty=1.0
            ),
            OptionLegRequest(
                symbol=condor['call_long_symbol'],
                side="buy",
                ratio_qty=1.0
            ),
        ]
        
        # Check if market is open to determine order type
        market_open = is_market_open()
        net_credit = condor['credit_received']
        
        print(f"\nMLEG Order Details:")
        print(f"  Target Credit: ${net_credit:.2f} per condor")
        print(f"  Max Loss: ${condor['max_loss']:.2f}")
        print(f"  Max Profit: ${condor['max_profit']:.2f}")
        
        # Use market order if market is open, limit order with GTC if closed
        if market_open:
            print(f"  ✅ Market is OPEN - Using MARKET order for immediate execution")
            order_request = MarketOrderRequest(
                qty=float(quantity),
                type=AlpacaOrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.MLEG,
                legs=legs,
            )
            use_market_order = True
        else:
            print(f"  ⏰ Market is CLOSED - Using LIMIT order with GTC (Good Till Canceled)")
            print(f"  Limit Price (credit): ${net_credit:.2f} → MLEG limit_price = {round(-net_credit, 2)}")
            mleg_limit_price = round(-net_credit, 2)
            order_request = LimitOrderRequest(
                qty=float(quantity),
                type=AlpacaOrderType.LIMIT,
                time_in_force=TimeInForce.GTC,  # GTC so order persists across trading days
                order_class=OrderClass.MLEG,
                legs=legs,
                limit_price=mleg_limit_price,
            )
            use_market_order = False
        
        # Submit order
        print(f"\n📤 Submitting MLEG order...")
        order = client.submit_order(order_request)
        
        # Display order details
        print(f"\n✅ Order Submitted Successfully!")
        print(f"  Order ID: {order.id}")
        print(f"  Status: {order.status}")
        print(f"  Quantity: {order.qty}")
        print(f"  Filled Qty: {order.filled_qty or 0}")
        if order.filled_avg_price is not None:
            print(f"  Filled Avg Price: {order.filled_avg_price}")
        print(f"  Submitted At: {order.submitted_at}")
        
        # Save to database
        print(f"\n💾 Saving to database...")
        try:
            session = get_db_session()
            try:
                # Create external tag
                now = datetime.utcnow()
                external_tag = f"CONDOR-{condor['symbol']}-{condor['expiration_date'].strftime('%Y%m%d')}-{int(now.timestamp())}"
                
                # Calculate entry net credit (per condor, in dollars)
                entry_net_credit = float(condor['credit_received']) * 100.0  # Convert to dollars
                
                # Create IronCondor record
                opening_order_id_str = str(order.id)
                iron_condor = IronCondor(
                    external_tag=external_tag,
                    underlying_symbol=condor['symbol'],
                    expiration_date=condor['expiration_date'],
                    dte_at_entry=get_days_to_expiration(condor['expiration_date']),
                    status=CondorStatus.PENDING_OPEN,
                    entry_time=None,  # Will be set when order fills
                    exit_time=None,
                    exit_reason=None,
                    opening_order_id=opening_order_id_str,  # Store Alpaca order ID for easy monitoring
                    closing_order_id=None,
                    entry_net_credit=entry_net_credit,
                    exit_net_debit=None,
                    realized_pnl=None,
                    max_profit=float(condor['max_profit']) * 100.0,  # Convert to dollars
                    max_loss=float(condor['max_loss']) * 100.0,  # Convert to dollars
                    credit_ratio=float(condor['credit_ratio']),
                    wing_width=float(condor['total_wing']),
                    otm_pct=None,  # Not tracked in current scanner
                    score=float(condor.get('score', 0)),
                    profit_target_pct=0.70,  # Default 70%
                    stop_loss_pct=0.70,  # Default 70%
                )
                session.add(iron_condor)
                session.flush()  # Get the ID
                
                # Create 4 leg records
                legs_data = [
                    {
                        'role': LegRole.PUT_LONG,
                        'symbol': condor['put_long_symbol'],
                        'type': 'PUT',
                        'strike': condor['put_long_strike'],
                        'side_open': OrderSide.BUY,
                        'price': condor['put_long_price'],
                    },
                    {
                        'role': LegRole.PUT_SHORT,
                        'symbol': condor['put_short_symbol'],
                        'type': 'PUT',
                        'strike': condor['put_short_strike'],
                        'side_open': OrderSide.SELL,
                        'price': condor['put_short_price'],
                    },
                    {
                        'role': LegRole.CALL_SHORT,
                        'symbol': condor['call_short_symbol'],
                        'type': 'CALL',
                        'strike': condor['call_short_strike'],
                        'side_open': OrderSide.SELL,
                        'price': condor['call_short_price'],
                    },
                    {
                        'role': LegRole.CALL_LONG,
                        'symbol': condor['call_long_symbol'],
                        'type': 'CALL',
                        'strike': condor['call_long_strike'],
                        'side_open': OrderSide.BUY,
                        'price': condor['call_long_price'],
                    },
                ]
                
                leg_objects = []
                for leg_data in legs_data:
                    leg = IronCondorLeg(
                        condor=iron_condor,
                        leg_role=leg_data['role'],
                        option_symbol=leg_data['symbol'],
                        option_type=leg_data['type'],
                        strike_price=float(leg_data['strike']),
                        expiration_date=condor['expiration_date'],
                        side_open=leg_data['side_open'],
                        side_close=None,
                        quantity=quantity,
                        open_avg_price=None,  # Will be updated when order fills
                        close_avg_price=None,
                        realized_pnl=None,
                    )
                    session.add(leg)
                    leg_objects.append(leg)
                
                session.flush()  # Get leg IDs
                
                # Create Order record for the MLEG parent order
                # Get order status as string
                order_status = str(order.status.value) if hasattr(order.status, 'value') else str(order.status)
                
                # Determine order type and limit price based on what was actually used
                if use_market_order:
                    db_order_type = DBOrderType.MARKET
                    db_limit_price = None
                    db_time_in_force = DBTimeInForce.DAY
                else:
                    db_order_type = DBOrderType.LIMIT
                    db_limit_price = float(net_credit)  # Store as positive credit value
                    db_time_in_force = DBTimeInForce.GTC
                
                db_order = Order(
                    alpaca_order_id=str(order.id),
                    client_order_id=order.client_order_id,
                    condor=iron_condor,
                    leg=None,  # MLEG parent order doesn't link to a specific leg
                    is_opening=True,
                    side=OrderSide.BUY,  # MLEG orders don't have a single side, use BUY as default
                    order_type=db_order_type,
                    time_in_force=db_time_in_force,
                    limit_price=db_limit_price,
                    stop_price=None,
                    status=order_status,
                    qty=int(order.qty),
                    filled_qty=int(order.filled_qty or 0),
                    avg_fill_price=float(order.filled_avg_price) if order.filled_avg_price is not None else None,
                    last_fill_price=None,
                    submitted_at=order.submitted_at if order.submitted_at else now,
                    filled_at=order.filled_at if hasattr(order, 'filled_at') and order.filled_at else None,
                    canceled_at=order.canceled_at if hasattr(order, 'canceled_at') and order.canceled_at else None,
                    expired_at=None,
                )
                session.add(db_order)
                
                # Commit transaction
                session.commit()
                
                print(f"✅ Saved to database:")
                print(f"   Condor ID: {iron_condor.id}")
                print(f"   External Tag: {external_tag}")
                print(f"   Legs: {len(leg_objects)} legs created")
                print(f"   Order ID: {db_order.id}")
                
            except Exception as db_error:
                session.rollback()
                print(f"⚠️  Database error: {db_error}")
                import traceback
                traceback.print_exc()
                # Don't fail the order placement if DB save fails
        except Exception as e:
            print(f"⚠️  Error saving to database: {e}")
            # Don't fail the order placement if DB save fails
        
        return order
        
    except APIError as e:
        print(f"\n❌ Order Error: {e.message}")
        if e.code:
            print(f"   Error Code: {e.code}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error placing order: {e}")
        raise

if __name__ == "__main__":
    try:
        print("=" * 100)
        print("IRON CONDOR SCANNER - After-Hours/Pre-Market Analysis")
        print("⚠️  Uses close prices (stale data) - suitable for planning outside market hours")
        print("=" * 100)
        
        # Get symbols
        print("\nEnter symbols (comma-separated or one per line, 'done' to finish):")
        symbols = []
        while True:
            user_input = input(f"Symbol {len(symbols) + 1}: ").strip().upper()
            if not user_input or user_input == 'DONE':
                break
            if ',' in user_input:
                symbols.extend([s.strip().upper() for s in user_input.split(',') if s.strip()])
            else:
                symbols.append(user_input)
        
        if not symbols:
            print("❌ No symbols entered")
            sys.exit(1)
        
        # Remove duplicates
        seen = set()
        symbols = [s for s in symbols if not (s in seen or seen.add(s))]
        
        print(f"\n📊 Scanning {len(symbols)} symbol(s): {', '.join(symbols)}")
        print("=" * 100)
        
        # Scan
        all_iron_condors = []
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] Scanning {symbol}...")
            try:
                condors = scan_iron_condors(symbol, target_dte=30, max_results=50)
                all_iron_condors.extend(condors)
                print(f"✅ Found {len(condors)} iron condor(s)")
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        # Filter
        if all_iron_condors:
            print(f"\n✅ Found {len(all_iron_condors)} total iron condor(s)")
            
            # First, filter to top 2 R/R per symbol
            print("\n📊 Filtering to top 2 R/R ratio condors per symbol...")
            top_rr_filtered = filter_top_rr_by_symbol(all_iron_condors, top_n=2)
            print(f"   After top R/R filter: {len(top_rr_filtered)} condors")
            
            # Then apply additional filters if any
            filters = get_filter_inputs()
            
            if filters:
                filtered = filter_iron_condors(top_rr_filtered, filters)
                print(f"\n📊 After additional filtering: {len(filtered)} match criteria")
                if filtered:
                    display_iron_condors_table(filtered)
                else:
                    print("⚠️  No condors match filters. Show top R/R only? (yes/no): ", end="")
                    if input().strip().lower() in ['yes', 'y']:
                        display_iron_condors_table(top_rr_filtered)
            else:
                display_iron_condors_table(top_rr_filtered)
            
            # Prompt for order placement
            if top_rr_filtered or (filters and filtered):
                results_to_trade = filtered if (filters and filtered) else top_rr_filtered
                
                if results_to_trade:
                    print("\n" + "="*100)
                    choice = input("Enter rank number to place order (or Enter to skip): ").strip()
                    
                    if choice:
                        try:
                            rank = int(choice)
                            if 1 <= rank <= len(results_to_trade):
                                selected_condor = results_to_trade[rank - 1]
                                
                                print(f"\nSelected: Rank {rank} - {selected_condor['symbol']}")
                                print(f"  Expiration: {selected_condor['expiration_date'].strftime('%Y-%m-%d')}")
                                print(f"  Credit: ${selected_condor['credit_received']:.2f} | Max Loss: ${selected_condor['max_loss']:.2f}")
                                print(f"  R/R: {selected_condor['rr_ratio']:.2f}:1")
                                
                                qty_input = input("\nEnter quantity (number of condors): ").strip()
                                try:
                                    qty = int(qty_input)
                                    if qty <= 0:
                                        raise ValueError
                                    
                                    confirm = input(f"\nConfirm placing {qty} condor(s) as MLEG order? (yes/no): ").strip().lower()
                                    if confirm in ['yes', 'y']:
                                        place_iron_condor_mleg_order(selected_condor, qty)
                                    else:
                                        print("Order cancelled.")
                                except ValueError:
                                    print("❌ Invalid quantity")
                            else:
                                print(f"❌ Invalid rank. Must be between 1 and {len(results_to_trade)}")
                        except ValueError:
                            print("❌ Invalid input")
        else:
            print("\n⚠️  No iron condors found")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

