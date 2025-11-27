"""
Modal script to monitor iron condor positions and execute stop-loss/take-profit closes.
Runs 3-5 times per day during market hours.

Features:
- Monitors open iron condor positions
- Checks if opening orders have filled
- Calculates current unrealized P/L
- Executes closes when profit target (70% of max profit) or stop loss (70% of max loss) is breached
- Sends Telegram notifications for order fills and position closes
"""
import os
import json
import requests
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING

# Modal imports - must be at top level
import modal

# Database models - try to import, but don't fail if not available locally
# (Modal will have them available at runtime)
try:
    from models import (
        get_db_session, IronCondor, IronCondorLeg, Order, PositionSnapshot,
        CondorStatus, ExitReason, LegRole, OrderSide as DBOrderSide,
        OrderType as DBOrderType, TimeInForce as DBTimeInForce
    )
except ImportError:
    # Models not available locally - will be available in Modal image
    # Define placeholders for type checking
    get_db_session = None
    IronCondor = None
    IronCondorLeg = None
    Order = None
    PositionSnapshot = None
    CondorStatus = None
    ExitReason = None
    LegRole = None
    DBOrderSide = None
    DBOrderType = None
    DBTimeInForce = None

# Alpaca imports - will be available in Modal image, use lazy imports
if TYPE_CHECKING:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest
    from alpaca.trading.enums import QueryOrderStatus, OrderClass, OrderType, TimeInForce
    from alpaca.common.exceptions import APIError
    from sqlalchemy import and_

# Modal app setup
app = modal.App("iron-condor-monitor")

# Image with dependencies - install packages directly
# Include models.py by adding it to the image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "alpaca-py>=0.20.0",
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "fastapi>=0.100.0",
    )
    .add_local_file("models.py", "/root/models.py")  # Add to working directory
)

# Secrets
# Modal secrets should contain:
# - telegram-creds: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
# - alpaca-creds: APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_ENV
# - postgres-secret: DATABASE_HOST, DATABASE_NAME, DATABASE_PORT, DATABASE_USER, DATABASE_PASSWORD
secrets = [
    modal.Secret.from_name("telegram-creds"),
    modal.Secret.from_name("alpaca-creds"),
    modal.Secret.from_name("postgres-secret"),
]

# Configuration
PROFIT_TARGET_PCT = 0.70  # 70% of max profit
STOP_LOSS_PCT = 0.70      # 70% of max loss




def get_telegram_creds() -> Tuple[str, str]:
    """
    Get Telegram bot token and chat ID from Modal secrets.
    Modal secret 'telegram-creds' should contain:
    - TELEGRAM_BOT_TOKEN
    - TELEGRAM_CHAT_ID
    """
    # Modal automatically injects secrets as environment variables
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in Modal secret 'telegram-creds'"
        )
    
    return bot_token, chat_id


def send_telegram_message(message: str):
    """Send a message via Telegram bot."""
    try:
        bot_token, chat_id = get_telegram_creds()
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"⚠️  Failed to send Telegram message: {e}")
        return False


def get_alpaca_client():
    """Initialize Alpaca TradingClient."""
    from alpaca.trading.client import TradingClient
    
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    env = os.getenv("APCA_ENV", "live")
    paper = env.lower() != "live"
    
    return TradingClient(api_key=api_key, secret_key=secret_key, paper=paper)


def get_current_positions(client) -> Dict[str, Dict]:
    """
    Get all current positions from Alpaca and organize by option symbol.
    
    Returns:
        Dict mapping option symbol to position data
    """
    try:
        positions = client.get_all_positions()
        position_map = {}
        
        for pos in positions:
            # Only track option positions (they have OCC symbols)
            if len(pos.symbol) > 6:  # Options have longer symbols
                position_map[pos.symbol] = {
                    'qty': float(pos.qty),
                    'side': str(pos.side),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'market_value': float(pos.market_value),
                }
        
        return position_map
    except Exception as e:
        print(f"⚠️  Error fetching positions: {e}")
        return {}


def get_order_status(client, order_id: str) -> Optional[Dict]:
    """Get order status from Alpaca."""
    try:
        order = client.get_order_by_id(order_id)
        return {
            'id': str(order.id),
            'status': str(order.status.value) if hasattr(order.status, 'value') else str(order.status),
            'filled_qty': float(order.filled_qty or 0),
            'qty': float(order.qty),
            'filled_at': order.filled_at,
            'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
        }
    except Exception as e:
        print(f"⚠️  Error fetching order {order_id}: {e}")
        return None


def calculate_condor_unrealized_pl(
    condor: IronCondor,
    current_positions: Dict[str, Dict]
) -> Tuple[float, float]:
    """
    Calculate unrealized P/L for an iron condor based on current positions.
    
    Returns:
        (unrealized_pl, unrealized_pl_pct) where pct is relative to max profit/loss
    """
    total_unrealized_pl = 0.0
    legs_found = 0
    
    # Sum up P/L from all 4 legs
    for leg in condor.legs:
        if leg.option_symbol in current_positions:
            pos = current_positions[leg.option_symbol]
            # The unrealized_pl from Alpaca is already for the entire position
            # We need to calculate per contract
            contracts_in_position = abs(pos['qty'])
            if contracts_in_position > 0:
                # P/L per contract
                pl_per_contract = pos['unrealized_pl'] / contracts_in_position
                # Multiply by our leg quantity
                leg_pl = pl_per_contract * leg.quantity
                total_unrealized_pl += leg_pl
                legs_found += 1
    
    # If we couldn't find all legs, return None to indicate incomplete data
    if legs_found < len(condor.legs):
        print(f"⚠️  Warning: Only found {legs_found}/{len(condor.legs)} legs in current positions")
        # Still calculate, but note it may be incomplete
    
    # Calculate percentage relative to max profit or max loss
    if total_unrealized_pl >= 0:
        # Profit: compare to max_profit
        max_value = float(condor.max_profit)
        pct = (total_unrealized_pl / max_value) if max_value > 0 else 0.0
    else:
        # Loss: compare to max_loss
        max_value = float(condor.max_loss)
        pct = abs(total_unrealized_pl / max_value) if max_value > 0 else 0.0
    
    return total_unrealized_pl, pct


def check_exit_conditions(
    condor: IronCondor,
    unrealized_pl: float,
    unrealized_pl_pct: float
) -> Tuple[bool, Optional[ExitReason], str]:
    """
    Check if exit conditions are met.
    
    Returns:
        (should_exit, exit_reason, reason_text)
    """
    profit_target = float(condor.profit_target_pct or PROFIT_TARGET_PCT)
    stop_loss = float(condor.stop_loss_pct or STOP_LOSS_PCT)
    
    if unrealized_pl >= 0:
        # In profit - check profit target
        if unrealized_pl_pct >= profit_target:
            return True, ExitReason.PROFIT_TARGET, f"Profit target reached: {unrealized_pl_pct:.1%} of max profit"
    else:
        # In loss - check stop loss
        if unrealized_pl_pct >= stop_loss:
            return True, ExitReason.STOP_LOSS, f"Stop loss triggered: {unrealized_pl_pct:.1%} of max loss"
    
    return False, None, ""


def place_closing_order(
    client,
    condor: IronCondor,
    current_positions: Dict[str, Dict]
) -> Optional[str]:
    """
    Place a closing MLEG order to close the iron condor.
    
    Returns:
        Alpaca order ID if successful, None otherwise
    """
    from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
    from alpaca.trading.enums import OrderClass, OrderType, TimeInForce
    
    try:
        # Build closing legs (opposite of opening)
        # Opening: Sell Put Short, Buy Put Long, Sell Call Short, Buy Call Long
        # Closing: Buy Put Short, Sell Put Long, Buy Call Short, Sell Call Long
        
        # Order legs in the same order as opening: Put Short, Put Long, Call Short, Call Long
        leg_order = [LegRole.PUT_SHORT, LegRole.PUT_LONG, LegRole.CALL_SHORT, LegRole.CALL_LONG]
        legs = []
        
        for role in leg_order:
            leg = next((l for l in condor.legs if l.leg_role == role), None)
            if not leg:
                print(f"⚠️  Missing leg with role {role}")
                return None
            
            # Determine closing side (opposite of opening)
            if leg.side_open == DBOrderSide.BUY:
                closing_side = "sell"
            else:
                closing_side = "buy"
            
            legs.append(OptionLegRequest(
                symbol=leg.option_symbol,
                side=closing_side,
                ratio_qty=1.0
            ))
        
        # Calculate current market value to estimate closing debit
        # For closing, we want to pay as little as possible
        # Estimate: current market value of the condor
        current_market_value = 0.0
        for leg in condor.legs:
            if leg.option_symbol in current_positions:
                pos = current_positions[leg.option_symbol]
                # Market value per contract
                contracts_in_position = abs(pos['qty'])
                if contracts_in_position > 0:
                    contract_value = pos['market_value'] / contracts_in_position
                    current_market_value += contract_value * leg.quantity
        
        # For MLEG closing order, limit_price should be positive (debit to pay)
        # Use a reasonable buffer above current market value, or minimum
        if current_market_value > 0:
            closing_debit = current_market_value * 1.10  # 10% buffer to ensure fill
        else:
            # Fallback: estimate from entry credit
            closing_debit = float(condor.entry_net_credit) * 0.5  # Rough estimate
        
        closing_debit = max(closing_debit, 0.01)  # Minimum
        
        print(f"  Estimated closing debit: ${closing_debit:.2f}")
        
        # Create limit order
        order_request = LimitOrderRequest(
            qty=float(condor.legs[0].quantity),  # All legs have same quantity
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.MLEG,
            legs=legs,
            limit_price=round(closing_debit, 2),  # Positive = debit to pay
        )
        
        # Submit order
        order = client.submit_order(order_request)
        print(f"  Closing order submitted: {order.id}")
        return str(order.id)
        
    except Exception as e:
        print(f"⚠️  Error placing closing order for condor {condor.id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def update_condor_status(
    session,
    condor: IronCondor,
    new_status: CondorStatus,
    exit_reason: Optional[ExitReason] = None,
    closing_order_id: Optional[str] = None
):
    """Update condor status in database."""
    condor.status = new_status
    if exit_reason:
        condor.exit_reason = exit_reason
    if closing_order_id:
        condor.closing_order_id = closing_order_id
    if new_status == CondorStatus.OPEN and not condor.entry_time:
        condor.entry_time = datetime.utcnow()
    if new_status == CondorStatus.CLOSED:
        condor.exit_time = datetime.utcnow()
    
    session.commit()


def create_position_snapshot(
    session,
    condor: IronCondor,
    unrealized_pl: float,
    unrealized_pl_pct: float,
    profit_target_breached: bool,
    stop_loss_breached: bool
):
    """Create a position snapshot for monitoring."""
    snapshot = PositionSnapshot(
        condor=condor,
        snapshot_time=datetime.utcnow(),
        unrealized_pl=unrealized_pl,
        unrealized_pl_pct=unrealized_pl_pct,
        current_market_value=None,  # Can be calculated if needed
        profit_target_breached=profit_target_breached,
        stop_loss_breached=stop_loss_breached,
    )
    session.add(snapshot)
    session.commit()


@app.function(
    image=image,
    secrets=secrets,
    schedule=modal.Cron("0 15,17,19,20,21 * * 1-5"),  # 10am, 12pm, 2pm, 3pm, 4pm ET (EST = UTC-5) on weekdays
)
def monitor_condors():
    """
    Main monitoring function.
    Runs on schedule and checks all open iron condor positions.
    """
    # Import models (available in Modal image working directory)
    from models import (
        get_db_session, IronCondor, IronCondorLeg, Order, PositionSnapshot,
        CondorStatus, ExitReason, LegRole, OrderSide as DBOrderSide,
        OrderType as DBOrderType, TimeInForce as DBTimeInForce
    )
    
    print(f"\n{'='*80}")
    print(f"Iron Condor Monitor - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'='*80}\n")
    
    client = get_alpaca_client()
    session = get_db_session()
    
    try:
        # Get all open condors from database
        open_condors = session.query(IronCondor).filter(
            IronCondor.status.in_([CondorStatus.PENDING_OPEN, CondorStatus.OPEN])
        ).all()
        
        print(f"Found {len(open_condors)} open condor(s) to monitor\n")
        
        if not open_condors:
            print("No open condors to monitor.")
            return {"status": "success", "message": "No open condors"}
        
        # Get current positions from Alpaca
        current_positions = get_current_positions(client)
        print(f"Current positions in Alpaca: {len(current_positions)} option contracts\n")
        
        notifications_sent = []
        
        for condor in open_condors:
            print(f"\n--- Condor ID {condor.id}: {condor.underlying_symbol} ---")
            print(f"Status: {condor.status}")
            print(f"Expiration: {condor.expiration_date}")
            
            # Check if PENDING_OPEN order has filled
            if condor.status == CondorStatus.PENDING_OPEN and condor.opening_order_id:
                order_status = get_order_status(client, condor.opening_order_id)
                
                if order_status:
                    print(f"Opening order status: {order_status['status']}")
                    
                    # Check if order filled
                    if order_status['status'] in ['filled', 'partially_filled']:
                        if order_status['filled_qty'] >= order_status['qty']:
                            # Fully filled - update to OPEN
                            update_condor_status(session, condor, CondorStatus.OPEN)
                            
                            # Send notification
                            message = (
                                f"✅ <b>Iron Condor Opened</b>\n\n"
                                f"Symbol: {condor.underlying_symbol}\n"
                                f"Expiration: {condor.expiration_date}\n"
                                f"Credit Received: ${float(condor.entry_net_credit):,.2f}\n"
                                f"Max Profit: ${float(condor.max_profit):,.2f}\n"
                                f"Max Loss: ${float(condor.max_loss):,.2f}\n"
                                f"Order ID: {condor.opening_order_id}"
                            )
                            send_telegram_message(message)
                            notifications_sent.append(f"Condor {condor.id} opened")
                            print(f"✅ Condor opened - notification sent")
                            continue
            
            # For OPEN condors, check exit conditions
            if condor.status == CondorStatus.OPEN:
                # Calculate unrealized P/L
                unrealized_pl, unrealized_pl_pct = calculate_condor_unrealized_pl(
                    condor, current_positions
                )
                
                print(f"Unrealized P/L: ${unrealized_pl:,.2f} ({unrealized_pl_pct:.1%})")
                
                # Check exit conditions
                should_exit, exit_reason, reason_text = check_exit_conditions(
                    condor, unrealized_pl, unrealized_pl_pct
                )
                
                # Create snapshot
                profit_breached = exit_reason == ExitReason.PROFIT_TARGET
                loss_breached = exit_reason == ExitReason.STOP_LOSS
                create_position_snapshot(
                    session, condor, unrealized_pl, unrealized_pl_pct,
                    profit_breached, loss_breached
                )
                
                if should_exit:
                    print(f"⚠️  Exit condition met: {reason_text}")
                    
                    # Place closing order
                    closing_order_id = place_closing_order(client, condor, current_positions)
                    
                    if closing_order_id:
                        # Update condor status
                        update_condor_status(
                            session, condor, CondorStatus.CLOSED,
                            exit_reason, closing_order_id
                        )
                        
                        # Calculate realized P/L (approximate)
                        realized_pnl = unrealized_pl  # Will be updated when order fills
                        condor.realized_pnl = realized_pnl
                        session.commit()
                        
                        # Send notification
                        pnl_text = f"+${realized_pnl:,.2f}" if realized_pnl >= 0 else f"-${abs(realized_pnl):,.2f}"
                        message = (
                            f"🔔 <b>Iron Condor Closed</b>\n\n"
                            f"Symbol: {condor.underlying_symbol}\n"
                            f"Expiration: {condor.expiration_date}\n"
                            f"Reason: {exit_reason.value}\n"
                            f"Realized P/L: {pnl_text}\n"
                            f"Closing Order ID: {closing_order_id}"
                        )
                        send_telegram_message(message)
                        notifications_sent.append(f"Condor {condor.id} closed")
                        print(f"✅ Closing order placed - notification sent")
                    else:
                        print(f"❌ Failed to place closing order")
                else:
                    print(f"✓ No exit conditions met")
        
        print(f"\n{'='*80}")
        print(f"Monitoring complete. Notifications sent: {len(notifications_sent)}")
        print(f"{'='*80}\n")
        
        return {
            "status": "success",
            "condors_checked": len(open_condors),
            "notifications_sent": len(notifications_sent)
        }
        
    except Exception as e:
        error_msg = f"Error in monitor_condors: {e}"
        print(f"❌ {error_msg}")
        send_telegram_message(f"⚠️ <b>Monitor Error</b>\n\n{error_msg}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    # For local testing
    with app.run():
        monitor_condors.remote()

