"""
Script to check Alpaca account information, balances, and investigate order failures.
"""
import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, GetOrderByIdRequest
from alpaca.trading.enums import QueryOrderStatus
from alpaca.common.exceptions import APIError
from models import get_db_session, IronCondor, Order, CondorStatus
from datetime import datetime, timedelta

# Load environment variables from .env file
load_dotenv()

# Get Alpaca credentials from environment variables
API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
ENV = os.getenv("APCA_ENV", "live")  # Default to "live" if not set

# Determine if we're using paper trading (paper=True) or live trading (paper=False)
paper = ENV.lower() != "live"

# Initialize the TradingClient globally
client = TradingClient(
    api_key=API_KEY,
    secret_key=SECRET_KEY,
    paper=paper
)


def get_account_info():
    """
    Get account information including balances and buying power.
    
    Returns:
        Account object with account details
    """
    try:
        account = client.get_account()
        return account
    except APIError as e:
        print(f"❌ Error fetching account info: {e.message}")
        if e.code:
            print(f"   Error Code: {e.code}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise


def display_account_info(account):
    """
    Display account information in a readable format.
    
    Args:
        account: Account object from Alpaca
    """
    print("\n" + "=" * 80)
    print("ACCOUNT INFORMATION")
    print("=" * 80)
    
    print(f"\n📊 Account Details:")
    print(f"  Account Number: {account.account_number}")
    print(f"  Status: {account.status}")
    print(f"  Trading Blocked: {account.trading_blocked}")
    print(f"  Account Blocked: {account.account_blocked}")
    print(f"  Pattern Day Trader: {account.pattern_day_trader}")
    
    print(f"\n💰 Balances:")
    print(f"  Cash: ${float(account.cash):,.2f}")
    print(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")
    print(f"  Equity: ${float(account.equity):,.2f}")
    print(f"  Long Market Value: ${float(account.long_market_value):,.2f}")
    print(f"  Short Market Value: ${float(account.short_market_value):,.2f}")
    
    print(f"\n💵 Buying Power:")
    print(f"  Buying Power: ${float(account.buying_power):,.2f}")
    print(f"  Day Trading Buying Power: ${float(account.daytrading_buying_power):,.2f}")
    print(f"  Reg T Buying Power: ${float(account.regt_buying_power):,.2f}")
    
    print(f"\n📈 Margin & Risk:")
    print(f"  Initial Margin: ${float(account.initial_margin):,.2f}")
    print(f"  Maintenance Margin: ${float(account.maintenance_margin):,.2f}")
    
    # These attributes may not exist for all account types
    daytrading_bp_used = getattr(account, 'daytrading_buying_power_used', None)
    regt_bp_used = getattr(account, 'regt_buying_power_used', None)
    
    if daytrading_bp_used is not None:
        print(f"  Day Trading Buying Power Used: ${float(daytrading_bp_used):,.2f}")
    if regt_bp_used is not None:
        print(f"  Reg T Buying Power Used: ${float(regt_bp_used):,.2f}")
    
    # Calculate margin usage
    if float(account.equity) > 0:
        margin_usage_pct = (float(account.initial_margin) / float(account.equity)) * 100
        print(f"  Margin Usage: {margin_usage_pct:.2f}%")
    
    # Calculate available buying power
    if regt_bp_used is not None:
        available_bp = float(account.buying_power) - float(regt_bp_used)
        print(f"  Available Buying Power: ${available_bp:,.2f}")
    else:
        # If regt_bp_used is not available, use buying_power as available
        print(f"  Available Buying Power: ${float(account.buying_power):,.2f}")
    
    print(f"\n⚠️  Warnings:")
    if account.trading_blocked:
        print(f"  ⛔ Trading is BLOCKED")
    if account.account_blocked:
        print(f"  ⛔ Account is BLOCKED")
    if float(account.buying_power) < 0:
        print(f"  ⛔ Negative buying power!")
    if float(account.maintenance_margin) > float(account.equity):
        print(f"  ⛔ Maintenance margin call risk!")
    
    print("=" * 80)


def get_recent_orders(days=7):
    """
    Get recent orders from Alpaca.
    
    Args:
        days: Number of days to look back
        
    Returns:
        List of Order objects
    """
    try:
        # Get orders from the last N days
        request = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            nested=True,
            limit=100
        )
        orders = client.get_orders(filter=request)
        
        # Filter by date if needed
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_orders = []
        for order in orders:
            if order.submitted_at and order.submitted_at.replace(tzinfo=None) >= cutoff_date:
                recent_orders.append(order)
        
        return recent_orders
    except APIError as e:
        print(f"❌ Error fetching orders: {e.message}")
        if e.code:
            print(f"   Error Code: {e.code}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise


def get_order_by_id(order_id):
    """
    Get a specific order by ID.
    
    Args:
        order_id: Alpaca order ID
        
    Returns:
        Order object or None
    """
    try:
        request = GetOrderByIdRequest(nested=True)
        order = client.get_order_by_id(order_id, filter=request)
        return order
    except APIError as e:
        print(f"❌ Error fetching order {order_id}: {e.message}")
        if e.code:
            print(f"   Error Code: {e.code}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


def display_order_details(order):
    """
    Display detailed order information including any errors.
    
    Args:
        order: Order object from Alpaca
    """
    print(f"\n  Order ID: {order.id}")
    print(f"  Status: {order.status}")
    print(f"  Symbol: {order.symbol if hasattr(order, 'symbol') and order.symbol else 'N/A (MLEG)'}")
    print(f"  Order Class: {order.order_class.value if hasattr(order, 'order_class') and order.order_class else 'N/A'}")
    print(f"  Type: {order.type}")
    print(f"  Side: {order.side if hasattr(order, 'side') and order.side else 'N/A'}")
    print(f"  Quantity: {order.qty}")
    print(f"  Filled Qty: {order.filled_qty or 0}")
    print(f"  Limit Price: ${order.limit_price if order.limit_price else 'N/A'}")
    print(f"  Stop Price: ${order.stop_price if order.stop_price else 'N/A'}")
    print(f"  Time in Force: {order.time_in_force}")
    print(f"  Submitted At: {order.submitted_at}")
    print(f"  Filled At: {order.filled_at if hasattr(order, 'filled_at') and order.filled_at else 'N/A'}")
    print(f"  Filled Avg Price: ${order.filled_avg_price if order.filled_avg_price else 'N/A'}")
    
    # Check for error messages
    if hasattr(order, 'error') and order.error:
        print(f"  ⚠️  Error: {order.error}")
    if hasattr(order, 'reject_reason') and order.reject_reason:
        print(f"  ⚠️  Reject Reason: {order.reject_reason}")
    if hasattr(order, 'canceled_at') and order.canceled_at:
        print(f"  ⚠️  Canceled At: {order.canceled_at}")
    if hasattr(order, 'expired_at') and order.expired_at:
        print(f"  ⚠️  Expired At: {order.expired_at}")
    
    # Display legs for MLEG orders
    if hasattr(order, 'legs') and order.legs:
        print(f"  Legs ({len(order.legs)}):")
        for i, leg in enumerate(order.legs, 1):
            print(f"    Leg {i}: {leg.symbol} - {leg.side} {leg.qty}")


def check_failed_orders(orders):
    """
    Check for failed, rejected, or canceled orders.
    
    Args:
        orders: List of Order objects
    """
    failed_statuses = ['rejected', 'canceled', 'expired', 'pending_cancel']
    
    failed_orders = []
    for order in orders:
        status = str(order.status.value if hasattr(order.status, 'value') else order.status).lower()
        if status in failed_statuses:
            failed_orders.append(order)
    
    if failed_orders:
        print(f"\n⚠️  Found {len(failed_orders)} failed/rejected/canceled order(s):")
        print("-" * 80)
        for order in failed_orders:
            display_order_details(order)
            print("-" * 80)
    else:
        print("\n✅ No failed orders found in recent history")


def check_pending_orders():
    """
    Check for pending orders that haven't filled.
    """
    try:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True)
        pending_orders = client.get_orders(filter=request)
        
        if pending_orders:
            print(f"\n⏳ Found {len(pending_orders)} pending order(s):")
            print("-" * 80)
            for order in pending_orders:
                display_order_details(order)
                print("-" * 80)
        else:
            print("\n✅ No pending orders")
    except Exception as e:
        print(f"\n⚠️  Error checking pending orders: {e}")


def check_database_orders():
    """
    Check orders from the database and their current status in Alpaca.
    """
    try:
        with get_db_session() as session:
            # Get recent pending/open condors
            recent_condors = session.query(IronCondor).filter(
                IronCondor.status.in_([CondorStatus.PENDING_OPEN, CondorStatus.OPEN])
            ).order_by(IronCondor.created_at.desc()).limit(10).all()
            
            if recent_condors:
                print(f"\n📋 Recent Iron Condors from Database:")
                print("-" * 80)
                for condor in recent_condors:
                    print(f"\nCondor ID: {condor.id}")
                    print(f"  Symbol: {condor.underlying_symbol}")
                    print(f"  Status: {condor.status.value}")
                    print(f"  Opening Order ID: {condor.opening_order_id}")
                    print(f"  Created: {condor.created_at}")
                    
                    if condor.opening_order_id:
                        print(f"\n  Checking order status in Alpaca...")
                        order = get_order_by_id(condor.opening_order_id)
                        if order:
                            display_order_details(order)
                        else:
                            print(f"  ⚠️  Could not fetch order from Alpaca")
                    print("-" * 80)
            else:
                print("\n✅ No recent pending/open condors in database")
                
    except Exception as e:
        print(f"\n⚠️  Error checking database orders: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        print("=" * 80)
        print("ACCOUNT & ORDER DIAGNOSTICS")
        print("=" * 80)
        
        # Get and display account info
        account = get_account_info()
        display_account_info(account)
        
        # Check pending orders
        check_pending_orders()
        
        # Check recent orders for failures
        print("\n" + "=" * 80)
        print("RECENT ORDER HISTORY (Last 7 Days)")
        print("=" * 80)
        recent_orders = get_recent_orders(days=7)
        print(f"\nFound {len(recent_orders)} order(s) in the last 7 days")
        check_failed_orders(recent_orders)
        
        # Check database orders
        print("\n" + "=" * 80)
        print("DATABASE ORDER STATUS CHECK")
        print("=" * 80)
        check_database_orders()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise

