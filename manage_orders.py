"""
Script to fetch and cancel open orders from Alpaca.
"""
import os
import sys
import re
from datetime import datetime
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.models import Order
from alpaca.trading.requests import GetOrdersRequest, GetOrderByIdRequest
from alpaca.trading.enums import QueryOrderStatus
from alpaca.common.exceptions import APIError

# Load environment variables
load_dotenv()

# Get Alpaca credentials
API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
ENV = os.getenv("APCA_ENV", "live")
paper = ENV.lower() != "live"

# Initialize TradingClient
client = TradingClient(api_key=API_KEY, secret_key=SECRET_KEY, paper=paper)

def get_open_orders():
    """
    Get all open orders from Alpaca.
    
    Returns:
        List[Order]: List of open orders with legs populated for MLEG orders
    """
    try:
        # Use nested=True to get legs for MLEG orders
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True)
        orders = client.get_orders(filter=request)
        
        # For MLEG orders, if legs aren't populated, try fetching individually
        orders_with_legs = []
        for order in orders:
            # Check if it's MLEG and doesn't have legs
            if (hasattr(order, 'order_class') and order.order_class and 
                str(order.order_class.value if hasattr(order.order_class, 'value') else order.order_class).lower() == 'mleg'):
                if not (hasattr(order, 'legs') and order.legs):
                    # Try to fetch with nested=True
                    try:
                        nested_request = GetOrderByIdRequest(nested=True)
                        full_order = client.get_order_by_id(order.id, filter=nested_request)
                        orders_with_legs.append(full_order)
                    except:
                        orders_with_legs.append(order)
                else:
                    orders_with_legs.append(order)
            else:
                orders_with_legs.append(order)
        
        return orders_with_legs
    except APIError as e:
        print(f"\n❌ Error fetching orders:")
        print(f"   Code: {e.code}")
        print(f"   Message: {e.message}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise

def display_orders(orders):
    """
    Display orders in a readable format.
    
    Args:
        orders: List of Order objects
    """
    if not orders:
        print("\n✅ No open orders found.")
        return
    
    print(f"\n{'='*120}")
    print(f"OPEN ORDERS ({len(orders)} total)")
    print(f"{'='*120}\n")
    
    header = (f"{'#':<4} {'Symbol':<12} {'Credit':<10} {'Order ID':<38} {'Type':<10} {'Qty':<8} "
              f"{'Filled':<8} {'Status':<12} {'Submitted':<20}")
    print(header)
    print("-" * 140)
    
    for i, order in enumerate(orders, 1):
        order_id = str(order.id)[:36] + "..." if len(str(order.id)) > 36 else str(order.id)
        
        # Get symbol - extract underlying symbol from option symbols or use main symbol
        symbol = order.symbol or "N/A"
        legs = None  # Initialize legs variable
        
        # For MLEG orders, the symbol is None, so we need to extract from legs
        # Check if this is an MLEG order
        is_mleg = False
        if hasattr(order, 'order_class') and order.order_class:
            order_class = str(order.order_class.value) if hasattr(order.order_class, 'value') else str(order.order_class)
            is_mleg = (order_class == "MLEG" or order_class.lower() == "mleg")
        
        if is_mleg:
            # Try to get legs from the order - check multiple possible attributes
            legs = None
            
            # Try order.legs (list) - most common
            if hasattr(order, 'legs') and order.legs:
                legs = order.legs
            
            # Try order.leg (single or list)
            if not legs and hasattr(order, 'leg') and order.leg:
                legs = order.leg if isinstance(order.leg, list) else [order.leg]
            
            if legs and len(legs) > 0:
                # Extract underlying symbol from first leg (e.g., "MSFT251226P00380000" -> "MSFT")
                for leg in legs:
                    leg_symbol = None
                    
                    # Try different ways to access leg symbol
                    if hasattr(leg, 'symbol') and leg.symbol:
                        leg_symbol = leg.symbol
                    elif isinstance(leg, dict) and 'symbol' in leg:
                        leg_symbol = leg['symbol']
                    elif hasattr(leg, '__dict__'):
                        leg_dict = leg.__dict__
                        leg_symbol = leg_dict.get('symbol')
                    
                    if leg_symbol:
                        # Option symbols are in OCC format: SYMBOL + YYMMDD + C/P + STRIKE
                        # Extract first part (symbol) - can be 1-5 characters
                        leg_symbol_str = str(leg_symbol).upper().strip()
                        # Match: 1-5 uppercase letters at the start
                        match = re.match(r'^([A-Z]{1,5})', leg_symbol_str)
                        if match:
                            symbol = match.group(1)
                            break
        
        # Calculate credit/debit from limit_price
        # For MLEG credit orders, limit_price is negative (credit to receive)
        # For debit orders, limit_price is positive (debit to pay)
        credit_display = "N/A"
        if order.limit_price:
            try:
                limit_price_val = float(order.limit_price)
                if limit_price_val < 0:
                    # Negative = credit (money received)
                    credit_display = f"${abs(limit_price_val):.2f} CR"
                elif limit_price_val > 0:
                    # Positive = debit (money paid)
                    credit_display = f"${limit_price_val:.2f} DB"
                else:
                    credit_display = "$0.00"
            except (ValueError, TypeError):
                credit_display = str(order.limit_price)
        
        order_type = str(order.type.value) if order.type else "N/A"
        
        # Handle qty - might be string or number
        if order.qty:
            try:
                qty = str(float(order.qty))
            except (ValueError, TypeError):
                qty = str(order.qty)
        else:
            qty = "N/A"
        
        # Handle filled_qty - might be string or number
        if order.filled_qty:
            try:
                filled_qty = str(float(order.filled_qty))
            except (ValueError, TypeError):
                filled_qty = str(order.filled_qty)
        else:
            filled_qty = "0"
        
        status = str(order.status.value) if hasattr(order.status, 'value') else str(order.status)
        
        # Format submitted time
        if order.submitted_at:
            submitted_str = order.submitted_at.strftime('%Y-%m-%d %H:%M:%S')
        else:
            submitted_str = "N/A"
        
        row = (f"{i:<4} {symbol:<12} {credit_display:<10} {order_id:<38} {order_type:<10} {qty:<8} "
               f"{filled_qty:<8} {status:<12} {submitted_str:<20}")
        print(row)
        
        # Show additional details for MLEG orders (legs already extracted above)
        if legs:
            print(f"     └─ MLEG Order (Multi-leg)")
            for leg in legs:
                leg_symbol = "N/A"
                leg_side = "N/A"
                
                if hasattr(leg, 'symbol'):
                    leg_symbol = leg.symbol or "N/A"
                elif isinstance(leg, dict):
                    leg_symbol = leg.get('symbol', 'N/A')
                
                if hasattr(leg, 'side'):
                    if leg.side:
                        leg_side = str(leg.side.value) if hasattr(leg.side, 'value') else str(leg.side)
                elif isinstance(leg, dict):
                    leg_side = str(leg.get('side', 'N/A'))
                
                print(f"        • {leg_side} {leg_symbol}")
    
    print("-" * 120)

def select_order(orders):
    """
    Prompt user to select an order by number or order ID.
    
    Args:
        orders: List of Order objects
        
    Returns:
        Order: Selected order or None if invalid
    """
    if not orders:
        return None
    
    while True:
        selection = input("\nSelect an order (enter order number or order ID): ").strip()
        
        # Try to parse as order number
        try:
            order_num = int(selection)
            if 1 <= order_num <= len(orders):
                return orders[order_num - 1]
            else:
                print(f"Invalid order number. Please enter a number between 1 and {len(orders)}.")
                continue
        except ValueError:
            # Not a number, try to match by order ID
            pass
        
        # Try to match by order ID (partial match is OK)
        order_match = None
        for order in orders:
            order_id_str = str(order.id)
            if order_id_str == selection or order_id_str.startswith(selection):
                order_match = order
                break
        
        if order_match:
            return order_match
        else:
            print(f"Order '{selection}' not found. Please try again.")

def cancel_order(order):
    """
    Cancel a specific order.
    
    Args:
        order: Order object to cancel
    """
    try:
        print(f"\nCancelling order...")
        print(f"  Order ID: {order.id}")
        print(f"  Symbol: {order.symbol or 'N/A'}")
        print(f"  Type: {order.type}")
        print(f"  Side: {order.side}")
        print(f"  Quantity: {order.qty}")
        print(f"  Status: {order.status}")
        
        client.cancel_order_by_id(order.id)
        print(f"\n✅ Order cancelled successfully!")
        
    except APIError as e:
        print(f"\n❌ Error cancelling order:")
        print(f"   Code: {e.code}")
        print(f"   Message: {e.message}")
        if "not found" in e.message.lower() or "404" in str(e.status_code):
            print(f"   The order may have already been filled or cancelled.")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise

def cancel_all_orders():
    """
    Cancel all open orders.
    """
    try:
        print(f"\n⚠️  Cancelling ALL open orders...")
        results = client.cancel_orders()
        
        if results:
            print(f"\n✅ Cancellation results:")
            for result in results:
                status = result.status if hasattr(result, 'status') else "Unknown"
                order_id = result.id if hasattr(result, 'id') else "Unknown"
                print(f"   Order {order_id}: Status {status}")
        else:
            print(f"\n✅ All orders cancelled (or no orders to cancel)")
            
    except APIError as e:
        print(f"\n❌ Error cancelling orders:")
        print(f"   Code: {e.code}")
        print(f"   Message: {e.message}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise

if __name__ == "__main__":
    try:
        print("=" * 100)
        print("ALPACA ORDER MANAGER")
        print("=" * 100)
        
        print("\nFetching open orders...")
        orders = get_open_orders()
        display_orders(orders)
        
        if orders:
            print("\nOptions:")
            print("  1. Cancel a specific order (enter order number or ID)")
            print("  2. Cancel all orders (type 'all')")
            print("  3. Exit (press Enter)")
            
            choice = input("\nYour choice: ").strip().lower()
            
            if choice == 'all':
                confirm = input("\n⚠️  Are you sure you want to cancel ALL open orders? (yes/no): ").strip().lower()
                if confirm in ['yes', 'y']:
                    cancel_all_orders()
                else:
                    print("Cancelled.")
            elif choice:
                selected_order = select_order(orders)
                if selected_order:
                    print(f"\nSelected order:")
                    print(f"  Order ID: {selected_order.id}")
                    print(f"  Symbol: {selected_order.symbol or 'N/A'}")
                    print(f"  Type: {selected_order.type}")
                    print(f"  Side: {selected_order.side}")
                    print(f"  Quantity: {selected_order.qty}")
                    print(f"  Filled: {selected_order.filled_qty or 0}")
                    if selected_order.limit_price:
                        print(f"  Limit Price: ${selected_order.limit_price:.2f}")
                    print(f"  Status: {selected_order.status}")
                    
                    confirm = input("\nConfirm cancelling this order? (yes/no): ").strip().lower()
                    if confirm in ['yes', 'y']:
                        cancel_order(selected_order)
                    else:
                        print("Cancellation cancelled.")
        else:
            print("\nNo open orders to manage.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user.")
        sys.exit(0)
    except APIError as e:
        print(f"\n❌ API Error: {e.message}")
        if e.code:
            print(f"   Error Code: {e.code}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

