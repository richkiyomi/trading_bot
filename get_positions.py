"""
Script to get open positions from Alpaca using the alpaca-py SDK.
"""
import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.models import Position
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, PositionSide
from models import get_db_session, IronCondor, IronCondorLeg, CondorStatus

# Load environment variables from .env file
load_dotenv()

# Get Alpaca credentials from environment variables
API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
ENV = os.getenv("APCA_ENV", "live")  # Default to "live" if not set

# Determine if we're using paper trading (paper=True) or live trading (paper=False)
# APCA_ENV should be "paper" for paper trading, "live" for live trading
paper = ENV.lower() != "live"

# Initialize the TradingClient globally
client = TradingClient(
    api_key=API_KEY,
    secret_key=SECRET_KEY,
    paper=paper
)

def get_open_positions():
    """
    Get all open positions from Alpaca.
    
    Returns:
        List[Position]: List of open positions
    """
    # Get all open positions
    positions = client.get_all_positions()
    
    return positions

def get_condor_info_for_positions(positions):
    """
    Query database to identify which positions are part of iron condors.
    
    Args:
        positions: List of Position objects
        
    Returns:
        Dict mapping position symbol to condor info (or None if not part of condor)
    """
    condor_info = {}
    
    if not positions:
        return condor_info
    
    try:
        with get_db_session() as session:
            # First, check all condors to see what's in the database
            all_condors = session.query(IronCondor).all()
            print(f"\n🔍 Debug: Total condors in database: {len(all_condors)}")
            for c in all_condors:
                print(f"  Condor ID {c.id}: {c.underlying_symbol}, Status: {c.status.value}")
            
            # Get all option symbols from open iron condors
            open_condors = session.query(IronCondor).filter(
                IronCondor.status.in_([CondorStatus.PENDING_OPEN, CondorStatus.OPEN])
            ).all()
            
            print(f"\n🔍 Debug: Found {len(open_condors)} open condor(s) in database")
            
            # Build a map of option symbol -> condor info
            symbol_to_condor = {}
            for condor in open_condors:
                print(f"  Condor ID {condor.id}: {condor.underlying_symbol}, Status: {condor.status.value}, Legs: {len(condor.legs)}")
                for leg in condor.legs:
                    symbol_to_condor[leg.option_symbol] = {
                        'condor_id': condor.id,
                        'underlying': condor.underlying_symbol,
                        'status': condor.status.value,
                        'leg_role': leg.leg_role.value,
                        'expiration': condor.expiration_date.strftime('%Y-%m-%d'),
                    }
                    print(f"    Leg: {leg.option_symbol} ({leg.leg_role.value})")
            
            print(f"\n🔍 Debug: Position symbols from Alpaca:")
            # Match positions to condors
            for position in positions:
                symbol = position.symbol
                print(f"  Position: {symbol}")
                if symbol in symbol_to_condor:
                    condor_info[symbol] = symbol_to_condor[symbol]
                    print(f"    ✅ Matched to Condor ID {symbol_to_condor[symbol]['condor_id']}")
                else:
                    condor_info[symbol] = None
                    print(f"    ❌ No match found")
                    # Try case-insensitive match
                    for db_symbol in symbol_to_condor.keys():
                        if db_symbol.upper() == symbol.upper():
                            print(f"    ⚠️  Found case-insensitive match: {db_symbol}")
                    
    except Exception as e:
        print(f"Warning: Could not query database for condor info: {e}")
        import traceback
        traceback.print_exc()
        # If DB query fails, mark all as None (unknown)
        for position in positions:
            condor_info[position.symbol] = None
    
    return condor_info


def display_positions(positions):
    """
    Display positions in a readable format, indicating which are part of iron condors.
    
    Args:
        positions: List of Position objects
    """
    if not positions:
        print("No open positions found.")
        return
    
    # Get condor information for all positions
    condor_info = get_condor_info_for_positions(positions)
    
    # Separate condor positions from standalone positions
    condor_positions = []
    standalone_positions = []
    
    for position in positions:
        if condor_info.get(position.symbol):
            condor_positions.append(position)
        else:
            standalone_positions.append(position)
    
    print(f"\nFound {len(positions)} open position(s):")
    print(f"  - {len(condor_positions)} Iron Condor leg(s)")
    print(f"  - {len(standalone_positions)} Standalone position(s)\n")
    print("=" * 80)
    
    # Display condor positions first
    if condor_positions:
        print("\n📊 IRON CONDOR POSITIONS:")
        print("-" * 80)
        for i, position in enumerate(condor_positions, 1):
            info = condor_info[position.symbol]
            print(f"Position {i} (Iron Condor Leg):")
            print(f"  Symbol: {position.symbol}")
            print(f"  Leg Role: {info['leg_role']}")
            print(f"  Underlying: {info['underlying']}")
            print(f"  Condor ID: {info['condor_id']}")
            print(f"  Condor Status: {info['status']}")
            print(f"  Expiration: {info['expiration']}")
            print(f"  Quantity: {position.qty}")
            print(f"  Side: {position.side}")
            print(f"  Market Value: ${float(position.market_value):,.2f}")
            print(f"  Average Entry Price: ${float(position.avg_entry_price):,.2f}")
            print(f"  Current Price: ${float(position.current_price):,.2f}")
            print(f"  Unrealized P/L: ${float(position.unrealized_pl):,.2f}")
            print(f"  Unrealized P/L %: {float(position.unrealized_plpc):.2%}")
            print("-" * 80)
    
    # Display standalone positions
    if standalone_positions:
        print("\n📈 STANDALONE POSITIONS:")
        print("-" * 80)
        for i, position in enumerate(standalone_positions, 1):
            print(f"Position {i}:")
            print(f"  Symbol: {position.symbol}")
            print(f"  Quantity: {position.qty}")
            print(f"  Side: {position.side}")
            print(f"  Market Value: ${float(position.market_value):,.2f}")
            print(f"  Average Entry Price: ${float(position.avg_entry_price):,.2f}")
            print(f"  Current Price: ${float(position.current_price):,.2f}")
            print(f"  Unrealized P/L: ${float(position.unrealized_pl):,.2f}")
            print(f"  Unrealized P/L %: {float(position.unrealized_plpc):.2%}")
            print("-" * 80)
    
    print("=" * 80)

def select_position(positions):
    """
    Prompt user to select a position by number or symbol.
    
    Args:
        positions: List of Position objects
        
    Returns:
        Position: Selected position or None if invalid
    """
    if not positions:
        return None
    
    while True:
        selection = input("\nSelect a position (enter position number or symbol): ").strip()
        
        # Try to parse as position number
        try:
            pos_num = int(selection)
            if 1 <= pos_num <= len(positions):
                return positions[pos_num - 1]
            else:
                print(f"Invalid position number. Please enter a number between 1 and {len(positions)}.")
                continue
        except ValueError:
            # Not a number, try to match by symbol
            pass
        
        # Try to match by symbol (case-insensitive)
        symbol_match = None
        for pos in positions:
            if pos.symbol.upper() == selection.upper():
                symbol_match = pos
                break
        
        if symbol_match:
            return symbol_match
        else:
            print(f"Position '{selection}' not found. Please try again.")

def get_order_type():
    """
    Prompt user to select order type (market or limit).
    
    Returns:
        str: 'market' or 'limit'
    """
    while True:
        order_type = input("\nClose at market or limit? (market/limit): ").strip().lower()
        if order_type in ['market', 'm']:
            return 'market'
        elif order_type in ['limit', 'l']:
            return 'limit'
        else:
            print("Invalid choice. Please enter 'market' or 'limit'.")

def get_limit_price():
    """
    Prompt user for limit price.
    
    Returns:
        float: Limit price
    """
    while True:
        try:
            price = float(input("Enter limit price: ").strip())
            if price > 0:
                return price
            else:
                print("Price must be greater than 0.")
        except ValueError:
            print("Invalid price. Please enter a valid number.")

def close_position(position, order_type='market', limit_price=None):
    """
    Close a position using market or limit order.
    
    Args:
        position: Position object to close
        order_type: 'market' or 'limit'
        limit_price: Limit price (required if order_type is 'limit')
        
    Returns:
        Order: The submitted order
    """
    # Determine the side: if position is long, we need to sell; if short, we need to buy
    if position.side == PositionSide.LONG:
        side = OrderSide.SELL
    else:
        side = OrderSide.BUY
    
    qty = abs(float(position.qty))
    
    if order_type == 'market':
        # Create market order
        order_request = MarketOrderRequest(
            symbol=position.symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )
    else:
        # Create limit order
        if limit_price is None:
            raise ValueError("limit_price is required for limit orders")
        
        order_request = LimitOrderRequest(
            symbol=position.symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price
        )
    
    # Submit the order
    order = client.submit_order(order_request)
    return order

if __name__ == "__main__":
    try:
        print("Fetching open positions from Alpaca...")
        positions = get_open_positions()
        display_positions(positions)
        
        if positions:
            # Prompt to select a position
            selected_position = select_position(positions)
            
            if selected_position:
                print(f"\nSelected position: {selected_position.symbol}")
                print(f"  Quantity: {selected_position.qty}")
                print(f"  Side: {selected_position.side}")
                print(f"  Current Price: ${float(selected_position.current_price):,.2f}")
                
                # Prompt for order type
                order_type = get_order_type()
                
                limit_price = None
                if order_type == 'limit':
                    limit_price = get_limit_price()
                    print(f"\nClosing position with limit order at ${limit_price:,.2f}")
                else:
                    print("\nClosing position with market order")
                
                # Confirm before closing
                confirm = input("\nConfirm closing this position? (yes/no): ").strip().lower()
                if confirm in ['yes', 'y']:
                    order = close_position(selected_position, order_type, limit_price)
                    print(f"\nOrder submitted successfully!")
                    print(f"  Order ID: {order.id}")
                    print(f"  Symbol: {order.symbol}")
                    print(f"  Quantity: {order.qty}")
                    print(f"  Side: {order.side}")
                    print(f"  Type: {order.type}")
                    if order_type == 'limit':
                        print(f"  Limit Price: ${order.limit_price:,.2f}")
                    print(f"  Status: {order.status}")
                else:
                    print("Order cancelled.")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

