"""
SQLAlchemy models for Iron Condor trading database.
"""
import os
from datetime import datetime, date
from decimal import Decimal
from typing import Optional
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime, Date,
    ForeignKey, Text, Enum as SQLEnum, Numeric
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import enum

# Load environment variables (optional - Modal injects env vars directly)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available (e.g., in Modal) - environment variables should be set directly
    pass

# Database connection
DB_HOST = os.getenv("DATABASE_HOST")
DB_NAME = os.getenv("DATABASE_NAME")
DB_PORT = os.getenv("DATABASE_PORT", "5432")
DB_USER = os.getenv("DATABASE_USER")
DB_PASSWORD = os.getenv("DATABASE_PASSWORD")

# Create connection string
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create engine
engine = create_engine(DATABASE_URL, echo=False)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# Enums
class CondorStatus(str, enum.Enum):
    PENDING_OPEN = "PENDING_OPEN"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"


class ExitReason(str, enum.Enum):
    PROFIT_TARGET = "PROFIT_TARGET"  # Hit 70% of max profit
    STOP_LOSS = "STOP_LOSS"  # Hit 70% of max loss
    EXPIRATION = "EXPIRATION"  # Expired naturally
    MANUAL = "MANUAL"  # Manually closed
    OTHER = "OTHER"


class LegRole(str, enum.Enum):
    PUT_LONG = "PUT_LONG"
    PUT_SHORT = "PUT_SHORT"
    CALL_SHORT = "CALL_SHORT"
    CALL_LONG = "CALL_LONG"


class OrderSide(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, enum.Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(str, enum.Enum):
    DAY = "day"
    GTC = "gtc"
    OPG = "opg"
    CLS = "cls"
    IOC = "ioc"
    FOK = "fok"


# Models
class IronCondor(Base):
    """
    Parent table for iron condor trades.
    Tracks the overall condor from entry to exit.
    """
    __tablename__ = "iron_condors"
    
    id = Column(Integer, primary_key=True, index=True)
    external_tag = Column(String(255), unique=True, nullable=False, index=True)
    
    # Basic info
    underlying_symbol = Column(String(10), nullable=False, index=True)
    expiration_date = Column(Date, nullable=False, index=True)
    dte_at_entry = Column(Integer, nullable=False)
    
    # Status tracking
    status = Column(SQLEnum(CondorStatus), nullable=False, default=CondorStatus.PENDING_OPEN, index=True)
    entry_time = Column(DateTime, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    exit_reason = Column(SQLEnum(ExitReason), nullable=True)
    
    # Order tracking (for monitoring/reconciliation)
    opening_order_id = Column(String(36), nullable=True, index=True)  # Alpaca order ID for opening MLEG order
    closing_order_id = Column(String(36), nullable=True, index=True)  # Alpaca order ID for closing MLEG order
    
    # Financial metrics
    entry_net_credit = Column(Numeric(10, 2), nullable=False)  # Credit received at entry (per condor)
    exit_net_debit = Column(Numeric(10, 2), nullable=True)  # Debit paid at exit (if closed early)
    realized_pnl = Column(Numeric(10, 2), nullable=True)  # Final realized P/L
    
    # Theoretical limits
    max_profit = Column(Numeric(10, 2), nullable=False)  # Theoretical max profit
    max_loss = Column(Numeric(10, 2), nullable=False)  # Theoretical max loss
    
    # Entry conditions
    credit_ratio = Column(Numeric(5, 4), nullable=False)  # Credit / wing width ratio
    wing_width = Column(Numeric(10, 2), nullable=False)  # Total wing width (call + put)
    otm_pct = Column(Numeric(5, 4), nullable=True)  # OTM percentage used
    score = Column(Numeric(10, 2), nullable=True)  # Scanner score
    
    # Exit thresholds (for monitoring)
    profit_target_pct = Column(Numeric(5, 4), default=0.70)  # Default 70% of max profit
    stop_loss_pct = Column(Numeric(5, 4), default=0.70)  # Default 70% of max loss
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    legs = relationship("IronCondorLeg", back_populates="condor", cascade="all, delete-orphan")
    orders = relationship("Order", back_populates="condor", cascade="all, delete-orphan")
    snapshots = relationship("PositionSnapshot", back_populates="condor", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<IronCondor(id={self.id}, symbol={self.underlying_symbol}, status={self.status})>"


class IronCondorLeg(Base):
    """
    Individual legs of an iron condor (4 legs per condor).
    Tracks each option contract in the condor.
    """
    __tablename__ = "iron_condor_legs"
    
    id = Column(Integer, primary_key=True, index=True)
    condor_id = Column(Integer, ForeignKey("iron_condors.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Leg identification
    leg_role = Column(SQLEnum(LegRole), nullable=False)
    option_symbol = Column(String(50), nullable=False)  # Full OCC symbol
    option_type = Column(String(10), nullable=False)  # "PUT" or "CALL"
    strike_price = Column(Numeric(10, 2), nullable=False)
    expiration_date = Column(Date, nullable=False)
    
    # Trade details
    side_open = Column(SQLEnum(OrderSide), nullable=False)  # buy or sell
    side_close = Column(SQLEnum(OrderSide), nullable=True)  # buy or sell (when closed)
    quantity = Column(Integer, nullable=False)
    
    # Pricing
    open_avg_price = Column(Numeric(10, 4), nullable=True)  # Average fill price when opened
    close_avg_price = Column(Numeric(10, 4), nullable=True)  # Average fill price when closed
    realized_pnl = Column(Numeric(10, 2), nullable=True)  # P/L for this leg
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    condor = relationship("IronCondor", back_populates="legs")
    orders = relationship("Order", back_populates="leg")
    
    def __repr__(self):
        return f"<IronCondorLeg(id={self.id}, role={self.leg_role}, symbol={self.option_symbol})>"


class Order(Base):
    """
    All orders (opening and closing) for iron condors.
    Links to both condor and leg for tracking.
    """
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Alpaca order identifiers
    alpaca_order_id = Column(String(36), nullable=False, unique=True, index=True)  # UUID
    client_order_id = Column(String(100), nullable=True, index=True)
    
    # Relationships
    condor_id = Column(Integer, ForeignKey("iron_condors.id", ondelete="CASCADE"), nullable=True, index=True)
    leg_id = Column(Integer, ForeignKey("iron_condor_legs.id", ondelete="SET NULL"), nullable=True, index=True)
    
    # Order classification
    is_opening = Column(Boolean, nullable=False, default=True, index=True)  # True = opening, False = closing
    
    # Order details
    side = Column(SQLEnum(OrderSide), nullable=False)
    order_type = Column(SQLEnum(OrderType), nullable=False)
    time_in_force = Column(SQLEnum(TimeInForce), nullable=False)
    limit_price = Column(Numeric(10, 4), nullable=True)  # For limit orders
    stop_price = Column(Numeric(10, 4), nullable=True)  # For stop orders
    
    # Order status
    status = Column(String(50), nullable=False, index=True)  # pending_new, accepted, filled, etc.
    qty = Column(Integer, nullable=False)
    filled_qty = Column(Integer, nullable=False, default=0)
    avg_fill_price = Column(Numeric(10, 4), nullable=True)
    last_fill_price = Column(Numeric(10, 4), nullable=True)
    
    # Timestamps
    submitted_at = Column(DateTime, nullable=True)
    filled_at = Column(DateTime, nullable=True)
    canceled_at = Column(DateTime, nullable=True)
    expired_at = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    condor = relationship("IronCondor", back_populates="orders")
    leg = relationship("IronCondorLeg", back_populates="orders")
    
    def __repr__(self):
        return f"<Order(id={self.id}, alpaca_id={self.alpaca_order_id}, status={self.status})>"


class PositionSnapshot(Base):
    """
    Periodic snapshots of condor positions for monitoring.
    Used to track P/L over time and trigger exit conditions.
    """
    __tablename__ = "position_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    condor_id = Column(Integer, ForeignKey("iron_condors.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Snapshot timestamp
    snapshot_time = Column(DateTime, nullable=False, index=True)
    
    # Position metrics at snapshot time
    unrealized_pl = Column(Numeric(10, 2), nullable=False)  # Current unrealized P/L
    unrealized_pl_pct = Column(Numeric(5, 4), nullable=False)  # % of max profit/loss
    current_market_value = Column(Numeric(10, 2), nullable=True)  # Total market value
    
    # Threshold checks
    profit_target_breached = Column(Boolean, default=False, nullable=False)
    stop_loss_breached = Column(Boolean, default=False, nullable=False)
    
    # Additional context
    notes = Column(Text, nullable=True)  # Optional notes
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    condor = relationship("IronCondor", back_populates="snapshots")
    
    def __repr__(self):
        return f"<PositionSnapshot(id={self.id}, condor_id={self.condor_id}, time={self.snapshot_time})>"


# Database session helper
def get_db_session():
    """
    Get a database session.
    Use with context manager or ensure to close:
    
    with get_db_session() as session:
        # use session
    """
    return SessionLocal()


def create_tables():
    """
    Create all tables in the database.
    """
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created successfully!")


def drop_tables():
    """
    Drop all tables (USE WITH CAUTION!).
    """
    print("⚠️  Dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    print("✅ Tables dropped!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "drop":
        confirm = input("⚠️  Are you sure you want to drop all tables? (yes/no): ").strip().lower()
        if confirm == "yes":
            drop_tables()
        else:
            print("Cancelled.")
    else:
        create_tables()
        print("\nTables created:")
        print("  - iron_condors")
        print("  - iron_condor_legs")
        print("  - orders")
        print("  - position_snapshots")

