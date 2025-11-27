"""
Test script to verify PostgreSQL database connection.
"""
import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Load environment variables
load_dotenv()

# Get database credentials
DB_HOST = os.getenv("DATABASE_HOST")
DB_NAME = os.getenv("DATABASE_NAME")
DB_PORT = os.getenv("DATABASE_PORT", "5432")
DB_USER = os.getenv("DATABASE_USER")
DB_PASSWORD = os.getenv("DATABASE_PASSWORD")

def test_connection():
    """Test database connection."""
    # Validate all required variables are present
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
        print("❌ Missing required database environment variables:")
        missing = []
        if not DB_HOST:
            missing.append("DATABASE_HOST")
        if not DB_NAME:
            missing.append("DATABASE_NAME")
        if not DB_USER:
            missing.append("DATABASE_USER")
        if not DB_PASSWORD:
            missing.append("DATABASE_PASSWORD")
        print(f"   Missing: {', '.join(missing)}")
        return False
    
    # Build connection string
    # Format: postgresql://user:password@host:port/database
    connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    print("=" * 80)
    print("DATABASE CONNECTION TEST")
    print("=" * 80)
    print(f"\nConnection Details:")
    print(f"  Host: {DB_HOST}")
    print(f"  Port: {DB_PORT}")
    print(f"  Database: {DB_NAME}")
    print(f"  User: {DB_USER}")
    print(f"  Password: {'*' * len(DB_PASSWORD) if DB_PASSWORD else 'NOT SET'}")
    
    try:
        print(f"\n🔌 Attempting to connect...")
        engine = create_engine(connection_string, echo=False)
        
        # Test connection
        with engine.connect() as conn:
            # Simple query to test connection
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print(f"✅ Connection successful!")
            print(f"\n📊 PostgreSQL Version:")
            print(f"   {version}")
            
            # Get database name
            result = conn.execute(text("SELECT current_database();"))
            db_name = result.fetchone()[0]
            print(f"\n📁 Connected to database: {db_name}")
            
            # Get current user
            result = conn.execute(text("SELECT current_user;"))
            user = result.fetchone()[0]
            print(f"👤 Connected as user: {user}")
            
            # Check if we can create tables (permissions check)
            try:
                result = conn.execute(text("SELECT has_database_privilege(current_user, current_database(), 'CREATE');"))
                can_create = result.fetchone()[0]
                if can_create:
                    print(f"✅ User has CREATE privileges")
                else:
                    print(f"⚠️  User does NOT have CREATE privileges")
            except Exception as e:
                print(f"⚠️  Could not check CREATE privileges: {e}")
            
            return True
            
    except SQLAlchemyError as e:
        print(f"\n❌ Database connection failed:")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error:")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)


