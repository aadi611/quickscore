"""
Database configuration and session management.
Supports both SQLAlchemy (direct PostgreSQL) and Supabase REST API.
"""
import os
import logging
from typing import Generator, Any
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import OperationalError

from app.core.config import settings

# Setup logging
logger = logging.getLogger(__name__)

# Database configuration strategy
USE_SUPABASE_REST = False
engine = None
SessionLocal = None

# Try to initialize SQLAlchemy first
try:
    logger.info("Attempting to initialize SQLAlchemy database connection...")
    
    if settings.DATABASE_URL.startswith("sqlite"):
        # SQLite specific configuration for testing
        engine = create_engine(
            settings.DATABASE_URL,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    else:
        # PostgreSQL configuration for production
        engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            connect_args={"connect_timeout": 10}
        )
    
    # Test the connection
    with engine.connect() as conn:
        conn.execute("SELECT 1")
    
    # Create SessionLocal class
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("✅ SQLAlchemy database connection successful")
    
except (OperationalError, Exception) as e:
    logger.warning(f"❌ SQLAlchemy connection failed: {e}")
    logger.info("🔄 Falling back to Supabase REST API...")
    USE_SUPABASE_REST = True

# Create Base class for models (used by SQLAlchemy)
Base = declarative_base()


def get_db() -> Generator[Any, None, None]:
    """
    Dependency to get database session.
    Returns either SQLAlchemy session or Supabase client.
    """
    if USE_SUPABASE_REST:
        # Return Supabase database instance
        try:
            from app.database.supabase_db import db as supabase_db
            yield supabase_db
        except ImportError:
            raise RuntimeError("Supabase database not available")
    else:
        # Return SQLAlchemy session
        if SessionLocal is None:
            raise RuntimeError("SQLAlchemy not initialized")
        
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()


def is_using_supabase() -> bool:
    """Check if we're using Supabase REST API instead of SQLAlchemy."""
    return USE_SUPABASE_REST


async def init_database():
    """Initialize database tables based on the current configuration."""
    if USE_SUPABASE_REST:
        logger.info("Initializing Supabase database...")
        try:
            from app.database.supabase_db import init_database
            return await init_database()
        except ImportError:
            logger.error("Supabase database not available")
            return False
    else:
        logger.info("Initializing SQLAlchemy database...")
        try:
            # Create all tables
            Base.metadata.create_all(bind=engine)
            logger.info("✅ SQLAlchemy tables created successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create SQLAlchemy tables: {e}")
            return False