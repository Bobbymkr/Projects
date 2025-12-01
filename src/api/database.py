"""
Database Connection Pooling and Optimization.

Implements async database connection pooling for PostgreSQL
with query optimization and connection management.
"""

from typing import Optional, AsyncGenerator, Any
import logging

from .config import settings

logger = logging.getLogger(__name__)

# Global database pool
_db_pool: Optional[Any] = None


async def get_db_pool():
    """
    Get or create database connection pool.
    
    Returns:
        Database connection pool or None if database disabled
    """
    global _db_pool
    
    # Check if database URL is configured
    if not settings.DATABASE_URL or settings.DATABASE_URL == "postgresql://user:password@localhost:5432/traffic_control":
        logger.warning("Database URL not configured. Database features disabled.")
        return None
    
    if _db_pool is None:
        try:
            from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
            from sqlalchemy.pool import NullPool
            
            # Convert PostgreSQL URL to async format
            async_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
            
            # Create async engine with connection pooling
            engine = create_async_engine(
                async_url,
                pool_size=settings.DATABASE_POOL_SIZE,
                max_overflow=settings.DATABASE_MAX_OVERFLOW,
                pool_pre_ping=True,  # Verify connections before using
                pool_recycle=3600,  # Recycle connections after 1 hour
                echo=False,  # Set to True for SQL query logging
            )
            
            # Create session factory
            _db_pool = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            
            logger.info(f"Database connection pool created (pool_size={settings.DATABASE_POOL_SIZE})")
            
        except ImportError:
            logger.warning("sqlalchemy/asyncpg not installed. Database features disabled.")
            return None
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            return None
    
    return _db_pool


async def get_db_session() -> AsyncGenerator[Any, None]:
    """
    Get database session for dependency injection.
    
    Usage:
        async def my_endpoint(db: AsyncSession = Depends(get_db_session)):
            ...
    """
    pool = await get_db_pool()
    if not pool:
        yield None
        return
    
    async with pool() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def close_db_pool():
    """Close database connection pool."""
    global _db_pool
    if _db_pool:
        try:
            await _db_pool.close_all()
            _db_pool = None
            logger.info("Database connection pool closed")
        except Exception as e:
            logger.error(f"Error closing database pool: {e}")


class QueryOptimizer:
    """Utilities for query optimization."""
    
    @staticmethod
    async def execute_with_timeout(query, timeout: float = 5.0):
        """
        Execute query with timeout.
        
        Args:
            query: SQLAlchemy query object
            timeout: Timeout in seconds
            
        Returns:
            Query results
        """
        # TODO: Implement query timeout
        # This would require asyncpg specific timeout handling
        return await query
    
    @staticmethod
    def add_indexes_hints(query, indexes: list):
        """
        Add index hints to query for optimization.
        
        Args:
            query: SQLAlchemy query object
            indexes: List of index names to hint
            
        Returns:
            Query with index hints
        """
        # TODO: Implement index hints
        # PostgreSQL supports index hints via query planner
        return query

