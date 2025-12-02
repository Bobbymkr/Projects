"""
Redis Caching Layer for Performance Optimization.

Implements intelligent caching strategies for API responses,
traffic data, and frequently accessed data.
"""

from typing import Optional, Any, Callable
from functools import wraps
import json
import hashlib
import logging
from datetime import timedelta

from .config import settings

logger = logging.getLogger(__name__)

# Global Redis client instance
_redis_client: Optional[Any] = None


async def get_redis_pool():
    """
    Get or create Redis connection pool.
    
    Returns:
        Redis client instance or None if Redis disabled
    """
    global _redis_client
    
    if not settings.ENABLE_REDIS:
        return None
    
    if _redis_client is None:
        try:
            from redis.asyncio import Redis, ConnectionPool
            
            pool = ConnectionPool(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                max_connections=50,
                decode_responses=True,
            )
            
            _redis_client = Redis(connection_pool=pool)
            logger.info("Redis connection pool created")
            
        except ImportError:
            logger.warning("redis package not installed. Caching disabled.")
            return None
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return None
    
    return _redis_client


async def close_redis_pool():
    """Close Redis connection pool."""
    global _redis_client
    if _redis_client:
        try:
            await _redis_client.aclose()
            _redis_client = None
            logger.info("Redis connection pool closed")
        except Exception as e:
            logger.error(f"Error closing Redis pool: {e}")


class CacheManager:
    """Manages caching operations with Redis."""
    
    def __init__(self):
        self.default_ttl = 300  # 5 minutes default
        self.key_prefix = "api:cache:"
    
    def _make_key(self, key: str) -> str:
        """Create a prefixed cache key."""
        return f"{self.key_prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        redis_client = await get_redis_pool()
        if not redis_client:
            return None
        
        try:
            full_key = self._make_key(key)
            value = await redis_client.get(full_key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time to live in seconds (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        redis_client = await get_redis_pool()
        if not redis_client:
            return False
        
        try:
            full_key = self._make_key(key)
            ttl = ttl or self.default_ttl
            serialized = json.dumps(value, default=str)
            await redis_client.setex(full_key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        redis_client = await get_redis_pool()
        if not redis_client:
            return False
        
        try:
            full_key = self._make_key(key)
            await redis_client.delete(full_key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching a pattern.
        
        Args:
            pattern: Redis pattern (e.g., "traffic:*")
            
        Returns:
            Number of keys deleted
        """
        redis_client = await get_redis_pool()
        if not redis_client:
            return 0
        
        try:
            full_pattern = self._make_key(pattern)
            keys = []
            async for key in redis_client.scan_iter(match=full_pattern):
                keys.append(key)
            
            if keys:
                return await redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache clear pattern error for {pattern}: {e}")
            return 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        redis_client = await get_redis_pool()
        if not redis_client:
            return False
        
        try:
            full_key = self._make_key(key)
            return bool(await redis_client.exists(full_key))
        except Exception as e:
            logger.error(f"Cache exists check error for key {key}: {e}")
            return False


# Global cache manager instance
cache_manager = CacheManager()


def cache_key(*args, **kwargs) -> str:
    """
    Generate a cache key from function arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Cache key string
    """
    # Sort kwargs for consistent key generation
    sorted_kwargs = sorted(kwargs.items())
    
    # Create hash from arguments
    key_data = {
        "args": str(args),
        "kwargs": str(sorted_kwargs),
    }
    key_string = json.dumps(key_data, sort_keys=True, default=str)
    key_hash = hashlib.md5(key_string.encode()).hexdigest()
    
    return key_hash


def cached(
    ttl: Optional[int] = None,
    key_prefix: Optional[str] = None,
    include_request: bool = False,
):
    """
    Decorator to cache async function results.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
        include_request: Include request object in cache key
        
    Example:
        @cached(ttl=300)
        async def get_metrics():
            return {"data": "..."}
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            prefix = key_prefix or f"{func.__module__}:{func.__name__}"
            
            # Include request in key if needed
            if include_request:
                request = None
                for arg in args:
                    if hasattr(arg, "url"):  # FastAPI Request object
                        request = arg
                        break
                
                if request:
                    key = f"{prefix}:{request.url.path}:{cache_key(*args, **kwargs)}"
                else:
                    key = f"{prefix}:{cache_key(*args, **kwargs)}"
            else:
                key = f"{prefix}:{cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_value = await cache_manager.get(key)
            if cached_value is not None:
                logger.debug(f"Cache HIT for {key}")
                return cached_value
            
            # Execute function
            logger.debug(f"Cache MISS for {key}")
            result = await func(*args, **kwargs)
            
            # Store in cache
            await cache_manager.set(key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator

