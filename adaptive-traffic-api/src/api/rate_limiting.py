"""
Redis-Based Rate Limiting Implementation.

Implements distributed rate limiting using Redis for horizontal scaling.
"""

import time
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import FastAPI components with graceful fallback
try:
    from fastapi import Request, HTTPException, status
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available. Rate limiting will have limited functionality.")
    from .fastapi_fallback import (
        MockRequest as Request,
        HTTPException,
        status,
    )

from .config import settings
from .cache import get_redis_pool


class RateLimiter:
    """Redis-based rate limiter for API endpoints."""
    
    def __init__(self):
        self.key_prefix = "rate_limit:"
    
    def _make_key(self, identifier: str, endpoint: str) -> str:
        """Create a rate limit key."""
        return f"{self.key_prefix}{identifier}:{endpoint}"
    
    async def check_rate_limit(
        self,
        identifier: str,
        endpoint: str,
        limit: int,
        window: int,
    ) -> Tuple[bool, Optional[int], Optional[int]]:
        """
        Check if request should be rate limited.
        
        Args:
            identifier: Client identifier (IP, user ID, etc.)
            endpoint: API endpoint path
            limit: Maximum requests allowed
            window: Time window in seconds
            
        Returns:
            Tuple of (allowed, remaining, reset_time)
        """
        redis_client = await get_redis_pool()
        if not redis_client:
            # If Redis unavailable, allow request (fail open)
            logger.warning("Redis unavailable, rate limiting disabled")
            return True, None, None
        
        try:
            key = self._make_key(identifier, endpoint)
            current_time = int(time.time())
            
            # Use Redis sliding window log algorithm
            pipeline = redis_client.pipeline()
            
            # Remove old entries outside window
            pipeline.zremrangebyscore(key, 0, current_time - window)
            
            # Count current requests in window
            pipeline.zcard(key)
            
            # Add current request
            pipeline.zadd(key, {str(current_time): current_time})
            
            # Set expiration
            pipeline.expire(key, window)
            
            results = await pipeline.execute()
            count = results[1]  # Count before adding current request
            current_count = count + 1
            
            # Check limit
            if current_count > limit:
                # Calculate reset time (oldest entry + window)
                oldest_entry = await redis_client.zrange(key, 0, 0, withscores=True)
                if oldest_entry:
                    reset_time = int(oldest_entry[0][1]) + window
                else:
                    reset_time = current_time + window
                
                return False, 0, reset_time
            
            remaining = max(0, limit - current_count)
            reset_time = current_time + window
            
            return True, remaining, reset_time
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            # Fail open - allow request if rate limiting fails
            return True, None, None
    
    async def get_client_identifier(self, request: Request) -> str:
        """
        Get client identifier for rate limiting.
        
        Priority:
        1. X-Forwarded-For header (if behind proxy)
        2. X-Real-IP header (if behind proxy)
        3. Direct client IP
        
        Args:
            request: FastAPI request object
            
        Returns:
            Client identifier string
        """
        # Check for forwarded IP (behind proxy/load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take first IP in chain
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        # Direct client IP
        if request.client:
            return request.client.host
        
        return "unknown"


# Global rate limiter instance
rate_limiter = RateLimiter()


async def check_rate_limit(
    request: Request,
    limit_per_minute: Optional[int] = None,
    limit_per_hour: Optional[int] = None,
) -> None:
    """
    Check and enforce rate limits for a request.
    
    Raises HTTPException if rate limit exceeded.
    
    Args:
        request: FastAPI request object
        limit_per_minute: Requests per minute limit
        limit_per_hour: Requests per hour limit
    """
    # Skip rate limiting if FastAPI not available
    if not FASTAPI_AVAILABLE:
        logger.debug("Rate limiting skipped - FastAPI not available")
        return
    
    if not settings.ENABLE_RATE_LIMITING:
        return
    
    # Use configured limits if not specified
    limit_per_minute = limit_per_minute or settings.RATE_LIMIT_PER_MINUTE
    limit_per_hour = limit_per_hour or settings.RATE_LIMIT_PER_HOUR
    
    # Get client identifier
    client_id = await rate_limiter.get_client_identifier(request)
    endpoint = request.url.path
    
    # Check per-minute limit
    allowed, remaining, reset_time = await rate_limiter.check_rate_limit(
        client_id,
        f"{endpoint}:minute",
        limit_per_minute,
        60,  # 1 minute window
    )
    
    if not allowed:
        reset_timestamp = reset_time or int(time.time()) + 60
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit exceeded",
                "message": f"Too many requests. Limit: {limit_per_minute} per minute",
                "retry_after": reset_timestamp - int(time.time()),
            },
            headers={
                "X-RateLimit-Limit": str(limit_per_minute),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_timestamp),
                "Retry-After": str(reset_timestamp - int(time.time())),
            },
        )
    
    # Check per-hour limit
    allowed, remaining, reset_time = await rate_limiter.check_rate_limit(
        client_id,
        f"{endpoint}:hour",
        limit_per_hour,
        3600,  # 1 hour window
    )
    
    if not allowed:
        reset_timestamp = reset_time or int(time.time()) + 3600
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit exceeded",
                "message": f"Too many requests. Limit: {limit_per_hour} per hour",
                "retry_after": reset_timestamp - int(time.time()),
            },
            headers={
                "X-RateLimit-Limit": str(limit_per_hour),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_timestamp),
                "Retry-After": str(reset_timestamp - int(time.time())),
            },
        )
    
    # Add rate limit headers to response
    request.state.rate_limit_remaining = remaining
    request.state.rate_limit_reset = reset_time

