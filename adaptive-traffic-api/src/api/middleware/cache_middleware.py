"""
Response Caching Middleware.

Implements intelligent HTTP response caching for performance optimization.
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse
from typing import Callable, Optional
import hashlib
import json
import logging

from ..cache import cache_manager

logger = logging.getLogger(__name__)


class ResponseCacheMiddleware(BaseHTTPMiddleware):
    """
    Middleware to cache HTTP responses.
    
    Caches GET requests based on URL and query parameters.
    Supports cache-control headers for fine-grained control.
    """
    
    def __init__(self, app, default_ttl: int = 300):
        super().__init__(app)
        self.default_ttl = default_ttl
        self.cacheable_methods = {"GET", "HEAD"}
    
    def _make_cache_key(self, request: Request) -> str:
        """Generate cache key from request."""
        # Include path and query parameters
        path = request.url.path
        query = str(sorted(request.query_params.items()))
        key_string = f"{request.method}:{path}:{query}"
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"response_cache:{key_hash}"
    
    def _is_cacheable(self, request: Request, response: Response) -> bool:
        """Check if response should be cached."""
        # Only cache GET/HEAD requests
        if request.method not in self.cacheable_methods:
            return False
        
        # Don't cache if Cache-Control: no-cache
        cache_control = response.headers.get("Cache-Control", "")
        if "no-cache" in cache_control or "no-store" in cache_control:
            return False
        
        # Only cache successful responses
        if response.status_code not in [200, 301, 302]:
            return False
        
        # Don't cache responses with Set-Cookie (auth tokens)
        if "Set-Cookie" in response.headers:
            return False
        
        return True
    
    def _get_cache_ttl(self, response: Response, default: int) -> int:
        """Extract TTL from Cache-Control header or use default."""
        cache_control = response.headers.get("Cache-Control", "")
        
        # Parse max-age from Cache-Control
        if "max-age" in cache_control:
            try:
                max_age_str = cache_control.split("max-age=")[1].split(",")[0].strip()
                return int(max_age_str)
            except (ValueError, IndexError):
                pass
        
        return default
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with caching."""
        # Only process cacheable methods
        if request.method not in self.cacheable_methods:
            return await call_next(request)
        
        # Check cache
        cache_key = self._make_cache_key(request)
        cached_response = await cache_manager.get(cache_key)
        
        if cached_response:
            logger.debug(f"Cache HIT for {request.url.path}")
            
            # Create response from cache
            response = StarletteResponse(
                content=json.dumps(cached_response["body"]).encode(),
                status_code=cached_response["status_code"],
                headers=cached_response["headers"],
            )
            response.headers["X-Cache"] = "HIT"
            return response
        
        # Process request
        response = await call_next(request)
        
        # Cache response if cacheable
        if self._is_cacheable(request, response):
            try:
                # Get response body
                body = await response.body()
                
                # Try to parse as JSON
                try:
                    body_data = json.loads(body)
                except json.JSONDecodeError:
                    body_data = body.decode()
                
                # Determine TTL
                ttl = self._get_cache_ttl(response, self.default_ttl)
                
                # Cache response
                cache_data = {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": body_data,
                }
                
                await cache_manager.set(cache_key, cache_data, ttl=ttl)
                response.headers["X-Cache"] = "MISS"
                logger.debug(f"Cached response for {request.url.path} (TTL: {ttl}s)")
            except Exception as e:
                logger.error(f"Error caching response: {e}")
        
        return response

