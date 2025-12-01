"""
Middleware Components for API.

Provides request processing middleware for logging, timing, and request tracking.
"""

import time
import uuid
import logging

logger = logging.getLogger(__name__)

# Try to import FastAPI and Starlette with graceful fallback
try:
    from fastapi import Request
    from starlette.middleware.base import BaseHTTPMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI/Starlette not available. Middleware will be disabled.")
    from ..fastapi_fallback import MockRequest as Request
    
    # Create minimal BaseHTTPMiddleware mock
    class BaseHTTPMiddleware:
        """Mock base middleware class."""
        pass


# Define middleware classes only if FastAPI is available
if FASTAPI_AVAILABLE:
    class RequestIDMiddleware(BaseHTTPMiddleware):
        """Middleware to add unique request ID to each request."""
        
        async def dispatch(self, request: Request, call_next):
            # Generate unique request ID
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            
            # Process request
            response = await call_next(request)
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response


    class TimingMiddleware(BaseHTTPMiddleware):
        """Middleware to track request processing time."""
        
        async def dispatch(self, request: Request, call_next):
            start_time = time.time()
            
            response = await call_next(request)
            
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = f"{process_time:.4f}"
            
            return response


    class LoggingMiddleware(BaseHTTPMiddleware):
        """Middleware for structured request/response logging."""
        
        async def dispatch(self, request: Request, call_next):
            request_id = getattr(request.state, "request_id", "unknown")
            start_time = time.time()
            
            # Log request
            logger.info(
                f"Request started",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "client": request.client.host if request.client else None,
                },
            )
            
            try:
                response = await call_next(request)
                process_time = time.time() - start_time
                
                # Log response
                logger.info(
                    f"Request completed",
                    extra={
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": response.status_code,
                        "process_time": process_time,
                    },
                )
                
                return response
                
            except Exception as e:
                process_time = time.time() - start_time
                logger.error(
                    f"Request failed",
                    extra={
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "process_time": process_time,
                        "error": str(e),
                    },
                    exc_info=True,
                )
                raise
else:
    # Create minimal mock classes when FastAPI not available
    class RequestIDMiddleware:
        """Mock middleware - FastAPI not available."""
        pass
    
    class TimingMiddleware:
        """Mock middleware - FastAPI not available."""
        pass
    
    class LoggingMiddleware:
        """Mock middleware - FastAPI not available."""
        pass


def add_request_id_middleware(app):
    """Add request ID middleware to application."""
    if FASTAPI_AVAILABLE:
        try:
            app.add_middleware(RequestIDMiddleware)
        except Exception as e:
            logger.warning(f"Failed to add request ID middleware: {e}")
    else:
        logger.debug("Request ID middleware skipped - FastAPI not available")


def add_timing_middleware(app):
    """Add timing middleware to application."""
    if FASTAPI_AVAILABLE:
        try:
            app.add_middleware(TimingMiddleware)
        except Exception as e:
            logger.warning(f"Failed to add timing middleware: {e}")
    else:
        logger.debug("Timing middleware skipped - FastAPI not available")


def add_logging_middleware(app):
    """Add logging middleware to application."""
    if FASTAPI_AVAILABLE:
        try:
            app.add_middleware(LoggingMiddleware)
        except Exception as e:
            logger.warning(f"Failed to add logging middleware: {e}")
    else:
        logger.debug("Logging middleware skipped - FastAPI not available")


# Export middleware functions
__all__ = [
    "add_request_id_middleware",
    "add_timing_middleware",
    "add_logging_middleware",
    "RequestIDMiddleware",
    "TimingMiddleware",
    "LoggingMiddleware",
]
