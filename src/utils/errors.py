"""Error handling utilities.

This module provides error handling utilities for the application, including:
- Custom exception classes
- Exception hooks for graceful handling
- Retry decorators for resilient operations
- Error reporting integrations
"""

import sys
import logging
import traceback
from functools import wraps
from typing import Optional, Callable, Dict, Any, Type, List, Union

# Configure module logger
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import sentry_sdk
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False


# Custom exception classes
class AdaptiveTrafficError(Exception):
    """Base exception for all application-specific errors."""
    pass


class ConfigurationError(AdaptiveTrafficError):
    """Error raised when there's an issue with configuration."""
    pass


class ResourceError(AdaptiveTrafficError):
    """Error raised when a required resource is unavailable."""
    pass


class ModelError(AdaptiveTrafficError):
    """Error raised when there's an issue with ML models."""
    pass


class SimulationError(AdaptiveTrafficError):
    """Error raised when there's an issue with the traffic simulation."""
    pass


class VisionError(AdaptiveTrafficError):
    """Error raised when there's an issue with the vision system."""
    pass


class APIError(AdaptiveTrafficError):
    """Error raised when there's an issue with API operations."""
    pass


# Error reporting
def init_error_reporting(dsn: Optional[str] = None, environment: str = "development"):
    """Initialize error reporting with Sentry.
    
    Args:
        dsn: Sentry DSN string
        environment: Environment name (development, staging, production)
    """
    if not SENTRY_AVAILABLE:
        logger.warning("Sentry SDK not available. Install with 'pip install sentry-sdk'")
        return
    
    if not dsn:
        logger.info("Sentry DSN not provided, skipping error reporting setup")
        return
    
    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        traces_sample_rate=0.2,  # Sample 20% of transactions for performance monitoring
        send_default_pii=False,  # Don't send personally identifiable information
    )
    
    logger.info(f"Sentry error reporting initialized for {environment} environment")


# Exception hooks
def install_global_exception_hook(exit_on_error: bool = True):
    """Install a global exception hook to catch unhandled exceptions.
    
    Args:
        exit_on_error: Whether to exit the program on unhandled exceptions
    """
    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't capture keyboard interrupt
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.critical(
            "Unhandled exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
        
        if exit_on_error:
            sys.exit(1)
    
    sys.excepthook = exception_handler
    logger.debug("Global exception hook installed")


# Retry decorators
def with_retry(max_attempts: int = 3, 
              retry_exceptions: List[Type[Exception]] = None,
              wait_min_seconds: float = 1.0,
              wait_max_seconds: float = 10.0):
    """Decorator to retry functions on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        retry_exceptions: List of exception types to retry on
        wait_min_seconds: Minimum wait time between retries
        wait_max_seconds: Maximum wait time between retries
    """
    if not TENACITY_AVAILABLE:
        logger.warning("Tenacity not available. Install with 'pip install tenacity'")
        
        # Fallback implementation without tenacity
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                attempts = 0
                last_exception = None
                
                while attempts < max_attempts:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        attempts += 1
                        last_exception = e
                        
                        # Check if we should retry this exception
                        should_retry = False
                        if retry_exceptions is None:
                            should_retry = True
                        else:
                            for exc_type in retry_exceptions:
                                if isinstance(e, exc_type):
                                    should_retry = True
                                    break
                        
                        if not should_retry or attempts >= max_attempts:
                            raise
                        
                        logger.warning(
                            f"Retry attempt {attempts}/{max_attempts} for {func.__name__} due to: {e}"
                        )
                        
                        # Simple exponential backoff
                        import time
                        import random
                        wait_time = min(
                            wait_max_seconds,
                            wait_min_seconds * (2 ** (attempts - 1)) + random.uniform(0, 1)
                        )
                        time.sleep(wait_time)
                
                # This should never be reached, but just in case
                if last_exception:
                    raise last_exception
                raise RuntimeError(f"Failed after {max_attempts} attempts")
            
            return wrapper
    else:
        # Use tenacity for more advanced retry logic
        def decorator(func):
            retry_config = dict(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=1, min=wait_min_seconds, max=wait_max_seconds),
                reraise=True,
                before_sleep=lambda retry_state: logger.warning(
                    f"Retry attempt {retry_state.attempt_number}/{max_attempts} for {func.__name__} "
                    f"due to: {retry_state.outcome.exception()}"
                )
            )
            
            if retry_exceptions:
                retry_config["retry"] = retry_if_exception_type(tuple(retry_exceptions))
            
            @retry(**retry_config)
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            return wrapper
    
    return decorator


# Context managers for error handling
class ErrorContext:
    """Context manager for handling errors in a specific context."""
    
    def __init__(self, context_name: str, 
                 fallback_value: Any = None,
                 reraise: bool = True,
                 log_level: int = logging.ERROR):
        """Initialize the error context.
        
        Args:
            context_name: Name of the context for logging
            fallback_value: Value to return if an exception occurs
            reraise: Whether to re-raise the exception
            log_level: Logging level for errors
        """
        self.context_name = context_name
        self.fallback_value = fallback_value
        self.reraise = reraise
        self.log_level = log_level
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.log(
                self.log_level,
                f"Error in {self.context_name}: {exc_val}",
                exc_info=(exc_type, exc_val, exc_tb)
            )
            
            # Report to Sentry if available
            if SENTRY_AVAILABLE and sentry_sdk.Hub.current.client:
                with sentry_sdk.push_scope() as scope:
                    scope.set_tag("context", self.context_name)
                    sentry_sdk.capture_exception(exc_val)
            
            return not self.reraise  # True = suppress exception
        return False  # Don't suppress exception


# Graceful fallback decorator
def with_fallback(fallback_value: Any, log_exception: bool = True):
    """Decorator to provide a fallback value on exception.
    
    Args:
        fallback_value: Value to return if an exception occurs
        log_exception: Whether to log the exception
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_exception:
                    logger.exception(f"Error in {func.__name__}, using fallback: {e}")
                
                # If fallback_value is callable, call it with the exception
                if callable(fallback_value):
                    return fallback_value(e)
                return fallback_value
        return wrapper
    return decorator


# API error handler for FastAPI
def create_api_error_handlers(app):
    """Register error handlers for a FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    try:
        from fastapi import Request, status
        from fastapi.responses import JSONResponse
        from fastapi.exceptions import RequestValidationError
        from starlette.exceptions import HTTPException
        
        @app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "detail": "Validation error",
                    "errors": exc.errors()
                }
            )
        
        @app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail}
            )
        
        @app.exception_handler(AdaptiveTrafficError)
        async def adaptive_traffic_exception_handler(request: Request, exc: AdaptiveTrafficError):
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            
            # Map exception types to status codes
            if isinstance(exc, ConfigurationError):
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            elif isinstance(exc, ResourceError):
                status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            elif isinstance(exc, APIError):
                status_code = status.HTTP_400_BAD_REQUEST
            
            return JSONResponse(
                status_code=status_code,
                content={"detail": str(exc)}
            )
        
        @app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            # Log the exception
            logger.exception(f"Unhandled API exception: {exc}")
            
            # Report to Sentry if available
            if SENTRY_AVAILABLE and sentry_sdk.Hub.current.client:
                with sentry_sdk.push_scope() as scope:
                    scope.set_tag("endpoint", request.url.path)
                    sentry_sdk.capture_exception(exc)
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error"}
            )
        
        logger.info("API error handlers registered")
    except ImportError:
        logger.warning("FastAPI not available, skipping API error handlers")