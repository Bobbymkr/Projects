"""API utilities for the adaptive traffic control system.

This module provides utilities for creating and configuring FastAPI applications,
including middleware for request quota enforcement, error handling, metrics collection,
and health check endpoints.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union

# Configure module logger
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from fastapi import FastAPI, Request, Response, Depends, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.routing import APIRouter
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available. API utilities will be disabled.")

# Import local modules
try:
    from .quota import RequestQuotaManager, QuotaExceeded
    from .errors import create_api_error_handlers, AdaptiveTrafficError
    from .health import HealthCheck
    from .metrics import MetricsRegistry, ApplicationMetrics
except ImportError:
    # Handle case where modules are not yet available
    logger.warning("Some utility modules not available. Some API features will be disabled.")
    RequestQuotaManager = None
    QuotaExceeded = Exception
    create_api_error_handlers = None
    AdaptiveTrafficError = Exception
    HealthCheck = None
    MetricsRegistry = None
    ApplicationMetrics = None


def create_api_app(
    title: str = "Adaptive Traffic Control API",
    description: str = "API for the adaptive traffic control system",
    version: str = "0.1.0",
    enable_cors: bool = True,
    enable_metrics: bool = True,
    enable_health: bool = True,
    enable_quota: bool = True,
    quota_manager: Optional[Any] = None,
    metrics_registry: Optional[Any] = None,
    health_check: Optional[Any] = None,
) -> Any:
    """Create a FastAPI application with common middleware and routes.
    
    Args:
        title: API title
        description: API description
        version: API version
        enable_cors: Whether to enable CORS middleware
        enable_metrics: Whether to enable metrics collection
        enable_health: Whether to enable health check endpoints
        enable_quota: Whether to enable request quota enforcement
        quota_manager: Request quota manager to use
        metrics_registry: Metrics registry to use
        health_check: Health check instance to use
        
    Returns:
        FastAPI application
        
    Raises:
        ImportError: If FastAPI is not available
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is not available. Please install it with 'pip install fastapi uvicorn[standard]'.")
    
    # Create FastAPI app
    app = FastAPI(
        title=title,
        description=description,
        version=version,
    )
    
    # Add CORS middleware if enabled
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Set up metrics if enabled
    if enable_metrics:
        metrics_registry = metrics_registry or MetricsRegistry()
        app_metrics = ApplicationMetrics(metrics_registry)
        
        # Add metrics middleware
        @app.middleware("http")
        async def metrics_middleware(request: Request, call_next: Callable) -> Response:
            endpoint = f"{request.method} {request.url.path}"
            start_time = time.time()
            
            try:
                response = await call_next(request)
                return response
            finally:
                latency = time.time() - start_time
                app_metrics.request_latency.labels(endpoint=endpoint).observe(latency)
                app_metrics.request_count.labels(endpoint=endpoint).inc()
    
    # Set up quota enforcement if enabled
    if enable_quota and RequestQuotaManager is not None:
        quota_manager = quota_manager or RequestQuotaManager()
        
        # Add quota middleware
        @app.middleware("http")
        async def quota_middleware(request: Request, call_next: Callable) -> Response:
            # Skip quota check for certain endpoints
            if request.url.path in ["/health", "/metrics", "/health/quotas"]:
                return await call_next(request)
            
            # Check quota
            remaining = quota_manager.remaining()
            if remaining <= 0:
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": "Request quota exceeded. Please try again later."}
                )
            
            # Increment quota and continue
            response = await call_next(request)
            quota_manager.increment()
            
            # Add quota headers
            response.headers["X-RateLimit-Remaining"] = str(quota_manager.remaining())
            response.headers["X-RateLimit-Limit"] = str(quota_manager.max_requests)
            
            return response
    
    # Set up health check endpoints if enabled
    if enable_health:
        health_router = create_health_router(health_check, quota_manager)
        app.include_router(health_router)
    
    # Set up error handlers
    if create_api_error_handlers is not None:
        create_api_error_handlers(app)
    
    return app


def create_health_router(health_check: Optional[Any] = None, quota_manager: Optional[Any] = None) -> Any:
    """Create a router for health check endpoints.
    
    Args:
        health_check: Health check instance to use
        quota_manager: Request quota manager to use
        
    Returns:
        APIRouter instance
        
    Raises:
        ImportError: If FastAPI is not available
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is not available. Please install it with 'pip install fastapi uvicorn[standard]'.")
    
    router = APIRouter(tags=["health"])
    
    @router.get("/health", summary="Get system health status")
    async def get_health() -> Dict[str, Any]:
        """Get the health status of the system."""
        if health_check is None:
            return {"status": "ok"}
        
        return await health_check.check_health()
    
    @router.get("/health/quotas", summary="Get quota status")
    async def get_quota_status() -> Dict[str, Any]:
        """Get the status of request quotas."""
        if quota_manager is None:
            return {"status": "ok", "quotas": {"enabled": False}}
        
        return {
            "status": "ok" if quota_manager.remaining() > 0 else "warning",
            "quotas": {
                "enabled": True,
                "remaining": quota_manager.remaining(),
                "limit": quota_manager.max_requests,
                "reset_at": quota_manager.reset_time.isoformat() if hasattr(quota_manager, "reset_time") else None,
            }
        }
    
    return router


def create_metrics_router(metrics_registry: Optional[Any] = None) -> Any:
    """Create a router for metrics endpoints.
    
    Args:
        metrics_registry: Metrics registry to use
        
    Returns:
        APIRouter instance
        
    Raises:
        ImportError: If FastAPI is not available
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is not available. Please install it with 'pip install fastapi uvicorn[standard]'.")
    
    router = APIRouter(tags=["metrics"])
    
    @router.get("/metrics", summary="Get Prometheus metrics")
    async def get_metrics() -> Response:
        """Get Prometheus metrics."""
        if metrics_registry is None or not hasattr(metrics_registry, "generate_latest"):
            return Response(content="# No metrics available\n", media_type="text/plain")
        
        metrics_data = metrics_registry.generate_latest()
        return Response(content=metrics_data, media_type="text/plain")
    
    return router


def create_api_key_dependency(api_keys: List[str]) -> Callable:
    """Create a dependency for API key authentication.
    
    Args:
        api_keys: List of valid API keys
        
    Returns:
        Dependency function
        
    Raises:
        ImportError: If FastAPI is not available
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is not available. Please install it with 'pip install fastapi uvicorn[standard]'.")
    
    async def verify_api_key(request: Request) -> None:
        """Verify the API key in the request.
        
        Args:
            request: FastAPI request
            
        Raises:
            HTTPException: If the API key is invalid
        """
        api_key = request.headers.get("X-API-Key")
        if api_key not in api_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )
    
    return verify_api_key


def run_api_server(
    app: Any,
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "info",
    reload: bool = False,
) -> None:
    """Run the FastAPI server.
    
    Args:
        app: FastAPI application
        host: Host to bind to
        port: Port to bind to
        log_level: Log level
        reload: Whether to enable auto-reload
        
    Raises:
        ImportError: If uvicorn is not available
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn is not available. Please install it with 'pip install uvicorn[standard]'.")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        reload=reload,
    )