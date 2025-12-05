"""
FastAPI Main Application Entry Point.

Production-grade API server with OpenAPI 3.0 documentation,
real-time WebSocket support, and comprehensive middleware.
"""

from contextlib import asynccontextmanager
import time
import logging
from typing import TYPE_CHECKING

# Setup structured logging first
from .config import settings
from .logging_config import setup_logging

setup_logging(
    level=settings.LOG_LEVEL,
    json_format=settings.ENVIRONMENT == "production",
)

logger = logging.getLogger(__name__)

# Check FastAPI availability with health check
from .fastapi_fallback import check_fastapi_health

FASTAPI_AVAILABLE, fastapi_error = check_fastapi_health()

# Production safeguard: Require FastAPI in production
if settings.ENVIRONMENT == "production" and not FASTAPI_AVAILABLE:
    error_msg = (
        f"FastAPI is required in production environment but is not available. "
        f"Error: {fastapi_error}. "
        f"Please install FastAPI: pip install fastapi uvicorn"
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg)

# Handle fallback mode based on configuration
if not FASTAPI_AVAILABLE:
    if settings.FASTAPI_FALLBACK_MODE == "error":
        error_msg = (
            f"FastAPI is not available and FASTAPI_FALLBACK_MODE is set to 'error'. "
            f"Error: {fastapi_error}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    elif settings.FASTAPI_FALLBACK_MODE == "warn":
        logger.warning(
            f"FastAPI not available. Running in fallback mode. "
            f"Error: {fastapi_error}. "
            f"Some functionality may be limited."
        )
    # silent mode: no logging

# Import FastAPI or fallbacks
if FASTAPI_AVAILABLE:
    try:
        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
        logger.debug("FastAPI imported successfully")
    except ImportError as e:
        logger.error(f"FastAPI import failed despite health check: {e}")
        FASTAPI_AVAILABLE = False
        from .fastapi_fallback import (
            MockFastAPI as FastAPI,
            MockRequest as Request,
            MockCORSMiddleware as CORSMiddleware,
            MockJSONResponse as JSONResponse,
        )
else:
    logger.info("Using FastAPI fallback implementations")
    from .fastapi_fallback import (
        MockFastAPI as FastAPI,
        MockRequest as Request,
        MockCORSMiddleware as CORSMiddleware,
        MockJSONResponse as JSONResponse,
    )
from .routes import api_router, websocket_router
from .middleware import (
    add_request_id_middleware,
    add_timing_middleware,
    add_logging_middleware,
)
from .middleware.cache_middleware import ResponseCacheMiddleware
from .dependencies import get_redis_client
from .cache import close_redis_pool
from .monitoring import setup_prometheus_metrics
from .error_tracking import setup_error_tracking

# Prometheus metrics
metrics = setup_prometheus_metrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    logger.info("Starting Adaptive Traffic Control API Server")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"API Version: {settings.API_VERSION}")
    
    # Initialize error tracking (Sentry)
    if settings.ENABLE_SENTRY and settings.SENTRY_DSN:
        setup_error_tracking(settings.SENTRY_DSN, settings.ENVIRONMENT)
    
    # Initialize Redis connection pool if enabled
    if settings.ENABLE_REDIS:
        from .cache import get_redis_pool
        redis_client = await get_redis_pool()
        if redis_client:
            logger.info("Redis connection pool initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Adaptive Traffic Control API Server")
    if settings.ENABLE_REDIS:
        await close_redis_pool()
        logger.info("Redis connection pool closed")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance (or mock if FastAPI unavailable)
    """
    app_title = "Adaptive Traffic Control API"
    if not FASTAPI_AVAILABLE:
        app_title = f"{app_title} (Limited Mode - FastAPI Unavailable)"
    
    app = FastAPI(
        title=app_title,
        description=(
            "Enterprise-grade REST API for intelligent traffic signal control system. "
            "Supports real-time traffic management, multi-agent coordination, "
            "and advanced analytics with AI-powered decision making."
        ),
        version=settings.API_VERSION,
        docs_url="/api/docs" if (settings.ENABLE_DOCS and FASTAPI_AVAILABLE) else None,
        redoc_url="/api/redoc" if (settings.ENABLE_DOCS and FASTAPI_AVAILABLE) else None,
        openapi_url="/api/openapi.json" if (settings.ENABLE_DOCS and FASTAPI_AVAILABLE) else None,
        lifespan=lifespan if FASTAPI_AVAILABLE else None,  # Lifespan only works with real FastAPI
    )
    
    # CORS middleware (only if FastAPI available)
    if FASTAPI_AVAILABLE and settings.ENABLE_CORS:
        try:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=settings.CORS_ORIGINS,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
                expose_headers=["X-Request-ID", "X-Process-Time"],
            )
        except Exception as e:
            logger.warning(f"Failed to add CORS middleware: {e}")
    
    # Custom middleware (only if FastAPI available)
    if FASTAPI_AVAILABLE:
        try:
            add_request_id_middleware(app)
            add_timing_middleware(app)
            add_logging_middleware(app)
            # Add security headers middleware
            from .middleware import add_security_headers_middleware
            add_security_headers_middleware(app)
        except Exception as e:
            logger.warning(f"Failed to add custom middleware: {e}")
    
    # Response caching middleware (if Redis enabled and FastAPI available)
    if FASTAPI_AVAILABLE and settings.ENABLE_REDIS:
        try:
            app.add_middleware(ResponseCacheMiddleware, default_ttl=300)
        except Exception as e:
            logger.warning(f"Failed to add cache middleware: {e}")
    
    # Include routers (works with both real and mock FastAPI)
    try:
        app.include_router(api_router, prefix=f"/api/{settings.API_VERSION}")
        app.include_router(websocket_router, prefix=f"/api/{settings.API_VERSION}/ws")
    except Exception as e:
        logger.warning(f"Failed to include routers: {e}")
    
    # Prometheus metrics endpoint (outside API versioning)
    if settings.ENABLE_PROMETHEUS:
        try:
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            from fastapi.responses import Response
            
            @app.get("/metrics", tags=["Monitoring"])
            async def metrics_endpoint():
                """Prometheus metrics endpoint for scraping."""
                try:
                    return Response(
                        content=generate_latest(),
                        media_type=CONTENT_TYPE_LATEST,
                    )
                except Exception as e:
                    logger.error(f"Error generating Prometheus metrics: {e}")
                    return Response(
                        content="# Error generating metrics\n",
                        media_type="text/plain",
                    )
        except ImportError:
            logger.warning("Prometheus client not available. Metrics endpoint disabled.")
    
    # Health check endpoint (outside versioning)
    @app.get("/health", tags=["System"])
    async def health_check():
        """Health check endpoint for load balancers and monitoring."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": settings.API_VERSION,
            "environment": settings.ENVIRONMENT,
        }
    
    # Root endpoint
    @app.get("/", tags=["System"])
    async def root():
        """API root endpoint with basic information."""
        return {
            "name": "Adaptive Traffic Control API",
            "version": settings.API_VERSION,
            "status": "operational",
            "docs_url": "/api/docs" if settings.ENABLE_DOCS else None,
        }
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler for unhandled errors."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        try:
            metrics.http_requests_total.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=500,
            ).inc()
        except Exception:
            pass  # Metrics not critical for error handling
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "request_id": getattr(request.state, "request_id", None),
            },
        )
    
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )

