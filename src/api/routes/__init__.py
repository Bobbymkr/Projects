"""
API Route Handlers.

Organized route handlers for REST API and WebSocket endpoints.
"""

import logging

logger = logging.getLogger(__name__)

# Try to import FastAPI APIRouter with graceful fallback
try:
    from fastapi import APIRouter
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available. Routes will use mock router.")
    from ..fastapi_fallback import MockAPIRouter as APIRouter

# Main API router
api_router = APIRouter()

# Try to import route modules with graceful error handling
_route_modules = {}

try:
    from .traffic import router as traffic_router
    _route_modules['traffic'] = traffic_router
except ImportError as e:
    logger.warning(f"Could not import traffic router: {e}")

try:
    from .metrics import router as metrics_router
    _route_modules['metrics'] = metrics_router
except ImportError as e:
    logger.warning(f"Could not import metrics router: {e}")

try:
    from .system import router as system_router
    _route_modules['system'] = system_router
except ImportError as e:
    logger.warning(f"Could not import system router: {e}")

try:
    from .analytics import router as analytics_router
    _route_modules['analytics'] = analytics_router
except ImportError as e:
    logger.warning(f"Could not import analytics router: {e}")

try:
    from .monitoring import router as monitoring_router
    _route_modules['monitoring'] = monitoring_router
except ImportError as e:
    logger.warning(f"Could not import monitoring router: {e}")

try:
    from .auth import router as auth_router
    _route_modules['auth'] = auth_router
except ImportError as e:
    logger.warning(f"Could not import auth router: {e}")

try:
    from .graphql import router as graphql_router
    _route_modules['graphql'] = graphql_router
except ImportError as e:
    logger.warning(f"Could not import graphql router: {e}")

# Include sub-routers (only if they were successfully imported)
if 'traffic' in _route_modules:
    api_router.include_router(_route_modules['traffic'], prefix="/traffic", tags=["Traffic Control"])

if 'metrics' in _route_modules:
    api_router.include_router(_route_modules['metrics'], prefix="/metrics", tags=["Metrics"])

if 'system' in _route_modules:
    api_router.include_router(_route_modules['system'], prefix="/system", tags=["System"])

if 'analytics' in _route_modules:
    api_router.include_router(_route_modules['analytics'], prefix="/analytics", tags=["Analytics"])

if 'monitoring' in _route_modules:
    api_router.include_router(_route_modules['monitoring'], prefix="/monitoring", tags=["Monitoring"])

if 'auth' in _route_modules:
    api_router.include_router(_route_modules['auth'], prefix="/auth", tags=["Authentication"])

if 'graphql' in _route_modules:
    api_router.include_router(_route_modules['graphql'], tags=["GraphQL"])

# WebSocket router
try:
    from .websocket import router as websocket_router
except ImportError as e:
    logger.warning(f"Could not import websocket router: {e}")
    # Create empty router as fallback
    websocket_router = APIRouter()

__all__ = ["api_router", "websocket_router"]

