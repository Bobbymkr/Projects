"""
FastAPI Dependencies.

Reusable dependencies for dependency injection across routes.
"""

import logging
from typing import Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

# Try to import FastAPI dependencies with graceful fallback
try:
    from fastapi import Depends, HTTPException, Header, Request
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available. Dependency injection will be limited.")
    from ..fastapi_fallback import (
        Depends,
        HTTPException,
        MockRequest as Request,
        Header,
    )

from .config import settings
from .rate_limiting import check_rate_limit

# Import services (may fail if dependencies missing)
try:
    from .services.traffic_controller import TrafficController
    TRAFFIC_CONTROLLER_AVAILABLE = True
except ImportError as e:
    TRAFFIC_CONTROLLER_AVAILABLE = False
    logger.warning(f"TrafficController not available: {e}")
    TrafficController = None


# Rate limiting dependency
async def rate_limit(request: Request):
    """
    Rate limiting dependency using Redis.
    
    Checks per-minute and per-hour rate limits.
    Raises HTTPException if limits exceeded.
    """
    await check_rate_limit(request)
    return None


# Authentication dependency (placeholder)
async def get_current_user(
    authorization: Optional[str] = Header(None),
):
    """
    Authentication dependency.
    
    TODO: Implement JWT token validation
    """
    if not authorization:
        # For now, allow unauthenticated access
        # In production, this should raise HTTPException
        return {"user_id": "anonymous", "role": "viewer"}
    
    # TODO: Validate JWT token
    # Extract and validate token
    # Return user information
    
    return {"user_id": "user_123", "role": "operator"}


# Service dependencies
_traffic_controller: Optional[TrafficController] = None


async def get_traffic_controller() -> TrafficController:
    """Get traffic controller service instance (singleton)."""
    global _traffic_controller
    if _traffic_controller is None:
        _traffic_controller = TrafficController()
    return _traffic_controller


# Redis client dependency
async def get_redis_client():
    """Get Redis client connection."""
    from .cache import get_redis_pool
    return await get_redis_pool()

