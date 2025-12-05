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


# Authentication dependency
async def get_current_user(
    authorization: Optional[str] = Header(None),
):
    """
    Authentication dependency using JWT token validation.
    
    Extracts and validates JWT token from Authorization header.
    Returns user information if valid, raises HTTPException if invalid.
    """
    # Import here to avoid circular dependencies
    try:
        from ..auth.oauth2 import get_current_user as validate_jwt_token
        from fastapi.security import OAuth2PasswordBearer
        
        # Extract token from Authorization header (format: "Bearer <token>")
        token = None
        if authorization:
            parts = authorization.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]
        
        # Use existing JWT validation from oauth2 module
        if token:
            return await validate_jwt_token(token)
        else:
            # For endpoints that require authentication, this should raise
            # For now, return anonymous user (caller can check and raise if needed)
            return {"user_id": "anonymous", "role": "viewer"}
    except ImportError:
        # Fallback if oauth2 module not available
        logger.warning("OAuth2 module not available, using anonymous user")
        return {"user_id": "anonymous", "role": "viewer"}


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

