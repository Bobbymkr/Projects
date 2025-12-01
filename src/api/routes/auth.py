"""
Authentication API Routes.

OAuth2 endpoints for login, token refresh, and user management.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from typing import Dict, Any
import logging

from ..auth.oauth2 import (
    create_access_token,
    create_refresh_token,
    verify_password,
    get_password_hash,
    verify_refresh_token,
    get_current_user,
)
from ..config import settings
from ..dependencies import rate_limit

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/token", summary="OAuth2 Token Endpoint")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    _rate_limit: None = Depends(rate_limit),
) -> Dict[str, Any]:
    """
    OAuth2 password flow token endpoint.
    
    Returns access token and refresh token upon successful authentication.
    """
    # TODO: Validate credentials against database
    # For now, using mock validation
    
    # Mock user database
    fake_users_db = {
        "admin": {
            "username": "admin",
            "hashed_password": get_password_hash("admin123"),
            "email": "admin@example.com",
            "role": "admin",
            "permissions": ["read", "write", "admin"],
        },
        "operator": {
            "username": "operator",
            "hashed_password": get_password_hash("operator123"),
            "email": "operator@example.com",
            "role": "operator",
            "permissions": ["read", "write"],
        },
    }
    
    user = fake_users_db.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create tokens
    access_token = create_access_token(
        data={
            "sub": user["username"],
            "email": user["email"],
            "role": user["role"],
            "permissions": user["permissions"],
        }
    )
    
    refresh_token = create_refresh_token(
        data={
            "sub": user["username"],
            "email": user["email"],
            "role": user["role"],
        }
    )
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }


@router.post("/refresh", summary="Refresh Access Token")
async def refresh_token(
    refresh_token: str,
    _rate_limit: None = Depends(rate_limit),
) -> Dict[str, Any]:
    """
    Refresh access token using refresh token.
    
    Returns new access token.
    """
    # Verify refresh token
    payload = await verify_refresh_token(refresh_token)
    
    # Create new access token
    access_token = create_access_token(
        data={
            "sub": payload["sub"],
            "email": payload.get("email"),
            "role": payload.get("role"),
        }
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }


@router.get("/me", summary="Get Current User")
async def get_current_user_info(
    current_user: Dict[str, Any] = Depends(get_current_user),
    _rate_limit: None = Depends(rate_limit),
) -> Dict[str, Any]:
    """Get current authenticated user information."""
    return current_user


@router.post("/register", summary="Register New User")
async def register(
    username: str,
    email: str,
    password: str,
    role: str = "viewer",
    _rate_limit: None = Depends(rate_limit),
) -> Dict[str, Any]:
    """
    Register a new user account.
    
    Note: In production, this should include email verification.
    """
    # TODO: Implement actual user registration with database
    # For now, return mock response
    
    return {
        "message": "User registration successful",
        "username": username,
        "email": email,
        "role": role,
    }

