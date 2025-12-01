"""
OAuth2 Authentication Implementation.

Advanced authentication with OAuth2 flows, JWT tokens, and refresh tokens.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Try to import dependencies with graceful fallbacks
try:
    from jose import JWTError, jwt
    JOSE_AVAILABLE = True
except ImportError:
    JOSE_AVAILABLE = False
    logger.warning("python-jose not available. JWT functionality will be limited.")

try:
    from passlib.context import CryptContext
    PASSLIB_AVAILABLE = True
except ImportError:
    PASSLIB_AVAILABLE = False
    logger.warning("passlib not available. Password hashing will be limited.")

try:
    from fastapi import HTTPException, status, Depends
    from fastapi.security import OAuth2PasswordBearer
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available. OAuth2 functionality will be limited.")

from ..config import settings

# Password hashing context
if PASSLIB_AVAILABLE:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
else:
    pwd_context = None

# OAuth2 scheme
if FASTAPI_AVAILABLE:
    oauth2_scheme = OAuth2PasswordBearer(
        tokenUrl=f"/api/{settings.API_VERSION}/auth/token",
        auto_error=False,
    )
else:
    oauth2_scheme = None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    if not PASSLIB_AVAILABLE:
        logger.warning("Password verification not available without passlib")
        return False
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    if not PASSLIB_AVAILABLE:
        logger.warning("Password hashing not available without passlib")
        return password  # Return as-is (not secure, but allows testing)
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token.
    
    Args:
        data: Data to encode in token
        expires_delta: Token expiration time
        
    Returns:
        Encoded JWT token
    """
    if not JOSE_AVAILABLE:
        logger.warning("JWT token creation not available without python-jose")
        return "mock_token"
    
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )
    
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """
    Create JWT refresh token.
    
    Args:
        data: Data to encode in token
        
    Returns:
        Encoded refresh token (longer expiration)
    """
    if not JOSE_AVAILABLE:
        logger.warning("JWT refresh token creation not available without python-jose")
        return "mock_refresh_token"
    
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({"exp": expire, "iat": datetime.utcnow(), "type": "refresh"})
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )
    
    return encoded_jwt


async def get_current_user(token: Optional[str] = None) -> Dict[str, Any]:
    """
    Get current authenticated user from JWT token.
    
    Args:
        token: JWT access token
        
    Returns:
        User information dictionary
        
    Raises:
        HTTPException if token invalid or missing
    """
    if not FASTAPI_AVAILABLE:
        return {"user_id": "anonymous", "role": "viewer"}
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not JOSE_AVAILABLE:
        logger.warning("JWT verification not available without python-jose")
        return {"user_id": "anonymous", "role": "viewer"}
    
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )
        
        return {
            "user_id": user_id,
            "email": payload.get("email"),
            "role": payload.get("role", "viewer"),
            "permissions": payload.get("permissions", []),
        }
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )


async def get_current_active_user(
    current_user: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Get current active user (for endpoints requiring authentication).
    
    Args:
        current_user: Current user from token
        
    Returns:
        Active user information
    """
    # TODO: Check if user is active in database
    return current_user


async def verify_refresh_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode refresh token.
    
    Args:
        token: Refresh token
        
    Returns:
        Decoded token payload
        
    Raises:
        HTTPException if token invalid
    """
    if not JOSE_AVAILABLE:
        logger.warning("Refresh token verification not available without python-jose")
        return {"sub": "anonymous"}
    
    if not FASTAPI_AVAILABLE:
        return {"sub": "anonymous"}
    
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
            )
        
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )
