"""
API Versioning System.

Supports multiple API versions with backward compatibility and migration paths.
"""

from typing import Optional
import re
import logging

logger = logging.getLogger(__name__)

# Try to import FastAPI Request, but handle gracefully if not available
try:
    from fastapi import Request
    FASTAPI_AVAILABLE = True
except ImportError:
    # Fallback for testing without FastAPI
    class Request:
        def __init__(self):
            self.url = type('obj', (object,), {'path': '/api/v1/test'})()
            self.headers = {}
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available. API versioning will have limited functionality.")


class APIVersion:
    """API Version handler."""
    
    def __init__(self, version: str):
        """
        Initialize API version.
        
        Args:
            version: Version string (e.g., "v1", "v2")
        """
        self.version = version
        self.major, self.minor = self._parse_version(version)
    
    def _parse_version(self, version: str) -> tuple[int, int]:
        """Parse version string into major and minor numbers."""
        match = re.match(r"v(\d+)(?:\.(\d+))?", version)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2)) if match.group(2) else 0
            return major, minor
        return 1, 0
    
    def __eq__(self, other):
        """Check version equality."""
        if isinstance(other, str):
            other = APIVersion(other)
        return self.major == other.major and self.minor == other.minor
    
    def __lt__(self, other):
        """Check if version is less than other."""
        if isinstance(other, str):
            other = APIVersion(other)
        if self.major < other.major:
            return True
        if self.major == other.major:
            return self.minor < other.minor
        return False
    
    def __le__(self, other):
        """Check if version is less than or equal to other."""
        return self < other or self == other
    
    def __gt__(self, other):
        """Check if version is greater than other."""
        return not self <= other
    
    def __ge__(self, other):
        """Check if version is greater than or equal to other."""
        return not self < other


def get_api_version(request: Request) -> str:
    """
    Extract API version from request.
    
    Checks:
    1. URL path (/api/v1/...)
    2. Accept header (application/vnd.api+json;version=v1)
    3. X-API-Version header
    
    Returns:
        API version string (default: "v1")
    """
    # Check URL path
    path = request.url.path
    version_match = re.search(r"/api/v(\d+(?:\.\d+)?)/", path)
    if version_match:
        return f"v{version_match.group(1)}"
    
    # Check Accept header
    accept_header = request.headers.get("Accept", "")
    version_match = re.search(r"version=([\d.]+)", accept_header)
    if version_match:
        return f"v{version_match.group(1)}"
    
    # Check X-API-Version header
    api_version_header = request.headers.get("X-API-Version")
    if api_version_header:
        return api_version_header if api_version_header.startswith("v") else f"v{api_version_header}"
    
    # Default version
    return "v1"


def validate_api_version(version: str, supported_versions: list[str]) -> bool:
    """
    Validate if API version is supported.
    
    Args:
        version: Version to validate
        supported_versions: List of supported versions
        
    Returns:
        True if version is supported
    """
    try:
        api_version = APIVersion(version)
        supported = [APIVersion(v) for v in supported_versions]
        return api_version in supported
    except Exception:
        return False


def create_versioned_router(
    base_path: str,
    versions: list[str],
    default_version: str = "v1",
):
    """
    Create a versioned API router.
    
    Args:
        base_path: Base path for router
        versions: List of supported versions
        default_version: Default version to use
        
    Returns:
        Configured APIRouter (if FastAPI available)
    """
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available. Cannot create versioned router.")
        return None
    
    from fastapi import APIRouter, HTTPException, status
    
    router = APIRouter()
    
    # Add version checking middleware
    @router.middleware("http")
    async def version_check_middleware(request: Request, call_next):
        version = get_api_version(request)
        
        if not validate_api_version(version, versions):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported API version: {version}. Supported versions: {', '.join(versions)}",
            )
        
        # Add version to request state
        request.state.api_version = version
        
        response = await call_next(request)
        response.headers["X-API-Version"] = version
        
        return response
    
    return router


# Supported API versions
SUPPORTED_VERSIONS = ["v1"]

# Current API version
CURRENT_VERSION = "v1"

# Deprecated versions (with deprecation dates)
DEPRECATED_VERSIONS = {}
