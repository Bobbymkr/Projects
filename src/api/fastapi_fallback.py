"""
FastAPI Fallback Classes for Graceful Degradation.

Provides comprehensive mock implementations of FastAPI components
that maintain protocol compatibility while allowing the codebase
to function without FastAPI installed.

These fallbacks are designed for:
- Development environments without FastAPI
- Testing scenarios
- Graceful degradation when FastAPI is unavailable

NOT intended for production use - production should require FastAPI.
"""

import logging
from typing import Any, Optional, Callable, Dict, List, Coroutine
from urllib.parse import urlparse, parse_qs
from functools import wraps

logger = logging.getLogger(__name__)

# ============================================================================
# Type Protocol Definitions for Type Safety
# ============================================================================

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    # Python < 3.8 fallback
    Protocol = object
    runtime_checkable = lambda x: x


@runtime_checkable
class FastAPIProtocol(Protocol):
    """Protocol defining FastAPI interface for type checking."""
    routes: list
    title: str
    version: str
    
    def include_router(self, router: Any, *, prefix: str = "", tags: Optional[list] = None) -> None: ...
    def add_middleware(self, middleware: type, *args: Any, **kwargs: Any) -> None: ...
    def get(self, path: str, *args: Any, **kwargs: Any) -> Callable: ...


# ============================================================================
# URL and Client Mock Classes
# ============================================================================

class MockURL:
    """Comprehensive URL mock with all FastAPI Request.url attributes."""
    
    def __init__(self, path: str = "/", query: str = ""):
        """
        Initialize mock URL.
        
        Args:
            path: URL path
            query: Query string
        """
        self.path = path
        self.query = query
        
        # Parse URL properly
        full_url = f"http://localhost{path}"
        if query:
            full_url += f"?{query}"
        
        parsed = urlparse(full_url)
        self.scheme = parsed.scheme or "http"
        self.netloc = parsed.netloc or "localhost"
        self.pathname = parsed.path
        self.query_string = query
        
        # Parse query params
        self.query_params = {}
        if query:
            self.query_params = dict(parse_qs(query))


class MockClient:
    """Mock client information."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        """Initialize mock client."""
        self.host = host
        self.port = port


# ============================================================================
# Request Mock Class
# ============================================================================

class MockRequest:
    """
    Comprehensive Request mock with all FastAPI Request attributes.
    
    Implements the essential interface used throughout the codebase.
    """
    
    def __init__(
        self,
        path: str = "/",
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        query: str = "",
    ):
        """
        Initialize mock request.
        
        Args:
            path: Request path
            method: HTTP method
            headers: Request headers
            query: Query string
        """
        self.url = MockURL(path, query)
        self.method = method
        self.headers = headers or {}
        self.cookies: Dict[str, str] = {}
        self.client = MockClient()
        
        # Request state for storing custom data
        class RequestState:
            """Minimal request state object."""
            pass
        self.state = RequestState()
        
        # Path and query parameters
        self.path_params: Dict[str, Any] = {}
        self.query_params = self.url.query_params
        
        # Body storage
        self._body = b""
        self._json: Optional[Dict[str, Any]] = None
    
    async def json(self) -> Dict[str, Any]:
        """Parse JSON body."""
        if self._json is not None:
            return self._json
        
        if self._body:
            try:
                import json
                self._json = json.loads(self._body)
                return self._json
            except (json.JSONDecodeError, ValueError):
                return {}
        return {}
    
    async def body(self) -> bytes:
        """Get raw body."""
        return self._body
    
    def set_body(self, body: bytes) -> None:
        """Set request body."""
        self._body = body
        self._json = None  # Reset cached JSON
    
    def set_json(self, data: Dict[str, Any]) -> None:
        """Set JSON data directly."""
        self._json = data
        try:
            import json
            self._body = json.dumps(data).encode()
        except Exception:
            pass


# ============================================================================
# FastAPI Application Mock
# ============================================================================

class MockFastAPI:
    """
    Comprehensive FastAPI application mock.
    
    Maintains protocol compatibility with real FastAPI while providing
    mock functionality for testing and development.
    """
    
    def __init__(self, *args: Any, **kwargs: Any):
        """
        Initialize mock FastAPI app.
        
        Args:
            *args: Positional arguments (ignored)
            **kwargs: Keyword arguments including title, version, etc.
        """
        self.routes: List[Dict[str, Any]] = []
        self.title = kwargs.get("title", "API (Mock)")
        self.version = kwargs.get("version", "1.0.0")
        self.description = kwargs.get("description", "")
        self.middleware_stack: List[str] = []
        self._router_map: Dict[str, Any] = {}
        self._openapi_schema: Optional[Dict[str, Any]] = None
        
        logger.debug(f"MockFastAPI initialized: {self.title} v{self.version}")
    
    def include_router(
        self,
        router: Any,
        *,
        prefix: str = "",
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Track router registration for debugging and introspection.
        
        Args:
            router: Router to include
            prefix: URL prefix
            tags: Route tags
            **kwargs: Additional router options
        """
        router_info = {
            "prefix": prefix,
            "tags": tags or [],
            "router": router,
        }
        self._router_map[prefix] = router_info
        
        # Track routes if router has them
        if hasattr(router, 'routes'):
            for route in router.routes:
                route_info = {
                    "path": getattr(route, 'path', prefix),
                    "method": getattr(route, 'methods', set()),
                    "prefix": prefix,
                    "tags": tags or [],
                }
                self.routes.append(route_info)
        
        logger.debug(f"Mock router registered at prefix: {prefix} with tags: {tags}")
    
    def add_middleware(
        self,
        middleware: type,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Track middleware registration.
        
        Args:
            middleware: Middleware class
            *args: Middleware arguments
            **kwargs: Middleware keyword arguments
        """
        middleware_name = getattr(middleware, '__name__', str(middleware))
        self.middleware_stack.append(middleware_name)
        logger.debug(f"Mock middleware added: {middleware_name}")
    
    def get(self, path: str, *args: Any, **kwargs: Any) -> Callable:
        """
        Register a GET route.
        
        Args:
            path: Route path
            *args: Route arguments
            **kwargs: Route options
        
        Returns:
            Decorator function
        """
        return self._create_route_decorator("GET", path, *args, **kwargs)
    
    def post(self, path: str, *args: Any, **kwargs: Any) -> Callable:
        """Register a POST route."""
        return self._create_route_decorator("POST", path, *args, **kwargs)
    
    def put(self, path: str, *args: Any, **kwargs: Any) -> Callable:
        """Register a PUT route."""
        return self._create_route_decorator("PUT", path, *args, **kwargs)
    
    def delete(self, path: str, *args: Any, **kwargs: Any) -> Callable:
        """Register a DELETE route."""
        return self._create_route_decorator("DELETE", path, *args, **kwargs)
    
    def patch(self, path: str, *args: Any, **kwargs: Any) -> Callable:
        """Register a PATCH route."""
        return self._create_route_decorator("PATCH", path, *args, **kwargs)
    
    def _create_route_decorator(
        self,
        method: str,
        path: str,
        *args: Any,
        **kwargs: Any,
    ) -> Callable:
        """Create a route decorator for the given method."""
        def decorator(func: Callable) -> Callable:
            route_info = {
                "path": path,
                "method": method,
                "handler": func.__name__,
                "dependencies": kwargs.get("dependencies", []),
                "tags": kwargs.get("tags", []),
            }
            self.routes.append(route_info)
            logger.debug(f"Mock route registered: {method} {path}")
            return func
        return decorator


# ============================================================================
# API Router Mock
# ============================================================================

class MockAPIRouter:
    """
    Comprehensive APIRouter mock.
    
    Maintains compatibility with FastAPI's APIRouter interface.
    """
    
    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize mock API router."""
        self.routes: List[Dict[str, Any]] = []
        self.prefix = kwargs.get("prefix", "")
        self.tags = kwargs.get("tags", [])
        
        logger.debug(f"MockAPIRouter initialized with prefix: {self.prefix}")
    
    def include_router(
        self,
        router: Any,
        *,
        prefix: str = "",
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Track router inclusion."""
        logger.debug(f"Mock router included at prefix: {prefix}")
        if hasattr(router, 'routes'):
            self.routes.extend(router.routes)
    
    def get(self, path: str, *args: Any, **kwargs: Any) -> Callable:
        """Register GET route."""
        return self._create_route_decorator("GET", path, *args, **kwargs)
    
    def post(self, path: str, *args: Any, **kwargs: Any) -> Callable:
        """Register POST route."""
        return self._create_route_decorator("POST", path, *args, **kwargs)
    
    def put(self, path: str, *args: Any, **kwargs: Any) -> Callable:
        """Register PUT route."""
        return self._create_route_decorator("PUT", path, *args, **kwargs)
    
    def delete(self, path: str, *args: Any, **kwargs: Any) -> Callable:
        """Register DELETE route."""
        return self._create_route_decorator("DELETE", path, *args, **kwargs)
    
    def patch(self, path: str, *args: Any, **kwargs: Any) -> Callable:
        """Register PATCH route."""
        return self._create_route_decorator("PATCH", path, *args, **kwargs)
    
    def _create_route_decorator(
        self,
        method: str,
        path: str,
        *args: Any,
        **kwargs: Any,
    ) -> Callable:
        """Create route decorator."""
        def decorator(func: Callable) -> Callable:
            self.routes.append({
                "path": path,
                "method": method,
                "handler": func.__name__,
            })
            return func
        return decorator


# ============================================================================
# Middleware and Response Mocks
# ============================================================================

class MockCORSMiddleware:
    """Mock CORS middleware."""
    pass


class MockJSONResponse:
    """Mock JSONResponse class."""
    
    def __init__(self, content: Any, *args: Any, **kwargs: Any):
        """Initialize mock JSON response."""
        self.content = content
        self.status_code = kwargs.get("status_code", 200)
        self.headers = kwargs.get("headers", {})
    
    def __call__(self, *args: Any, **kwargs: Any) -> "MockJSONResponse":
        """Make instance callable."""
        return self


class HTMLResponse:
    """Mock HTMLResponse class."""
    
    def __init__(self, content: str, *args: Any, **kwargs: Any):
        """Initialize mock HTML response."""
        self.content = content
        self.status_code = kwargs.get("status_code", 200)


# ============================================================================
# Exception Classes
# ============================================================================

class HTTPException(Exception):
    """
    Mock HTTPException compatible with FastAPI's HTTPException.
    """
    
    def __init__(
        self,
        status_code: int,
        detail: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize HTTP exception.
        
        Args:
            status_code: HTTP status code
            detail: Exception detail message
            headers: Optional headers
        """
        self.status_code = status_code
        self.detail = detail or f"HTTP {status_code} Error"
        self.headers = headers or {}
        super().__init__(self.detail)


class StatusCodes:
    """HTTP status code constants matching FastAPI's status module."""
    
    HTTP_100_CONTINUE = 100
    HTTP_101_SWITCHING_PROTOCOLS = 101
    HTTP_200_OK = 200
    HTTP_201_CREATED = 201
    HTTP_202_ACCEPTED = 202
    HTTP_204_NO_CONTENT = 204
    HTTP_400_BAD_REQUEST = 400
    HTTP_401_UNAUTHORIZED = 401
    HTTP_403_FORBIDDEN = 403
    HTTP_404_NOT_FOUND = 404
    HTTP_405_METHOD_NOT_ALLOWED = 405
    HTTP_422_UNPROCESSABLE_ENTITY = 422
    HTTP_429_TOO_MANY_REQUESTS = 429
    HTTP_500_INTERNAL_SERVER_ERROR = 500
    HTTP_502_BAD_GATEWAY = 502
    HTTP_503_SERVICE_UNAVAILABLE = 503


status = StatusCodes()


# ============================================================================
# Dependency Injection Mock
# ============================================================================

class MockDepends:
    """
    Mock dependency injection that properly handles async dependencies.
    """
    
    def __init__(self, dependency: Any, *args: Any, **kwargs: Any):
        """
        Initialize dependency wrapper.
        
        Args:
            dependency: Dependency function/class
            *args: Additional arguments
            **kwargs: Dependency options (use_cache, etc.)
        """
        self.dependency = dependency
        self.use_cache = kwargs.get("use_cache", True)
    
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute dependency if it's a callable.
        
        Properly handles both sync and async dependencies.
        """
        if callable(self.dependency):
            result = self.dependency(*args, **kwargs)
            # Handle async dependencies
            if isinstance(result, Coroutine):
                return await result
            return result
        return self.dependency


def Depends(*args: Any, **kwargs: Any) -> Any:
    """
    Create dependency wrapper.
    
    Usage:
        Depends(get_dependency)  # Function as dependency
        Depends(DependencyClass)  # Class as dependency
    """
    if args and callable(args[0]):
        # Function/class passed directly
        return MockDepends(args[0], **kwargs)
    elif args:
        # Dependency passed as first argument
        return MockDepends(*args, **kwargs)
    else:
        # Return a factory function
        return lambda dep: MockDepends(dep, **kwargs)


def Header(default: Any = None, *args: Any, **kwargs: Any) -> Any:
    """
    Mock Header dependency.
    
    Returns None or default value when FastAPI not available.
    """
    return default


class OAuth2PasswordBearer:
    """Mock OAuth2PasswordBearer for authentication."""
    
    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize mock OAuth2 scheme."""
        self.tokenUrl = kwargs.get("tokenUrl", "/token")
        self.auto_error = kwargs.get("auto_error", True)
    
    async def __call__(self, request: Any) -> Optional[str]:
        """Extract token from request."""
        # Mock implementation returns None
        return None


# ============================================================================
# Health Check Function
# ============================================================================

def check_fastapi_health() -> tuple[bool, Optional[str]]:
    """
    Check if FastAPI is properly installed and functional.
    
    Returns:
        (is_available, error_message): Tuple indicating availability and any error
    
    This function performs a comprehensive health check:
    - Verifies FastAPI can be imported
    - Checks critical components exist
    - Attempts to create a minimal app to verify functionality
    """
    try:
        # Try to import FastAPI
        import fastapi
        
        # Verify critical components exist
        required_components = [
            'FastAPI',
            'APIRouter',
            'Request',
            'Depends',
            'HTTPException',
        ]
        
        missing = []
        for component in required_components:
            if not hasattr(fastapi, component):
                missing.append(component)
        
        if missing:
            return False, f"FastAPI installation incomplete. Missing: {', '.join(missing)}"
        
        # Try to create a minimal app to verify functionality
        try:
            test_app = fastapi.FastAPI(title="Health Check")
            # Try to create a router
            test_router = fastapi.APIRouter()
            test_app.include_router(test_router)
        except Exception as e:
            return False, f"FastAPI functionality test failed: {str(e)}"
        
        return True, None
        
    except ImportError as e:
        return False, f"FastAPI not installed: {str(e)}"
    except Exception as e:
        return False, f"FastAPI health check failed: {str(e)}"


# ============================================================================
# Availability Check
# ============================================================================

def get_fastapi_available() -> bool:
    """
    Lazy check for FastAPI availability.
    
    Returns:
        True if FastAPI is available and functional
    """
    available, _ = check_fastapi_health()
    return available

