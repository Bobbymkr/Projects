"""
Security Utilities for API.

Implements Phase 8 from SCORE_IMPROVEMENT_ROADMAP.md:
- Input validation and sanitization
- Rate limiting
- Authentication/authorization helpers
- Security headers
- Secrets management
"""

import logging
import hashlib
import hmac
import secrets
import re
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from functools import wraps

logger = logging.getLogger(__name__)


class InputValidator:
    """Input validation and sanitization utilities."""
    
    @staticmethod
    def validate_intersection_id(intersection_id: str) -> bool:
        """
        Validate intersection ID format.
        
        Args:
            intersection_id: Intersection identifier
            
        Returns:
            True if valid, False otherwise
        """
        if not intersection_id or not isinstance(intersection_id, str):
            return False
        
        # Allow alphanumeric, hyphens, underscores
        pattern = r'^[a-zA-Z0-9_-]+$'
        return bool(re.match(pattern, intersection_id)) and len(intersection_id) <= 100
    
    @staticmethod
    def validate_queue_lengths(queue_lengths: List[float]) -> bool:
        """
        Validate queue length values.
        
        Args:
            queue_lengths: List of queue lengths
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(queue_lengths, list):
            return False
        
        if len(queue_lengths) == 0 or len(queue_lengths) > 20:  # Reasonable limit
            return False
        
        for q in queue_lengths:
            if not isinstance(q, (int, float)):
                return False
            if q < 0 or q > 1000:  # Reasonable bounds
                return False
        
        return True
    
    @staticmethod
    def validate_wait_times(wait_times: List[float]) -> bool:
        """
        Validate wait time values.
        
        Args:
            wait_times: List of wait times
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(wait_times, list):
            return False
        
        if len(wait_times) == 0 or len(wait_times) > 20:
            return False
        
        for w in wait_times:
            if not isinstance(w, (int, float)):
                return False
            if w < 0 or w > 3600:  # Max 1 hour wait
                return False
        
        return True
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """
        Sanitize string input.
        
        Args:
            value: String to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            return ""
        
        # Remove null bytes and control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
        
        # Truncate to max length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[datetime]] = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """
        Check if request is allowed.
        
        Args:
            identifier: Client identifier (IP, user ID, etc.)
            
        Returns:
            True if allowed, False if rate limited
        """
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        # Clean old entries
        if identifier in self.requests:
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if req_time > window_start
            ]
        else:
            self.requests[identifier] = []
        
        # Check limit
        if len(self.requests[identifier]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[identifier].append(now)
        return True
    
    def get_remaining(self, identifier: str) -> int:
        """
        Get remaining requests in current window.
        
        Args:
            identifier: Client identifier
            
        Returns:
            Number of remaining requests
        """
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        if identifier not in self.requests:
            return self.max_requests
        
        recent_requests = [
            req_time for req_time in self.requests[identifier]
            if req_time > window_start
        ]
        
        return max(0, self.max_requests - len(recent_requests))


class SecurityHeaders:
    """Security headers for API responses."""
    
    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """
        Get standard security headers.
        
        Returns:
            Dictionary of security headers
        """
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
        }


class SecretsManager:
    """Simple secrets management (for development; use proper secret management in production)."""
    
    def __init__(self):
        """Initialize secrets manager."""
        self.secrets: Dict[str, str] = {}
        self._load_secrets()
    
    def _load_secrets(self):
        """Load secrets from environment or config file."""
        import os
        
        # Load from environment variables
        api_key = os.getenv("API_KEY")
        if api_key:
            self.secrets["api_key"] = api_key
        
        # In production, use proper secret management (AWS Secrets Manager, HashiCorp Vault, etc.)
        logger.info("Secrets manager initialized")
    
    def get_secret(self, key: str) -> Optional[str]:
        """
        Get secret value.
        
        Args:
            key: Secret key
            
        Returns:
            Secret value or None
        """
        return self.secrets.get(key)
    
    def verify_api_key(self, provided_key: str) -> bool:
        """
        Verify API key.
        
        Args:
            provided_key: Provided API key
            
        Returns:
            True if valid, False otherwise
        """
        expected_key = self.get_secret("api_key")
        if not expected_key:
            return False
        
        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(provided_key, expected_key)
    
    def generate_token(self, length: int = 32) -> str:
        """
        Generate secure random token.
        
        Args:
            length: Token length in bytes
            
        Returns:
            Hex-encoded token
        """
        return secrets.token_hex(length)


class SecurityAuditLogger:
    """Security audit logging."""
    
    @staticmethod
    def log_security_event(
        event_type: str,
        details: Dict[str, Any],
        severity: str = "info"
    ):
        """
        Log security event.
        
        Args:
            event_type: Type of security event
            details: Event details
            severity: Severity level (info, warning, error, critical)
        """
        log_message = f"Security Event [{event_type}]: {details}"
        
        if severity == "critical":
            logger.critical(log_message)
        elif severity == "error":
            logger.error(log_message)
        elif severity == "warning":
            logger.warning(log_message)
        else:
            logger.info(log_message)


def require_api_key(func):
    """Decorator to require API key authentication."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        from fastapi import Request, HTTPException
        from ..security import SecretsManager
        
        request: Request = kwargs.get('request') or (args[0] if args and hasattr(args[0], 'headers') else None)
        
        if not request:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")
        
        secrets_manager = SecretsManager()
        if not secrets_manager.verify_api_key(api_key):
            SecurityAuditLogger.log_security_event(
                "invalid_api_key",
                {"ip": request.client.host if request.client else "unknown"},
                severity="warning"
            )
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        return await func(*args, **kwargs)
    
    return wrapper

