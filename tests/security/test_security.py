"""
Security testing suite.

Tests for input validation, rate limiting, and security features.
"""

import pytest
from fastapi import HTTPException

from src.api.security import (
    InputValidator,
    RateLimiter,
    SecurityHeaders,
    SecretsManager,
    SecurityAuditLogger,
)


class TestInputValidator:
    """Test input validation."""
    
    def test_validate_intersection_id_valid(self):
        """Test valid intersection IDs."""
        assert InputValidator.validate_intersection_id("intersection_1")
        assert InputValidator.validate_intersection_id("main-street-1")
        assert InputValidator.validate_intersection_id("ABC123")
    
    def test_validate_intersection_id_invalid(self):
        """Test invalid intersection IDs."""
        assert not InputValidator.validate_intersection_id("")
        assert not InputValidator.validate_intersection_id(None)
        assert not InputValidator.validate_intersection_id("intersection@1")  # Invalid char
        assert not InputValidator.validate_intersection_id("a" * 101)  # Too long
    
    def test_validate_queue_lengths_valid(self):
        """Test valid queue lengths."""
        assert InputValidator.validate_queue_lengths([0.0, 1.0, 2.5])
        assert InputValidator.validate_queue_lengths([10, 20, 30])
        assert InputValidator.validate_queue_lengths([0])
    
    def test_validate_queue_lengths_invalid(self):
        """Test invalid queue lengths."""
        assert not InputValidator.validate_queue_lengths([])
        assert not InputValidator.validate_queue_lengths([-1.0])
        assert not InputValidator.validate_queue_lengths([1001])  # Too large
        assert not InputValidator.validate_queue_lengths([0.0] * 21)  # Too many
        assert not InputValidator.validate_queue_lengths("not a list")
    
    def test_validate_wait_times_valid(self):
        """Test valid wait times."""
        assert InputValidator.validate_wait_times([0.0, 10.0, 30.0])
        assert InputValidator.validate_wait_times([100, 200])
    
    def test_validate_wait_times_invalid(self):
        """Test invalid wait times."""
        assert not InputValidator.validate_wait_times([])
        assert not InputValidator.validate_wait_times([-1.0])
        assert not InputValidator.validate_wait_times([3601])  # Too large
        assert not InputValidator.validate_wait_times("not a list")
    
    def test_sanitize_string(self):
        """Test string sanitization."""
        assert InputValidator.sanitize_string("normal string") == "normal string"
        assert InputValidator.sanitize_string("test\x00null") == "testnull"
        assert len(InputValidator.sanitize_string("a" * 2000, max_length=100)) == 100


class TestRateLimiter:
    """Test rate limiting."""
    
    def test_rate_limiter_allows_requests(self):
        """Test that rate limiter allows requests within limit."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        
        for i in range(5):
            assert limiter.is_allowed("client1")
        
        # Should be rate limited
        assert not limiter.is_allowed("client1")
    
    def test_rate_limiter_different_clients(self):
        """Test that different clients have separate limits."""
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        
        # Client 1 uses all requests
        for i in range(3):
            assert limiter.is_allowed("client1")
        assert not limiter.is_allowed("client1")
        
        # Client 2 should still have requests
        assert limiter.is_allowed("client2")
    
    def test_rate_limiter_remaining(self):
        """Test remaining requests calculation."""
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        
        assert limiter.get_remaining("client1") == 10
        
        for i in range(3):
            limiter.is_allowed("client1")
        
        assert limiter.get_remaining("client1") == 7


class TestSecurityHeaders:
    """Test security headers."""
    
    def test_get_security_headers(self):
        """Test security headers generation."""
        headers = SecurityHeaders.get_security_headers()
        
        assert "X-Content-Type-Options" in headers
        assert "X-Frame-Options" in headers
        assert "X-XSS-Protection" in headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"


class TestSecretsManager:
    """Test secrets management."""
    
    def test_generate_token(self):
        """Test token generation."""
        manager = SecretsManager()
        
        token1 = manager.generate_token(16)
        token2 = manager.generate_token(16)
        
        assert len(token1) == 32  # Hex encoding of 16 bytes
        assert token1 != token2  # Should be different
    
    def test_verify_api_key(self):
        """Test API key verification."""
        import os
        
        manager = SecretsManager()
        
        # Set test API key
        test_key = "test-api-key-123"
        os.environ["API_KEY"] = test_key
        manager._load_secrets()
        
        assert manager.verify_api_key(test_key)
        assert not manager.verify_api_key("wrong-key")
        assert not manager.verify_api_key("")


class TestSecurityAuditLogger:
    """Test security audit logging."""
    
    def test_log_security_event(self, caplog):
        """Test security event logging."""
        SecurityAuditLogger.log_security_event(
            "test_event",
            {"detail": "test"},
            severity="info"
        )
        
        assert "Security Event" in caplog.text
    
    def test_log_critical_event(self, caplog):
        """Test critical security event logging."""
        SecurityAuditLogger.log_security_event(
            "critical_event",
            {"detail": "critical"},
            severity="critical"
        )
        
        assert "Security Event" in caplog.text

