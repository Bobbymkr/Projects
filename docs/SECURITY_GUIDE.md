# Security Guide

This guide covers the security framework implemented in the Adaptive Traffic Control System API.

## Table of Contents

1. [Overview](#overview)
2. [Input Validation](#input-validation)
3. [Rate Limiting](#rate-limiting)
4. [Security Headers](#security-headers)
5. [API Key Authentication](#api-key-authentication)
6. [Security Audit Logging](#security-audit-logging)
7. [Best Practices](#best-practices)

---

## Overview

The security framework provides:
- Input validation and sanitization
- Rate limiting per client
- Security headers on all responses
- API key authentication (optional)
- Security audit logging

All security features are automatically applied to API routes.

---

## Input Validation

### Automatic Validation

Input validation is automatically applied to all POST/PUT endpoints in the traffic API:

```python
from src.api.security import InputValidator

# Validates intersection IDs
if not InputValidator.validate_intersection_id(intersection_id):
    raise HTTPException(400, "Invalid intersection ID")

# Validates queue lengths
if not InputValidator.validate_queue_lengths(queue_lengths):
    raise HTTPException(400, "Invalid queue lengths")

# Validates wait times
if not InputValidator.validate_wait_times(wait_times):
    raise HTTPException(400, "Invalid wait times")
```

### Validation Rules

**Intersection IDs:**
- Alphanumeric, hyphens, underscores only
- Maximum 100 characters
- Pattern: `^[a-zA-Z0-9_-]+$`

**Queue Lengths:**
- List of floats/integers
- Values: 0 to 1000
- Maximum 20 lanes

**Wait Times:**
- List of floats/integers
- Values: 0 to 3600 seconds (1 hour max)
- Maximum 20 lanes

### String Sanitization

```python
from src.api.security import InputValidator

# Sanitize user input
sanitized = InputValidator.sanitize_string(user_input, max_length=1000)
# Removes control characters, enforces length limits
```

---

## Rate Limiting

### In-Memory Rate Limiter

```python
from src.api.security import RateLimiter

limiter = RateLimiter(max_requests=100, window_seconds=60)

# Check if request is allowed
if limiter.is_allowed(client_id):
    # Process request
    pass
else:
    # Rate limited
    raise HTTPException(429, "Rate limit exceeded")

# Get remaining requests
remaining = limiter.get_remaining(client_id)
```

### Configuration

- **`max_requests`**: Maximum requests per window (default: 100)
- **`window_seconds`**: Time window in seconds (default: 60)

### Production Considerations

For distributed deployments, use Redis-based rate limiting:

```python
# Future: Redis-based rate limiting
from src.api.rate_limiting import check_rate_limit  # Uses Redis if available
```

---

## Security Headers

Security headers are automatically applied to all API responses via middleware.

### Headers Applied

- **X-Content-Type-Options**: `nosniff` - Prevents MIME type sniffing
- **X-Frame-Options**: `DENY` - Prevents clickjacking
- **X-XSS-Protection**: `1; mode=block` - XSS protection
- **Strict-Transport-Security**: `max-age=31536000; includeSubDomains` - HSTS
- **Content-Security-Policy**: `default-src 'self'` - CSP

### Customization

```python
from src.api.security import SecurityHeaders

# Get security headers
headers = SecurityHeaders.get_security_headers()

# Customize if needed
headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'"
```

---

## API Key Authentication

### Using the Decorator

```python
from src.api.security import require_api_key
from fastapi import APIRouter

router = APIRouter()

@router.post("/protected-endpoint")
@require_api_key
async def protected_endpoint(request: Request):
    # This endpoint requires API key
    return {"message": "Authenticated"}
```

### Setting API Key

```bash
# Set API key as environment variable
export API_KEY=your-secret-api-key-here
```

### Client Usage

```bash
# Include API key in request header
curl -H "X-API-Key: your-secret-api-key-here" \
     http://localhost:8000/api/v1/traffic/decision
```

### Secrets Management

```python
from src.api.security import SecretsManager

manager = SecretsManager()

# Get secret
api_key = manager.get_secret("api_key")

# Verify API key (constant-time comparison)
is_valid = manager.verify_api_key(provided_key)

# Generate secure token
token = manager.generate_token(length=32)
```

**Production Note**: For production, integrate with:
- AWS Secrets Manager
- HashiCorp Vault
- Azure Key Vault
- Google Secret Manager

---

## Security Audit Logging

### Logging Security Events

```python
from src.api.security import SecurityAuditLogger

# Log security event
SecurityAuditLogger.log_security_event(
    event_type="invalid_api_key",
    details={"ip": "192.168.1.1", "endpoint": "/api/v1/traffic/decision"},
    severity="warning"  # "info", "warning", "error", "critical"
)
```

### Event Types

Common event types:
- `invalid_api_key` - Invalid API key attempt
- `invalid_input` - Invalid input validation failure
- `rate_limit_exceeded` - Rate limit violation
- `authentication_failure` - Authentication error
- `authorization_failure` - Authorization error

### Severity Levels

- **`info`**: Normal security events (e.g., successful authentication)
- **`warning`**: Suspicious activity (e.g., invalid input, rate limiting)
- **`error`**: Security errors (e.g., authentication failures)
- **`critical`**: Critical security issues (e.g., potential attacks)

---

## Best Practices

### 1. Always Validate Input

```python
from src.api.security import InputValidator
from fastapi import HTTPException

@router.post("/endpoint")
async def endpoint(request: Request):
    # Validate all inputs
    if not InputValidator.validate_intersection_id(request.intersection_id):
        raise HTTPException(400, "Invalid intersection ID")
    # ... process request
```

### 2. Use Rate Limiting

```python
from src.api.dependencies import rate_limit

@router.post("/endpoint")
async def endpoint(_rate_limit: None = Depends(rate_limit)):
    # Rate limiting applied automatically
    # ... process request
```

### 3. Apply Security Headers

Security headers are applied automatically via middleware. No action needed.

### 4. Log Security Events

```python
from src.api.security import SecurityAuditLogger

try:
    # Process request
    pass
except ValueError as e:
    SecurityAuditLogger.log_security_event(
        "invalid_input",
        {"error": str(e), "endpoint": request.url.path},
        severity="warning"
    )
    raise HTTPException(400, str(e))
```

### 5. Use API Keys for Production

```python
@router.post("/sensitive-endpoint")
@require_api_key
async def sensitive_endpoint(request: Request):
    # Protected endpoint
    pass
```

### 6. Sanitize User Input

```python
from src.api.security import InputValidator

user_input = InputValidator.sanitize_string(raw_input, max_length=1000)
```

---

## Testing Security

### Run Security Tests

```bash
pytest tests/security/test_security.py -v
```

### Test Coverage

Security tests cover:
- Input validation (intersection IDs, queue lengths, wait times)
- Rate limiting (per-client limits, window management)
- Security headers
- Secrets management
- API key verification
- Security audit logging

---

## Production Deployment

### Environment Variables

```bash
# API Key
export API_KEY=your-production-api-key

# Rate Limiting (if using Redis)
export REDIS_URL=redis://localhost:6379

# Security Settings
export SECURITY_HEADERS_ENABLED=true
export RATE_LIMITING_ENABLED=true
```

### Monitoring

Monitor security events via:
- Security audit logs
- Prometheus metrics (if enabled)
- Application logs

### Incident Response

1. Check security audit logs for suspicious activity
2. Review rate limiting violations
3. Analyze invalid input patterns
4. Monitor API key usage

---

## References

- OWASP Top 10
- CWE Top 25
- NIST Cybersecurity Framework
- FastAPI Security Best Practices

