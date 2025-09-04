"""Request Quota Management System.

This module provides a thread-safe request quota management system that can be used
to limit the number of requests made to external services or APIs. It supports:

- Persistent counters (Redis or SQLite)
- Thread-safe operations
- Configurable quotas and alert thresholds
- Time-windowed quotas
- Decorator for easy integration
"""

import time
import threading
import logging
import os
import json
import sqlite3
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Callable, Dict, Any, Union
from pathlib import Path

# Configure module logger
logger = logging.getLogger(__name__)

# Try to import Redis, but don't fail if it's not available
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class QuotaExceeded(Exception):
    """Exception raised when a quota limit is exceeded."""
    pass


class QuotaStore:
    """Abstract base class for quota storage backends."""
    
    def get_count(self, key: str) -> int:
        """Get the current count for a key."""
        raise NotImplementedError
    
    def increment(self, key: str, amount: int = 1) -> int:
        """Increment the count for a key and return the new value."""
        raise NotImplementedError
    
    def reset(self, key: str) -> None:
        """Reset the count for a key."""
        raise NotImplementedError
    
    def set_expiry(self, key: str, seconds: int) -> None:
        """Set an expiry time for a key."""
        raise NotImplementedError


class RedisQuotaStore(QuotaStore):
    """Redis-backed quota storage."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize Redis connection."""
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available. Install with 'pip install redis'")
        self.redis = redis.from_url(redis_url)
    
    def get_count(self, key: str) -> int:
        """Get the current count from Redis."""
        value = self.redis.get(key)
        return int(value) if value else 0
    
    def increment(self, key: str, amount: int = 1) -> int:
        """Increment the count in Redis."""
        return self.redis.incrby(key, amount)
    
    def reset(self, key: str) -> None:
        """Reset the count in Redis."""
        self.redis.delete(key)
    
    def set_expiry(self, key: str, seconds: int) -> None:
        """Set expiry time in Redis."""
        self.redis.expire(key, seconds)


class SQLiteQuotaStore(QuotaStore):
    """SQLite-backed quota storage."""
    
    def __init__(self, db_path: str = "quota.db"):
        """Initialize SQLite connection."""
        self.db_path = db_path
        self.lock = threading.RLock()
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                CREATE TABLE IF NOT EXISTS quotas (
                    key TEXT PRIMARY KEY,
                    count INTEGER NOT NULL DEFAULT 0,
                    expires_at TIMESTAMP
                )
                """)
                conn.commit()
    
    def get_count(self, key: str) -> int:
        """Get the current count from SQLite."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                # First, clean up expired entries
                conn.execute("""
                DELETE FROM quotas 
                WHERE expires_at IS NOT NULL AND expires_at < datetime('now')
                """)
                
                # Then get the count
                cursor = conn.execute("""
                SELECT count FROM quotas WHERE key = ?
                """, (key,))
                result = cursor.fetchone()
                return result[0] if result else 0
    
    def increment(self, key: str, amount: int = 1) -> int:
        """Increment the count in SQLite."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                # Insert or update the count
                conn.execute("""
                INSERT INTO quotas (key, count) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET count = count + ?
                """, (key, amount, amount))
                conn.commit()
                
                # Get the new count
                cursor = conn.execute("""
                SELECT count FROM quotas WHERE key = ?
                """, (key,))
                result = cursor.fetchone()
                return result[0] if result else amount
    
    def reset(self, key: str) -> None:
        """Reset the count in SQLite."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                DELETE FROM quotas WHERE key = ?
                """, (key,))
                conn.commit()
    
    def set_expiry(self, key: str, seconds: int) -> None:
        """Set expiry time in SQLite."""
        expiry_time = datetime.now() + timedelta(seconds=seconds)
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                UPDATE quotas 
                SET expires_at = datetime(?, 'unixepoch') 
                WHERE key = ?
                """, (expiry_time.timestamp(), key))
                conn.commit()


class MemoryQuotaStore(QuotaStore):
    """In-memory quota storage (for testing)."""
    
    def __init__(self):
        """Initialize in-memory storage."""
        self.counts = {}
        self.expiry = {}
        self.lock = threading.RLock()
    
    def get_count(self, key: str) -> int:
        """Get the current count from memory."""
        with self.lock:
            # Check expiry
            if key in self.expiry and self.expiry[key] < time.time():
                self.reset(key)
                return 0
            return self.counts.get(key, 0)
    
    def increment(self, key: str, amount: int = 1) -> int:
        """Increment the count in memory."""
        with self.lock:
            # Check expiry before incrementing
            if key in self.expiry and self.expiry[key] < time.time():
                self.reset(key)
                self.counts[key] = amount
            else:
                self.counts[key] = self.counts.get(key, 0) + amount
            return self.counts[key]
    
    def reset(self, key: str) -> None:
        """Reset the count in memory."""
        with self.lock:
            if key in self.counts:
                del self.counts[key]
            if key in self.expiry:
                del self.expiry[key]
    
    def set_expiry(self, key: str, seconds: int) -> None:
        """Set expiry time in memory."""
        with self.lock:
            self.expiry[key] = time.time() + seconds


class Alerter:
    """Handles alerts when quota thresholds are reached."""
    
    def __init__(self, slack_webhook: Optional[str] = None, email_config: Optional[Dict[str, str]] = None):
        """Initialize alerter with notification channels."""
        self.slack_webhook = slack_webhook
        self.email_config = email_config
        self.alert_history = set()  # Track which thresholds have been alerted
    
    def send_alert(self, message: str, level: str = "warning") -> None:
        """Send an alert through configured channels."""
        # Always log the alert
        if level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "critical":
            logger.critical(message)
        else:
            logger.info(message)
        
        # Send to Slack if configured
        if self.slack_webhook:
            self._send_slack_alert(message, level)
        
        # Send email if configured
        if self.email_config:
            self._send_email_alert(message, level)
    
    def _send_slack_alert(self, message: str, level: str) -> None:
        """Send alert to Slack webhook."""
        try:
            import requests
            color = {
                "info": "#36a64f",
                "warning": "#ffcc00",
                "error": "#ff0000",
                "critical": "#7b001c"
            }.get(level, "#36a64f")
            
            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"Quota Alert - {level.upper()}",
                        "text": message,
                        "ts": time.time()
                    }
                ]
            }
            
            requests.post(self.slack_webhook, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def _send_email_alert(self, message: str, level: str) -> None:
        """Send alert via email."""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            msg = MIMEMultipart()
            msg["From"] = self.email_config.get("from", "alerts@adaptive-traffic.com")
            msg["To"] = self.email_config.get("to", "admin@adaptive-traffic.com")
            msg["Subject"] = f"Quota Alert - {level.upper()}"
            
            msg.attach(MIMEText(message, "plain"))
            
            server = smtplib.SMTP(self.email_config.get("smtp_server", "localhost"))
            if self.email_config.get("use_tls", False):
                server.starttls()
            
            if "username" in self.email_config and "password" in self.email_config:
                server.login(self.email_config["username"], self.email_config["password"])
            
            server.send_message(msg)
            server.quit()
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")


class RequestQuotaManager:
    """Manages request quotas with persistent storage and alerting."""
    
    def __init__(self, 
                 store: Optional[QuotaStore] = None,
                 max_requests: int = 150, 
                 alert_at_remaining: int = 10,
                 key: str = "global",
                 window_hours: int = 24,
                 alerter: Optional[Alerter] = None,
                 config_path: Optional[str] = None):
        """Initialize the quota manager.
        
        Args:
            store: QuotaStore instance for persistence
            max_requests: Maximum number of requests allowed
            alert_at_remaining: Threshold for triggering alerts
            key: Identifier for this quota (for multi-quota support)
            window_hours: Time window for quota in hours
            alerter: Alerter instance for notifications
            config_path: Path to quota configuration file
        """
        # Load config from file if provided
        if config_path:
            self._load_config(config_path)
        else:
            self.max_requests = max_requests
            self.alert_at_remaining = alert_at_remaining
            self.key = key
            self.window_hours = window_hours
        
        # Initialize storage backend
        if store is None:
            # Try Redis first, fall back to SQLite
            if REDIS_AVAILABLE:
                try:
                    self.store = RedisQuotaStore()
                    logger.info("Using Redis for quota storage")
                except Exception as e:
                    logger.warning(f"Failed to connect to Redis: {e}. Falling back to SQLite.")
                    self.store = SQLiteQuotaStore("quota.db")
            else:
                self.store = SQLiteQuotaStore("quota.db")
                logger.info("Using SQLite for quota storage")
        else:
            self.store = store
        
        # Set up alerter
        self.alerter = alerter or Alerter()
        
        # Set up thread safety
        self.lock = threading.RLock()
        
        # Track alert levels that have been triggered
        self.alerted_levels = set()
        
        # Set expiry if using a time window
        if self.window_hours > 0:
            self.store.set_expiry(self.key, self.window_hours * 3600)
    
    def _load_config(self, config_path: str) -> None:
        """Load configuration from a file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.max_requests = config.get("max_requests", 150)
            self.alert_at_remaining = config.get("alert_threshold_remaining", 10)
            self.key = config.get("key", "global")
            self.window_hours = config.get("window", 24)
            
            logger.info(f"Loaded quota config from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load quota config from {config_path}: {e}")
            # Use defaults
            self.max_requests = 150
            self.alert_at_remaining = 10
            self.key = "global"
            self.window_hours = 24
    
    def remaining(self) -> int:
        """Get the number of remaining requests."""
        with self.lock:
            current = self.store.get_count(self.key)
            return max(0, self.max_requests - current)
    
    def increment(self, amount: int = 1) -> int:
        """Increment the request count and return remaining."""
        with self.lock:
            current = self.store.increment(self.key, amount)
            return max(0, self.max_requests - current)
    
    def check_and_alert(self, logger: Optional[logging.Logger] = None) -> None:
        """Check if we need to send alerts based on remaining quota."""
        remaining = self.remaining()
        logger = logger or logging.getLogger(__name__)
        
        # Define alert thresholds and their levels
        thresholds = {
            self.alert_at_remaining: "warning",  # First warning at configured threshold
            5: "warning",                       # Second warning at 5 remaining
            1: "error"                         # Final warning at 1 remaining
        }
        
        for threshold, level in thresholds.items():
            # Only alert if we're at or below the threshold and haven't alerted for this threshold yet
            alert_key = f"{self.key}_{threshold}"
            if remaining <= threshold and alert_key not in self.alerted_levels:
                message = f"Quota alert: {remaining} of {self.max_requests} requests remaining for {self.key}"
                self.alerter.send_alert(message, level)
                self.alerted_levels.add(alert_key)
                logger.log(
                    logging.WARNING if level == "warning" else logging.ERROR,
                    message
                )
    
    def reset(self) -> None:
        """Reset the quota counter."""
        with self.lock:
            self.store.reset(self.key)
            self.alerted_levels.clear()


def enforce_quota(quota_mgr: RequestQuotaManager):
    """Decorator to enforce quota limits on functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            remaining = quota_mgr.remaining()
            if remaining <= 0:
                raise QuotaExceeded(f"Quota exceeded for {quota_mgr.key}")
            
            result = func(*args, **kwargs)
            
            # Increment after successful execution
            quota_mgr.increment()
            quota_mgr.check_and_alert()
            
            return result
        return wrapper
    return decorator


# Middleware for web frameworks
def create_fastapi_quota_middleware(quota_mgr: RequestQuotaManager):
    """Create a FastAPI middleware for quota enforcement."""
    try:
        from fastapi import Request, Response
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import JSONResponse
        
        class QuotaMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                # Skip quota check for health endpoints
                if request.url.path.startswith("/health"):
                    return await call_next(request)
                
                remaining = quota_mgr.remaining()
                if remaining <= 0:
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Rate limit exceeded", "remaining": 0}
                    )
                
                response = await call_next(request)
                
                # Only count successful responses
                if 200 <= response.status_code < 400:
                    quota_mgr.increment()
                    quota_mgr.check_and_alert()
                
                # Add quota headers
                response.headers["X-RateLimit-Limit"] = str(quota_mgr.max_requests)
                response.headers["X-RateLimit-Remaining"] = str(quota_mgr.remaining())
                
                return response
        
        return QuotaMiddleware
    except ImportError:
        logger.warning("FastAPI not available, skipping middleware creation")
        return None