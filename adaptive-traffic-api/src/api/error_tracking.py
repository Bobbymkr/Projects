"""
Error Tracking and Monitoring with Sentry.

Integration with Sentry for error tracking, performance monitoring,
and alerting in production environments.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_sentry_initialized = False


def setup_error_tracking(dsn: Optional[str] = None, environment: str = "development") -> None:
    """
    Setup Sentry error tracking and performance monitoring.
    
    Args:
        dsn: Sentry DSN (Data Source Name) - if None, Sentry is disabled
        environment: Environment name (development, staging, production)
    """
    global _sentry_initialized
    
    if not dsn:
        logger.info("Sentry DSN not provided. Error tracking disabled.")
        return
    
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration
        
        # Configure Sentry
        sentry_sdk.init(
            dsn=dsn,
            environment=environment,
            traces_sample_rate=1.0 if environment == "development" else 0.1,
            profiles_sample_rate=1.0 if environment == "development" else 0.1,
            enable_tracing=True,
            integrations=[
                FastApiIntegration(transaction_style="endpoint"),
                SqlalchemyIntegration(),
                LoggingIntegration(
                    level=logging.INFO,
                    event_level=logging.ERROR,
                ),
            ],
            # Release tracking
            release=f"adaptive-traffic-api@1.0.0",
            # Performance monitoring
            attach_stacktrace=True,
            send_default_pii=False,  # Don't send personally identifiable information
        )
        
        _sentry_initialized = True
        logger.info(f"Sentry error tracking initialized for environment: {environment}")
        
    except ImportError:
        logger.warning("sentry-sdk not installed. Error tracking disabled. Install with: pip install sentry-sdk[fastapi]")
    except Exception as e:
        logger.error(f"Failed to initialize Sentry: {e}", exc_info=True)


def capture_exception(error: Exception, context: Optional[dict] = None) -> None:
    """
    Manually capture an exception with additional context.
    
    Args:
        error: Exception to capture
        context: Additional context dictionary
    """
    if not _sentry_initialized:
        return
    
    try:
        import sentry_sdk
        
        with sentry_sdk.push_scope() as scope:
            if context:
                for key, value in context.items():
                    scope.set_context(key, value)
            sentry_sdk.capture_exception(error)
    except Exception as e:
        logger.error(f"Failed to capture exception in Sentry: {e}")


def capture_message(message: str, level: str = "info", context: Optional[dict] = None) -> None:
    """
    Capture a message/event in Sentry.
    
    Args:
        message: Message to capture
        level: Severity level (debug, info, warning, error, fatal)
        context: Additional context dictionary
    """
    if not _sentry_initialized:
        return
    
    try:
        import sentry_sdk
        
        with sentry_sdk.push_scope() as scope:
            if context:
                for key, value in context.items():
                    scope.set_context(key, value)
            sentry_sdk.capture_message(message, level=level)
    except Exception as e:
        logger.error(f"Failed to capture message in Sentry: {e}")

