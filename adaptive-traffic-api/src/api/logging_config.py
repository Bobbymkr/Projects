"""
Structured Logging Configuration.

Enterprise-grade logging setup with JSON formatting, correlation IDs,
and integration with ELK stack for centralized log aggregation.
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from typing import Any, Dict
import traceback


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging compatible with ELK stack."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add request ID if available
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        
        # Add correlation ID if available
        if hasattr(record, "correlation_id"):
            log_data["correlation_id"] = record.correlation_id
        
        # Add extra fields
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }
        
        # Add any additional context from extra parameter
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)
        
        # Add process/thread information
        log_data["process"] = {
            "id": record.process,
            "name": record.processName,
            "thread_id": record.thread,
            "thread_name": record.threadName,
        }
        
        return json.dumps(log_data, default=str)


class ContextualFilter(logging.Filter):
    """Filter to add contextual information to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add contextual information to log record."""
        # Add environment information
        import os
        record.environment = os.getenv("ENVIRONMENT", "development")
        record.service_name = "adaptive-traffic-api"
        
        return True


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    log_file: str | None = None,
) -> None:
    """
    Setup structured logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON formatting (for ELK stack) or plain text
        log_file: Optional log file path
    """
    # Remove existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers = []
    
    # Set log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(log_level)
    
    # Create formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(ContextualFilter())
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(ContextualFilter())
        root_logger.addHandler(file_handler)
    
    # Set levels for third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(log_level)
    logging.getLogger("fastapi").setLevel(log_level)
    
    logging.info("Logging configured", extra={"level": level, "json_format": json_format})


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with contextual information.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

