"""
Unit tests for structured logging.
"""

import pytest
from unittest.mock import Mock, patch
from src.monitoring.logging import setup_structured_logging, get_logger, JSONFormatter
import logging


class TestLogging:
    """Test logging functionality."""
    
    def test_setup_structured_logging(self):
        """Test structured logging setup."""
        setup_structured_logging(level="INFO", json_output=True)
        # Should not raise exception
        assert True
    
    def test_get_logger(self):
        """Test getting logger instance."""
        logger = get_logger("test_module")
        assert logger is not None
        assert isinstance(logger, logging.Logger)
    
    def test_json_formatter(self):
        """Test JSON formatter."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        formatted = formatter.format(record)
        assert isinstance(formatted, str)
        assert "test" in formatted.lower()

