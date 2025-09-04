#!/usr/bin/env python
"""
Script to implement professional logging and error handling
for the Adaptive Traffic project.
"""

import os
import sys
import json
import logging
import logging.config
from pathlib import Path

# Define colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(message):
    print(f"{Colors.HEADER}{Colors.BOLD}\n{message}{Colors.ENDC}")

def print_success(message):
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")

def print_warning(message):
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")

def print_error(message):
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")

def print_info(message):
    print(f"{Colors.BLUE}ℹ {message}{Colors.ENDC}")

def get_project_root():
    """Get the project root directory."""
    # Assuming this script is in the scripts directory
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    return script_dir.parent

def create_logging_utility():
    """Create a logging utility in src/utils."""
    print_header("Creating Logging Utility")
    
    project_root = get_project_root()
    utils_dir = os.path.join(project_root, "src", "utils")
    logging_utility_path = os.path.join(utils_dir, "logging.py")
    
    logging_utility_content = """
"""Logging utilities for the Adaptive Traffic Control System."""

import os
import json
import logging
import logging.config
from pathlib import Path
from typing import Dict, Optional, Union


class LoggingError(Exception):
    """Exception raised for logging errors."""
    pass


def setup_logging(
    config_file: Optional[str] = None,
    default_level: int = logging.INFO,
    env_key: str = "LOG_CFG"
) -> logging.Logger:
    """Setup logging configuration.

    Args:
        config_file: Path to the logging configuration file.
        default_level: Default logging level.
        env_key: Environment variable that can be used to override the config file.

    Returns:
        Logger object.

    Raises:
        LoggingError: If the configuration file cannot be loaded.
    """
    # Check if logging config file is specified in environment variable
    path = os.getenv(env_key, None)
    if path is None and config_file is not None:
        path = config_file

    if path is not None and os.path.exists(path):
        try:
            with open(path, "r") as f:
                config = json.load(f)

            # Create log directory if it doesn't exist
            for handler in config.get("handlers", {}).values():
                if "filename" in handler:
                    log_dir = os.path.dirname(handler["filename"])
                    if log_dir:
                        os.makedirs(log_dir, exist_ok=True)

            logging.config.dictConfig(config)
        except Exception as e:
            raise LoggingError(f"Failed to load logging configuration from {path}: {str(e)}")
    else:
        # Basic configuration
        logging.basicConfig(
            level=default_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    return logging.getLogger("adaptive_traffic")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Logger name.

    Returns:
        Logger object.
    """
    return logging.getLogger(f"adaptive_traffic.{name}")


def log_exception(logger: logging.Logger, exception: Exception, message: str = "An error occurred") -> None:
    """Log an exception.

    Args:
        logger: Logger object.
        exception: Exception object.
        message: Error message.
    """
    logger.error(f"{message}: {str(exception)}", exc_info=True)


def log_to_file(message: str, file_path: str, mode: str = "a") -> None:
    """Log a message to a file.

    Args:
        message: Message to log.
        file_path: Path to the log file.
        mode: File open mode.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode) as f:
        f.write(f"{message}\n")
"""
    
    # Create logging utility file
    if not os.path.exists(logging_utility_path):
        os.makedirs(os.path.dirname(logging_utility_path), exist_ok=True)
        with open(logging_utility_path, "w") as f:
            f.write(logging_utility_content.strip())
        print_success("Created logging utility")
    else:
        print_info("Logging utility already exists")
    
    return True

def create_error_handling_utility():
    """Create an error handling utility in src/utils."""
    print_header("Creating Error Handling Utility")
    
    project_root = get_project_root()
    utils_dir = os.path.join(project_root, "src", "utils")
    error_handling_path = os.path.join(utils_dir, "error_handling.py")
    
    error_handling_content = """
"""Error handling utilities for the Adaptive Traffic Control System."""

import sys
import traceback
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast

from .logging import get_logger, log_exception

# Type variables for generic function signatures
F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')


class AdaptiveTrafficError(Exception):
    """Base exception for all Adaptive Traffic Control System errors."""
    pass


class ConfigurationError(AdaptiveTrafficError):
    """Exception raised for configuration errors."""
    pass


class EnvironmentError(AdaptiveTrafficError):
    """Exception raised for environment errors."""
    pass


class ModelError(AdaptiveTrafficError):
    """Exception raised for model errors."""
    pass


class DataError(AdaptiveTrafficError):
    """Exception raised for data errors."""
    pass


class APIError(AdaptiveTrafficError):
    """Exception raised for API errors."""
    pass


def handle_exceptions(
    logger_name: str,
    reraise: bool = True,
    fallback_return: Optional[Any] = None,
    handled_exceptions: Optional[List[Type[Exception]]] = None
) -> Callable[[F], F]:
    """Decorator to handle exceptions.

    Args:
        logger_name: Name of the logger to use.
        reraise: Whether to reraise the exception after logging.
        fallback_return: Value to return if an exception is caught and not reraised.
        handled_exceptions: List of exception types to handle. If None, all exceptions are handled.

    Returns:
        Decorated function.
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger(logger_name)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if handled_exceptions is None or any(isinstance(e, exc) for exc in handled_exceptions):
                    log_exception(logger, e, f"Error in {func.__name__}")
                    if reraise:
                        raise
                    return fallback_return
                else:
                    raise
        return cast(F, wrapper)
    return decorator


def safe_execute(func: Callable[..., T], *args: Any, **kwargs: Any) -> Union[T, None]:
    """Execute a function safely, catching and logging any exceptions.

    Args:
        func: Function to execute.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        Result of the function call, or None if an exception was raised.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger = get_logger("safe_execute")
        log_exception(logger, e, f"Error executing {func.__name__}")
        return None


def format_exception(e: Exception) -> str:
    """Format an exception as a string.

    Args:
        e: Exception to format.

    Returns:
        Formatted exception string.
    """
    return f"{type(e).__name__}: {str(e)}\n{''.join(traceback.format_tb(e.__traceback__))}"


def get_exception_context(e: Exception) -> Dict[str, Any]:
    """Get context information for an exception.

    Args:
        e: Exception to get context for.

    Returns:
        Dictionary with exception context information.
    """
    return {
        "type": type(e).__name__,
        "message": str(e),
        "traceback": traceback.format_tb(e.__traceback__),
        "module": e.__class__.__module__,
    }
"""
    
    # Create error handling utility file
    if not os.path.exists(error_handling_path):
        os.makedirs(os.path.dirname(error_handling_path), exist_ok=True)
        with open(error_handling_path, "w") as f:
            f.write(error_handling_content.strip())
        print_success("Created error handling utility")
    else:
        print_info("Error handling utility already exists")
    
    return True

def create_logging_config():
    """Create a logging configuration file."""
    print_header("Creating Logging Configuration")
    
    project_root = get_project_root()
    config_dir = os.path.join(project_root, "configs")
    logging_config_path = os.path.join(config_dir, "logging.json")
    
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]"
            },
            "json": {
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(filename)s %(lineno)d %(funcName)s %(module)s",
                "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": "logs/adaptive_traffic.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": "logs/error.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            }
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console", "file"],
                "level": "INFO"
            },
            "adaptive_traffic": {
                "handlers": ["console", "file", "error_file"],
                "level": "DEBUG",
                "propagate": False
            },
            "adaptive_traffic.env": {
                "level": "DEBUG",
                "propagate": True
            },
            "adaptive_traffic.rl": {
                "level": "DEBUG",
                "propagate": True
            },
            "adaptive_traffic.vision": {
                "level": "DEBUG",
                "propagate": True
            },
            "adaptive_traffic.forecast": {
                "level": "DEBUG",
                "propagate": True
            },
            "adaptive_traffic.api": {
                "level": "INFO",
                "propagate": True
            },
            "adaptive_traffic.web": {
                "level": "INFO",
                "propagate": True
            }
        }
    }
    
    # Create logging configuration file
    if not os.path.exists(logging_config_path):
        os.makedirs(os.path.dirname(logging_config_path), exist_ok=True)
        with open(logging_config_path, "w") as f:
            json.dump(logging_config, f, indent=2)
        print_success("Created logging configuration file")
    else:
        print_info("Logging configuration file already exists")
    
    return True

def create_example_usage():
    """Create an example usage file."""
    print_header("Creating Example Usage")
    
    project_root = get_project_root()
    examples_dir = os.path.join(project_root, "examples")
    example_path = os.path.join(examples_dir, "logging_example.py")
    
    example_content = """
"""Example usage of logging and error handling utilities."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import setup_logging, get_logger
from src.utils.error_handling import handle_exceptions, safe_execute, AdaptiveTrafficError


@handle_exceptions(logger_name="example", reraise=False)
def example_function(x, y):
    """Example function that might raise an exception."""
    return x / y


def main():
    """Main function."""
    # Setup logging
    config_file = os.path.join(project_root, "configs", "logging.json")
    logger = setup_logging(config_file)
    
    # Log some messages
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Get a module-specific logger
    module_logger = get_logger("example")
    module_logger.info("This is a module-specific message")
    
    # Example of exception handling
    try:
        result = example_function(10, 0)  # This will raise a ZeroDivisionError
        print(f"Result: {result}")  # This won't be executed
    except Exception as e:
        module_logger.error(f"Caught exception: {e}")
    
    # Example of safe execution
    result = safe_execute(example_function, 10, 2)  # This will succeed
    print(f"Safe execution result: {result}")
    
    result = safe_execute(example_function, 10, 0)  # This will fail but not raise
    print(f"Safe execution with error result: {result}")
    
    # Example of custom exception
    try:
        raise AdaptiveTrafficError("This is a custom exception")
    except AdaptiveTrafficError as e:
        module_logger.error(f"Caught custom exception: {e}")


if __name__ == "__main__":
    main()
"""
    
    # Create example usage file
    if not os.path.exists(example_path):
        os.makedirs(os.path.dirname(example_path), exist_ok=True)
        with open(example_path, "w") as f:
            f.write(example_content.strip())
        print_success("Created example usage file")
    else:
        print_info("Example usage file already exists")
    
    return True

def main():
    """Main function to set up logging and error handling."""
    print_header("Adaptive Traffic Project - Logging and Error Handling Setup")
    
    # Create logging utility
    create_logging_utility()
    
    # Create error handling utility
    create_error_handling_utility()
    
    # Create logging configuration
    create_logging_config()
    
    # Create example usage
    create_example_usage()
    
    print_header("Logging and Error Handling Setup Complete")
    print_info("Professional logging and error handling have been set up successfully")

if __name__ == "__main__":
    main()