"""
Logging Configuration for Adaptive Traffic Vision System

Provides specialized logging setup for different vision components:
- Video input/output pipeline logging
- YOLO detection and tracking logging  
- Performance monitoring logs
- ROI management and configuration logs
- Real-time inference pipeline logs
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

from .config import LoggingConfig, LogLevel


class VisionLogger:
    """Specialized logger factory for vision system components."""
    
    _loggers: Dict[str, logging.Logger] = {}
    _initialized = False
    
    @classmethod
    def initialize(cls, config: LoggingConfig):
        """Initialize the logging system with configuration."""
        if cls._initialized:
            return
        
        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, config.level.value))
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        if config.console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, config.level.value))
            
            console_formatter = ColoredFormatter(
                config.log_format,
                use_colors=True
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if config.log_to_file and config.log_file_path:
            cls._setup_file_logging(config)
        
        # Module-specific loggers
        cls._setup_module_loggers(config)
        
        cls._initialized = True
        logging.info("Vision logging system initialized")
    
    @classmethod
    def _setup_file_logging(cls, config: LoggingConfig):
        """Setup file-based logging with rotation."""
        log_path = Path(config.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Main log file
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path,
            maxBytes=config.max_log_file_size_mb * 1024 * 1024,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, config.level.value))
        
        file_formatter = logging.Formatter(
            config.log_format + " [%(processName)s:%(threadName)s]"
        )
        file_handler.setFormatter(file_formatter)
        
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        
        # Separate performance log
        perf_log_path = log_path.parent / f"{log_path.stem}_performance.log"
        cls._setup_performance_logging(perf_log_path, config)
        
        # Separate error log
        error_log_path = log_path.parent / f"{log_path.stem}_errors.log"
        cls._setup_error_logging(error_log_path, config)
    
    @classmethod
    def _setup_performance_logging(cls, log_path: Path, config: LoggingConfig):
        """Setup specialized performance logging."""
        perf_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_path),
            maxBytes=config.max_log_file_size_mb * 1024 * 1024,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        perf_handler.setLevel(logging.INFO)
        
        perf_formatter = PerformanceFormatter()
        perf_handler.setFormatter(perf_formatter)
        
        # Add filter to only capture performance logs
        perf_handler.addFilter(PerformanceLogFilter())
        
        perf_logger = logging.getLogger('vision.performance')
        perf_logger.addHandler(perf_handler)
        perf_logger.setLevel(logging.INFO)
    
    @classmethod
    def _setup_error_logging(cls, log_path: Path, config: LoggingConfig):
        """Setup specialized error logging."""
        error_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_path),
            maxBytes=config.max_log_file_size_mb * 1024 * 1024,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        
        error_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s\n"
            "%(pathname)s:%(lineno)d in %(funcName)s\n"
            "%(exc_info)s\n" + "-" * 80
        )
        error_handler.setFormatter(error_formatter)
        
        root_logger = logging.getLogger()
        root_logger.addHandler(error_handler)
    
    @classmethod
    def _setup_module_loggers(cls, config: LoggingConfig):
        """Setup module-specific loggers."""
        module_configs = {
            'vision.video_pipeline': 'INFO',
            'vision.yolo_queue': 'INFO', 
            'vision.performance': 'INFO',
            'vision.roi_helper': 'INFO',
            'rl.video_env': 'INFO',
            'rl.inference_viz': 'INFO'
        }
        
        # Override with user configuration
        module_configs.update(config.module_specific_levels)
        
        for module_name, level_str in module_configs.items():
            logger = logging.getLogger(module_name)
            logger.setLevel(getattr(logging, level_str.upper()))
            cls._loggers[module_name] = logger
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger for a specific component."""
        if not cls._initialized:
            # Fallback initialization
            default_config = LoggingConfig()
            cls.initialize(default_config)
        
        if name in cls._loggers:
            return cls._loggers[name]
        
        # Create new logger
        logger = logging.getLogger(name)
        cls._loggers[name] = logger
        return logger


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def __init__(self, fmt: str, use_colors: bool = True):
        super().__init__(fmt)
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def format(self, record):
        if self.use_colors:
            # Add color to level name
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = (
                    f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
                )
        
        # Format the message
        formatted = super().format(record)
        
        # Reset levelname for other formatters
        record.levelname = levelname if self.use_colors else record.levelname
        
        return formatted


class PerformanceFormatter(logging.Formatter):
    """Specialized formatter for performance logs."""
    
    def format(self, record):
        if hasattr(record, 'performance_data'):
            # Format performance data as JSON
            perf_data = record.performance_data
            timestamp = datetime.fromtimestamp(record.created).isoformat()
            
            return json.dumps({
                'timestamp': timestamp,
                'level': record.levelname,
                'message': record.getMessage(),
                'performance': perf_data
            }, indent=None, separators=(',', ':'))
        
        return super().format(record)


class PerformanceLogFilter(logging.Filter):
    """Filter to capture only performance-related log records."""
    
    def filter(self, record):
        # Only allow records with performance data or from performance modules
        return (
            hasattr(record, 'performance_data') or 
            'performance' in record.name.lower() or
            'fps' in record.getMessage().lower() or
            'timing' in record.getMessage().lower()
        )


class VisionLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for vision components with context."""
    
    def __init__(self, logger: logging.Logger, component: str, context: Optional[Dict] = None):
        self.component = component
        self.context = context or {}
        super().__init__(logger, self.context)
    
    def process(self, msg, kwargs):
        # Add component context to log messages
        prefix = f"[{self.component}]"
        
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            prefix += f" ({context_str})"
        
        return f"{prefix} {msg}", kwargs
    
    def log_performance(self, level: int, msg: str, performance_data: Dict[str, Any]):
        """Log performance data with structured format."""
        if self.isEnabledFor(level):
            record = self.logger.makeRecord(
                self.logger.name, level, "(unknown file)", 0,
                msg, (), None, func="log_performance"
            )
            record.performance_data = performance_data
            self.logger.handle(record)
    
    def log_fps(self, fps: float, additional_data: Optional[Dict] = None):
        """Log FPS data."""
        perf_data = {"fps": fps}
        if additional_data:
            perf_data.update(additional_data)
        
        self.log_performance(
            logging.INFO,
            f"FPS: {fps:.2f}",
            perf_data
        )
    
    def log_timing(self, operation: str, duration_ms: float, additional_data: Optional[Dict] = None):
        """Log timing data."""
        perf_data = {
            "operation": operation,
            "duration_ms": duration_ms
        }
        if additional_data:
            perf_data.update(additional_data)
        
        self.log_performance(
            logging.DEBUG,
            f"Timing: {operation} took {duration_ms:.2f}ms",
            perf_data
        )


# Convenience functions for common logging tasks

def get_video_logger(component: str, context: Optional[Dict] = None) -> VisionLoggerAdapter:
    """Get a logger for video pipeline components."""
    base_logger = VisionLogger.get_logger('vision.video_pipeline')
    return VisionLoggerAdapter(base_logger, component, context)

def get_yolo_logger(component: str, context: Optional[Dict] = None) -> VisionLoggerAdapter:
    """Get a logger for YOLO detection components."""
    base_logger = VisionLogger.get_logger('vision.yolo_queue')
    return VisionLoggerAdapter(base_logger, component, context)

def get_performance_logger(component: str, context: Optional[Dict] = None) -> VisionLoggerAdapter:
    """Get a logger for performance monitoring."""
    base_logger = VisionLogger.get_logger('vision.performance')
    return VisionLoggerAdapter(base_logger, component, context)

def get_roi_logger(component: str, context: Optional[Dict] = None) -> VisionLoggerAdapter:
    """Get a logger for ROI management."""
    base_logger = VisionLogger.get_logger('vision.roi_helper')
    return VisionLoggerAdapter(base_logger, component, context)

def get_inference_logger(component: str, context: Optional[Dict] = None) -> VisionLoggerAdapter:
    """Get a logger for inference pipeline."""
    base_logger = VisionLogger.get_logger('rl.inference_viz')
    return VisionLoggerAdapter(base_logger, component, context)


def setup_logging_from_config(config: LoggingConfig):
    """Setup logging from configuration object."""
    VisionLogger.initialize(config)

def log_system_info():
    """Log system information at startup."""
    import platform
    import psutil
    
    logger = VisionLogger.get_logger('vision.system')
    
    logger.info("System Information:")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"CPU Count: {psutil.cpu_count()}")
    logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    try:
        import cv2
        logger.info(f"OpenCV: {cv2.__version__}")
    except ImportError:
        logger.warning("OpenCV not available")
    
    try:
        import ultralytics
        logger.info(f"Ultralytics: {ultralytics.__version__}")
    except ImportError:
        logger.warning("Ultralytics not available")
    
    try:
        import torch
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA Devices: {torch.cuda.device_count()}")
    except ImportError:
        logger.info("PyTorch not available")


def create_debug_log_snapshot(output_path: str):
    """Create a debug log snapshot for troubleshooting."""
    snapshot_path = Path(output_path)
    snapshot_path.mkdir(parents=True, exist_ok=True)
    
    # System info
    log_system_info()
    
    # Configuration snapshot
    logger = VisionLogger.get_logger('vision.debug')
    logger.info("Creating debug snapshot...")
    
    # TODO: Add more debug information gathering
    # - Current configuration state
    # - Recent performance metrics
    # - System resource utilization
    # - Active components status
    
    logger.info(f"Debug snapshot created at {snapshot_path}")