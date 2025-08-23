"""
Logging Configuration for Adaptive Traffic Signal Control System

This module provides centralized logging configuration with multiple handlers,
formatters, and integration with TensorBoard and Weights & Biases.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from tensorboardX import SummaryWriter
    TENSORBOARDX_AVAILABLE = True
except ImportError:
    TENSORBOARDX_AVAILABLE = False


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


class MetricsLogger:
    """Logger for training metrics and performance data."""
    
    def __init__(self, log_dir: str, experiment_name: str = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_name = experiment_name
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.jsonl"
        self.tensorboard_writer = None
        self.wandb_run = None
        
        # Initialize TensorBoard
        if TENSORBOARDX_AVAILABLE:
            tensorboard_dir = self.log_dir / "tensorboard" / experiment_name
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(str(tensorboard_dir))
        
        # Initialize Weights & Biases
        if WANDB_AVAILABLE:
            try:
                self.wandb_run = wandb.init(
                    project="adaptive-traffic",
                    name=experiment_name,
                    config={},
                    dir=str(self.log_dir)
                )
            except Exception as e:
                print(f"Warning: Failed to initialize Weights & Biases: {e}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: int = None):
        """Log metrics to all configured outputs."""
        timestamp = datetime.utcnow().isoformat()
        
        # Log to JSONL file
        log_entry = {
            'timestamp': timestamp,
            'step': step,
            **metrics
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_writer.add_scalar(key, value, step or 0)
        
        # Log to Weights & Biases
        if self.wandb_run:
            wandb.log(metrics, step=step)
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Log hyperparameters."""
        if self.wandb_run:
            wandb.config.update(hyperparams)
        
        # Save to file
        hp_file = self.log_dir / f"{self.experiment_name}_hyperparams.json"
        with open(hp_file, 'w') as f:
            json.dump(hyperparams, f, indent=2)
    
    def log_model(self, model_path: str, metadata: Dict[str, Any] = None):
        """Log model artifacts."""
        if self.wandb_run:
            artifact = wandb.Artifact(
                name=f"model-{self.experiment_name}",
                type="model",
                description="Trained traffic signal control model"
            )
            artifact.add_file(model_path)
            if metadata:
                artifact.metadata.update(metadata)
            wandb.log_artifact(artifact)
    
    def close(self):
        """Close all loggers."""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        if self.wandb_run:
            wandb.finish()


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    experiment_name: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    enable_json: bool = False,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> MetricsLogger:
    """
    Set up comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        log_dir: Directory for log files
        experiment_name: Name for the current experiment
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_json: Enable JSON structured logging
        max_file_size: Maximum size for log files before rotation
        backup_count: Number of backup log files to keep
    
    Returns:
        MetricsLogger instance for logging metrics
    """
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    json_formatter = JSONFormatter()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if enable_file:
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"traffic_control_{timestamp}.log"
        else:
            log_file = log_dir / log_file
        
        # Use rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # JSON file handler
    if enable_json:
        json_log_file = log_dir / f"traffic_control_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        json_handler = logging.handlers.RotatingFileHandler(
            json_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        json_handler.setLevel(getattr(logging, log_level.upper()))
        json_handler.setFormatter(json_formatter)
        root_logger.addHandler(json_handler)
    
    # Create metrics logger
    metrics_logger = MetricsLogger(str(log_dir), experiment_name)
    
    # Log initial setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized with level: {log_level}")
    logger.info(f"Log directory: {log_dir}")
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    return metrics_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)


def log_with_context(logger: logging.Logger, message: str, extra_fields: Dict[str, Any] = None):
    """Log a message with additional context fields."""
    if extra_fields:
        record = logger.makeRecord(
            logger.name, logging.INFO, "", 0, message, (), None
        )
        record.extra_fields = extra_fields
        logger.handle(record)
    else:
        logger.info(message)


# Convenience functions for common logging patterns
def log_training_step(logger: logging.Logger, step: int, loss: float, reward: float, **kwargs):
    """Log training step metrics."""
    log_with_context(logger, f"Training step {step}", {
        'step': step,
        'loss': loss,
        'reward': reward,
        'type': 'training_step',
        **kwargs
    })


def log_evaluation(logger: logging.Logger, episode: int, avg_reward: float, **kwargs):
    """Log evaluation metrics."""
    log_with_context(logger, f"Evaluation episode {episode}", {
        'episode': episode,
        'avg_reward': avg_reward,
        'type': 'evaluation',
        **kwargs
    })


def log_environment_step(logger: logging.Logger, step: int, action: int, reward: float, queues: list, **kwargs):
    """Log environment step details."""
    log_with_context(logger, f"Environment step {step}", {
        'step': step,
        'action': action,
        'reward': reward,
        'queues': queues,
        'type': 'environment_step',
        **kwargs
    })


if __name__ == "__main__":
    # Example usage
    metrics_logger = setup_logging(
        log_level="DEBUG",
        log_dir="logs",
        experiment_name="test_experiment",
        enable_json=True
    )
    
    logger = get_logger(__name__)
    logger.info("Testing logging system")
    
    # Log some metrics
    metrics_logger.log_metrics({
        'loss': 0.5,
        'reward': -100.0,
        'queue_length': 15
    }, step=1)
    
    # Log hyperparameters
    metrics_logger.log_hyperparameters({
        'learning_rate': 0.001,
        'batch_size': 64,
        'gamma': 0.99
    })
    
    logger.info("Logging test completed")
