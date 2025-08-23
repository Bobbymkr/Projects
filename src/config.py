"""
Configuration Management for Adaptive Traffic Signal Control System

This module provides centralized configuration management with validation,
default values, and environment-specific settings.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
import yaml


class TrafficConfig(BaseModel):
    """Configuration for traffic simulation parameters."""
    num_lanes: int = Field(default=4, ge=1, le=16, description="Number of approach lanes")
    phase_lanes: List[List[int]] = Field(default=[[0, 1], [2, 3]], description="Lane groups for each phase")
    min_green: int = Field(default=5, ge=1, le=30, description="Minimum green time in seconds")
    max_green: int = Field(default=60, ge=10, le=120, description="Maximum green time in seconds")
    green_step: int = Field(default=5, ge=1, le=10, description="Green time increment in seconds")
    cycle_yellow: int = Field(default=3, ge=1, le=10, description="Yellow light duration")
    cycle_all_red: int = Field(default=1, ge=0, le=5, description="All-red clearance time")
    arrival_rates: List[float] = Field(default=[0.3, 0.25, 0.35, 0.2], description="Vehicle arrival rates per lane")
    queue_capacity: int = Field(default=40, ge=10, le=200, description="Maximum queue length per lane")
    episode_horizon: int = Field(default=3600, ge=300, le=7200, description="Episode duration in seconds")
    
    reward_weights: Dict[str, float] = Field(
        default={"queue": -1.0, "wait_penalty": -0.1, "efficiency": 0.01, "max_queue": -0.05},
        description="Reward function weights"
    )

    @validator('phase_lanes')
    def validate_phase_lanes(cls, v):
        if len(v) != 2:
            raise ValueError("Must have exactly 2 phases")
        all_lanes = set()
        for phase in v:
            all_lanes.update(phase)
        if len(all_lanes) != 4:  # Assuming 4 lanes
            raise ValueError("All lanes must be assigned to phases")
        return v

    @validator('arrival_rates')
    def validate_arrival_rates(cls, v, values):
        if 'num_lanes' in values and len(v) != values['num_lanes']:
            raise ValueError("arrival_rates length must match num_lanes")
        if any(rate < 0 for rate in v):
            raise ValueError("Arrival rates must be non-negative")
        return v


class RLConfig(BaseModel):
    """Configuration for reinforcement learning parameters."""
    algorithm: str = Field(default="DQN", description="RL algorithm to use")
    learning_rate: float = Field(default=1e-3, ge=1e-5, le=1e-1, description="Learning rate")
    gamma: float = Field(default=0.99, ge=0.8, le=0.999, description="Discount factor")
    epsilon_start: float = Field(default=1.0, ge=0.0, le=1.0, description="Initial exploration rate")
    epsilon_end: float = Field(default=0.05, ge=0.0, le=0.5, description="Final exploration rate")
    epsilon_decay: int = Field(default=20000, ge=1000, le=100000, description="Exploration decay steps")
    batch_size: int = Field(default=64, ge=16, le=256, description="Training batch size")
    buffer_size: int = Field(default=50000, ge=1000, le=1000000, description="Replay buffer size")
    target_update: int = Field(default=1000, ge=100, le=10000, description="Target network update frequency")
    warmup_steps: int = Field(default=1000, ge=100, le=10000, description="Warmup steps before training")
    
    # Network architecture
    hidden_size: int = Field(default=128, ge=32, le=512, description="Hidden layer size")
    num_layers: int = Field(default=2, ge=1, le=4, description="Number of hidden layers")
    
    # Training parameters
    max_episodes: int = Field(default=1000, ge=100, le=10000, description="Maximum training episodes")
    eval_frequency: int = Field(default=50, ge=10, le=200, description="Evaluation frequency")
    save_frequency: int = Field(default=100, ge=10, le=500, description="Model save frequency")


class VisionConfig(BaseModel):
    """Configuration for computer vision components."""
    model_path: Optional[str] = Field(default=None, description="Path to YOLO model")
    confidence_threshold: float = Field(default=0.4, ge=0.1, le=0.9, description="Detection confidence threshold")
    nms_threshold: float = Field(default=0.5, ge=0.1, le=0.9, description="NMS threshold")
    max_tracking_distance: float = Field(default=75.0, ge=10.0, le=200.0, description="Maximum tracking distance")
    max_missing_frames: int = Field(default=15, ge=1, le=50, description="Maximum missing frames for tracking")
    queue_smoothing_window: int = Field(default=3, ge=1, le=10, description="Queue smoothing window size")
    
    # Video processing
    target_fps: float = Field(default=15.0, ge=1.0, le=60.0, description="Target processing FPS")
    frame_width: int = Field(default=640, ge=320, le=1920, description="Frame width")
    frame_height: int = Field(default=480, ge=240, le=1080, description="Frame height")
    buffer_size: int = Field(default=30, ge=10, le=100, description="Frame buffer size")


class ForecastingConfig(BaseModel):
    """Configuration for traffic forecasting."""
    input_timesteps: int = Field(default=10, ge=5, le=50, description="Input sequence length")
    output_timesteps: int = Field(default=5, ge=1, le=20, description="Prediction horizon")
    lstm_units: int = Field(default=50, ge=16, le=200, description="LSTM units")
    cnn_filters: int = Field(default=64, ge=16, le=128, description="CNN filters")
    cnn_kernel: int = Field(default=3, ge=1, le=7, description="CNN kernel size")
    training_epochs: int = Field(default=50, ge=10, le=200, description="Training epochs")
    batch_size: int = Field(default=32, ge=8, le=128, description="Training batch size")


class LoggingConfig(BaseModel):
    """Configuration for logging and monitoring."""
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    tensorboard_dir: str = Field(default="logs/tensorboard", description="TensorBoard log directory")
    wandb_project: Optional[str] = Field(default=None, description="Weights & Biases project name")
    save_models: bool = Field(default=True, description="Whether to save trained models")
    save_plots: bool = Field(default=True, description="Whether to save training plots")


@dataclass
class Config:
    """Main configuration class that combines all sub-configurations."""
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    forecasting: ForecastingConfig = field(default_factory=ForecastingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Environment-specific settings
    use_sumo: bool = False
    use_marl: bool = False
    use_vision: bool = False
    use_forecasting: bool = False
    
    # Paths
    model_dir: str = "runs"
    config_dir: str = "configs"
    data_dir: str = "data"
    
    def __post_init__(self):
        """Create directories and validate configuration."""
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.logging.tensorboard_dir, exist_ok=True)

    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """Load configuration from file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.yaml':
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        return cls(
            traffic=TrafficConfig(**data.get("traffic", {})),
            rl=RLConfig(**data.get("rl", {})),
            vision=VisionConfig(**data.get("vision", {})),
            forecasting=ForecastingConfig(**data.get("forecasting", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            **{k: v for k, v in data.items() if k not in ["traffic", "rl", "vision", "forecasting", "logging"]}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "traffic": self.traffic.dict(),
            "rl": self.rl.dict(),
            "vision": self.vision.dict(),
            "forecasting": self.forecasting.dict(),
            "logging": self.logging.dict(),
            "use_sumo": self.use_sumo,
            "use_marl": self.use_marl,
            "use_vision": self.use_vision,
            "use_forecasting": self.use_forecasting,
            "model_dir": self.model_dir,
            "config_dir": self.config_dir,
            "data_dir": self.data_dir
        }
    
    def save(self, config_path: str):
        """Save configuration to file."""
        config_path = Path(config_path)
        os.makedirs(config_path.parent, exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() == '.yaml':
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
            else:
                json.dump(self.to_dict(), f, indent=2)
    
    def get_traffic_config(self) -> Dict[str, Any]:
        """Get traffic configuration as dictionary for environment."""
        return self.traffic.dict()


def load_config(config_name: str = "default") -> Config:
    """Load configuration by name."""
    config_paths = [
        f"configs/{config_name}.json",
        f"configs/{config_name}.yaml",
        f"configs/{config_name}.yml"
    ]
    
    for path in config_paths:
        if os.path.exists(path):
            return Config.from_file(path)
    
    # Return default configuration if file not found
    return Config()


def create_default_configs():
    """Create default configuration files."""
    configs = {
        "default": Config(),
        "sumo": Config(use_sumo=True),
        "marl": Config(use_marl=True),
        "vision": Config(use_vision=True),
        "full": Config(use_sumo=True, use_marl=True, use_vision=True, use_forecasting=True)
    }
    
    for name, config in configs.items():
        config.save(f"configs/{name}.json")
        config.save(f"configs/{name}.yaml")


if __name__ == "__main__":
    create_default_configs()
    print("Default configuration files created in configs/ directory.")
