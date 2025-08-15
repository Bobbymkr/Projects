"""
Configuration Management for Adaptive Traffic Vision System

Provides comprehensive configuration classes and utilities for:
- Video input sources and processing parameters
- YOLO model configuration and performance tuning
- ROI management and visualization settings
- Logging and monitoring configuration
- Model download and management utilities
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import logging
from enum import Enum
import os

from .video_pipeline import VideoSourceType, VideoConfig, ROIManager
from .yolo_queue import ROIConfig

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported YOLO model types and sizes."""
    YOLOV8_NANO = "yolov8n.pt"
    YOLOV8_SMALL = "yolov8s.pt"
    YOLOV8_MEDIUM = "yolov8m.pt"
    YOLOV8_LARGE = "yolov8l.pt"
    YOLOV8_XLARGE = "yolov8x.pt"
    CUSTOM = "custom"


class LogLevel(Enum):
    """Logging levels for vision system."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class YOLOConfig:
    """Configuration for YOLO model and detection parameters."""
    model_type: ModelType = ModelType.YOLOV8_NANO
    model_path: Optional[str] = None
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_tracking_distance: float = 50.0
    max_missing_frames: int = 10
    queue_smoothing_window: int = 5
    vehicle_classes: List[int] = field(default_factory=lambda: [2, 3, 5, 7])  # car, motorcycle, bus, truck
    device: str = "auto"  # auto, cpu, cuda, mps
    half_precision: bool = False  # Use FP16 inference
    batch_size: int = 1
    
    def get_model_path(self) -> str:
        """Get the resolved model path."""
        if self.model_path and Path(self.model_path).exists():
            return self.model_path
        return self.model_type.value


@dataclass 
class PerformanceConfig:
    """Configuration for performance monitoring and optimization."""
    enable_fps_monitoring: bool = True
    fps_window_size: int = 30
    enable_processing_time_logs: bool = True
    log_performance_interval: int = 100  # frames
    memory_monitoring_enabled: bool = False
    gpu_monitoring_enabled: bool = False
    profiling_enabled: bool = False
    stats_save_interval: int = 300  # seconds
    stats_output_path: Optional[str] = None


@dataclass
class VisualizationConfig:
    """Configuration for visualization and overlay options."""
    show_bounding_boxes: bool = True
    show_track_ids: bool = True
    show_confidence_scores: bool = False
    show_roi_overlays: bool = True
    show_queue_lines: bool = True
    show_stop_lines: bool = True
    show_direction_vectors: bool = False
    show_performance_overlay: bool = True
    overlay_transparency: float = 0.3
    font_scale: float = 0.6
    font_thickness: int = 2
    bbox_thickness: int = 2
    
    # Color schemes (BGR format)
    roi_colors: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue  
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ])
    track_colors: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (255, 100, 100), (100, 255, 100), (100, 100, 255),
        (255, 255, 100), (255, 100, 255), (100, 255, 255)
    ])


@dataclass
class LoggingConfig:
    """Configuration for logging and debugging."""
    level: LogLevel = LogLevel.INFO
    log_to_file: bool = False
    log_file_path: Optional[str] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_log_file_size_mb: int = 10
    backup_count: int = 5
    console_logging: bool = True
    module_specific_levels: Dict[str, str] = field(default_factory=dict)


@dataclass
class VisionSystemConfig:
    """Complete configuration for the vision system."""
    video: VideoConfig
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # ROI configuration
    roi_config_path: Optional[str] = None
    auto_generate_rois: bool = True
    default_roi_count: int = 4
    
    # Model management
    models_directory: str = "./models"
    auto_download_models: bool = True
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Video source validation
        if self.video.source_type == VideoSourceType.WEBCAM:
            try:
                int(self.video.source)
            except ValueError:
                issues.append("Webcam source must be an integer")
        
        elif self.video.source_type == VideoSourceType.FILE:
            if not Path(self.video.source).exists():
                issues.append(f"Video file not found: {self.video.source}")
        
        # YOLO configuration validation
        if self.yolo.confidence_threshold < 0 or self.yolo.confidence_threshold > 1:
            issues.append("YOLO confidence threshold must be between 0 and 1")
        
        if self.yolo.nms_threshold < 0 or self.yolo.nms_threshold > 1:
            issues.append("YOLO NMS threshold must be between 0 and 1")
        
        # Performance configuration validation
        if self.performance.fps_window_size < 1:
            issues.append("FPS window size must be positive")
        
        # ROI validation
        if self.roi_config_path and not Path(self.roi_config_path).exists():
            issues.append(f"ROI config file not found: {self.roi_config_path}")
        
        return issues


class ConfigManager:
    """Manages loading, saving, and validation of vision system configurations."""
    
    @staticmethod
    def load_from_file(config_path: str) -> VisionSystemConfig:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return ConfigManager.from_dict(config_dict)
    
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> VisionSystemConfig:
        """Create configuration from dictionary."""
        
        # Parse video config
        video_dict = config_dict.get("video", {})
        video_config = VideoConfig(
            source=video_dict.get("source", "0"),
            source_type=VideoSourceType(video_dict.get("source_type", "webcam")),
            target_fps=video_dict.get("target_fps", 30.0),
            frame_width=video_dict.get("frame_width", 1280),
            frame_height=video_dict.get("frame_height", 720),
            buffer_size=video_dict.get("buffer_size", 30),
            auto_resize=video_dict.get("auto_resize", True),
            flip_horizontal=video_dict.get("flip_horizontal", False),
            flip_vertical=video_dict.get("flip_vertical", False),
            recording_enabled=video_dict.get("recording_enabled", False),
            recording_path=video_dict.get("recording_path"),
            recording_fps=video_dict.get("recording_fps", 30.0)
        )
        
        # Parse YOLO config
        yolo_dict = config_dict.get("yolo", {})
        yolo_config = YOLOConfig(
            model_type=ModelType(yolo_dict.get("model_type", "yolov8n.pt")),
            model_path=yolo_dict.get("model_path"),
            confidence_threshold=yolo_dict.get("confidence_threshold", 0.5),
            nms_threshold=yolo_dict.get("nms_threshold", 0.4),
            max_tracking_distance=yolo_dict.get("max_tracking_distance", 50.0),
            max_missing_frames=yolo_dict.get("max_missing_frames", 10),
            queue_smoothing_window=yolo_dict.get("queue_smoothing_window", 5),
            vehicle_classes=yolo_dict.get("vehicle_classes", [2, 3, 5, 7]),
            device=yolo_dict.get("device", "auto"),
            half_precision=yolo_dict.get("half_precision", False),
            batch_size=yolo_dict.get("batch_size", 1)
        )
        
        # Parse performance config
        perf_dict = config_dict.get("performance", {})
        performance_config = PerformanceConfig(
            enable_fps_monitoring=perf_dict.get("enable_fps_monitoring", True),
            fps_window_size=perf_dict.get("fps_window_size", 30),
            enable_processing_time_logs=perf_dict.get("enable_processing_time_logs", True),
            log_performance_interval=perf_dict.get("log_performance_interval", 100),
            memory_monitoring_enabled=perf_dict.get("memory_monitoring_enabled", False),
            gpu_monitoring_enabled=perf_dict.get("gpu_monitoring_enabled", False),
            profiling_enabled=perf_dict.get("profiling_enabled", False),
            stats_save_interval=perf_dict.get("stats_save_interval", 300),
            stats_output_path=perf_dict.get("stats_output_path")
        )
        
        # Parse visualization config
        viz_dict = config_dict.get("visualization", {})
        visualization_config = VisualizationConfig(
            show_bounding_boxes=viz_dict.get("show_bounding_boxes", True),
            show_track_ids=viz_dict.get("show_track_ids", True),
            show_confidence_scores=viz_dict.get("show_confidence_scores", False),
            show_roi_overlays=viz_dict.get("show_roi_overlays", True),
            show_queue_lines=viz_dict.get("show_queue_lines", True),
            show_stop_lines=viz_dict.get("show_stop_lines", True),
            show_direction_vectors=viz_dict.get("show_direction_vectors", False),
            show_performance_overlay=viz_dict.get("show_performance_overlay", True),
            overlay_transparency=viz_dict.get("overlay_transparency", 0.3),
            font_scale=viz_dict.get("font_scale", 0.6),
            font_thickness=viz_dict.get("font_thickness", 2),
            bbox_thickness=viz_dict.get("bbox_thickness", 2)
        )
        
        # Parse logging config
        log_dict = config_dict.get("logging", {})
        logging_config = LoggingConfig(
            level=LogLevel(log_dict.get("level", "INFO")),
            log_to_file=log_dict.get("log_to_file", False),
            log_file_path=log_dict.get("log_file_path"),
            log_format=log_dict.get("log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            max_log_file_size_mb=log_dict.get("max_log_file_size_mb", 10),
            backup_count=log_dict.get("backup_count", 5),
            console_logging=log_dict.get("console_logging", True),
            module_specific_levels=log_dict.get("module_specific_levels", {})
        )
        
        return VisionSystemConfig(
            video=video_config,
            yolo=yolo_config,
            performance=performance_config,
            visualization=visualization_config,
            logging=logging_config,
            roi_config_path=config_dict.get("roi_config_path"),
            auto_generate_rois=config_dict.get("auto_generate_rois", True),
            default_roi_count=config_dict.get("default_roi_count", 4),
            models_directory=config_dict.get("models_directory", "./models"),
            auto_download_models=config_dict.get("auto_download_models", True)
        )
    
    @staticmethod
    def save_to_file(config: VisionSystemConfig, config_path: str):
        """Save configuration to JSON file."""
        config_dict = ConfigManager.to_dict(config)
        
        # Ensure directory exists
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
    
    @staticmethod
    def to_dict(config: VisionSystemConfig) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "video": {
                "source": config.video.source,
                "source_type": config.video.source_type.value,
                "target_fps": config.video.target_fps,
                "frame_width": config.video.frame_width,
                "frame_height": config.video.frame_height,
                "buffer_size": config.video.buffer_size,
                "auto_resize": config.video.auto_resize,
                "flip_horizontal": config.video.flip_horizontal,
                "flip_vertical": config.video.flip_vertical,
                "recording_enabled": config.video.recording_enabled,
                "recording_path": config.video.recording_path,
                "recording_fps": config.video.recording_fps
            },
            "yolo": {
                "model_type": config.yolo.model_type.value,
                "model_path": config.yolo.model_path,
                "confidence_threshold": config.yolo.confidence_threshold,
                "nms_threshold": config.yolo.nms_threshold,
                "max_tracking_distance": config.yolo.max_tracking_distance,
                "max_missing_frames": config.yolo.max_missing_frames,
                "queue_smoothing_window": config.yolo.queue_smoothing_window,
                "vehicle_classes": config.yolo.vehicle_classes,
                "device": config.yolo.device,
                "half_precision": config.yolo.half_precision,
                "batch_size": config.yolo.batch_size
            },
            "performance": asdict(config.performance),
            "visualization": asdict(config.visualization),
            "logging": {
                "level": config.logging.level.value,
                "log_to_file": config.logging.log_to_file,
                "log_file_path": config.logging.log_file_path,
                "log_format": config.logging.log_format,
                "max_log_file_size_mb": config.logging.max_log_file_size_mb,
                "backup_count": config.logging.backup_count,
                "console_logging": config.logging.console_logging,
                "module_specific_levels": config.logging.module_specific_levels
            },
            "roi_config_path": config.roi_config_path,
            "auto_generate_rois": config.auto_generate_rois,
            "default_roi_count": config.default_roi_count,
            "models_directory": config.models_directory,
            "auto_download_models": config.auto_download_models
        }
    
    @staticmethod
    def create_default_config(video_source: str = "0") -> VisionSystemConfig:
        """Create a default configuration."""
        # Determine source type
        if video_source.isdigit():
            source_type = VideoSourceType.WEBCAM
        elif video_source.startswith(('http://', 'https://', 'rtsp://')):
            source_type = VideoSourceType.RTSP if video_source.startswith('rtsp://') else VideoSourceType.HTTP
        elif Path(video_source).exists():
            source_type = VideoSourceType.FILE
        else:
            source_type = VideoSourceType.WEBCAM
            video_source = "0"
        
        video_config = VideoConfig(
            source=video_source,
            source_type=source_type,
            target_fps=30.0,
            frame_width=1280,
            frame_height=720
        )
        
        return VisionSystemConfig(video=video_config)


def setup_logging(config: LoggingConfig):
    """Setup logging based on configuration."""
    
    # Set up logging level
    log_level = getattr(logging, config.level.value)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    if config.console_logging:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(config.log_format)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if config.log_to_file and config.log_file_path:
        from logging.handlers import RotatingFileHandler
        
        # Ensure log directory exists
        Path(config.log_file_path).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            config.log_file_path,
            maxBytes=config.max_log_file_size_mb * 1024 * 1024,
            backupCount=config.backup_count
        )
        file_handler.setLevel(log_level)
        formatter = logging.Formatter(config.log_format)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Module-specific levels
    for module_name, level_str in config.module_specific_levels.items():
        module_logger = logging.getLogger(module_name)
        module_level = getattr(logging, level_str.upper())
        module_logger.setLevel(module_level)