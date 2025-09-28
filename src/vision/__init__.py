"""Vision package for adaptive_traffic.

Exports:
- ROIConfig: Region of Interest configuration
- VehicleTracker: YOLOv8-based vehicle detection and queue estimation (dataclass)
- YOLOQueueEstimator: Main queue estimation class with YOLOv11n support
- VideoInputStream, VideoConfig: Video capture pipeline
- ROIManager: ROI configuration manager
- VideoSourceType: Enum for video source types
- YOLOModelManager: Enhanced model management with fallback and A/B testing
- YOLOConfig, ModelType: Configuration classes for model management
- DualModelValidator: A/B testing and validation framework
- AutomatedRollbackManager: Automated rollback with health monitoring
- PerformanceMonitor: Enhanced performance monitoring
- MigrationTestFramework: Comprehensive testing framework
- NISTComplianceManager: NIST cybersecurity framework compliance
"""

from .yolo_queue import ROIConfig, VehicleTracker, YOLOQueueEstimator, process_frame_for_queues, run_stream_queue_estimation
from .video_pipeline import VideoInputStream, VideoConfig, ROIManager, VideoSourceType
from .model_manager import YOLOModelManager, ModelMetrics, ModelState
from .config import YOLOConfig, ModelType, VisionSystemConfig, ConfigManager
from .validation import DualModelValidator, ValidationConfig, ValidationStrategy
from .rollback import AutomatedRollbackManager, RollbackConfig, RollbackTrigger
from .performance import PerformanceMonitor, PerformanceMetrics
from .testing import MigrationTestFramework, run_migration_validation_tests
from .compliance import NISTComplianceManager, SecurityLevel, NISTFunction

__all__ = [
    "ROIConfig",
    "VehicleTracker",
    "YOLOQueueEstimator",
    "process_frame_for_queues",
    "run_stream_queue_estimation",
    "VideoInputStream",
    "VideoConfig",
    "ROIManager",
    "VideoSourceType",
    "YOLOModelManager",
    "ModelMetrics",
    "ModelState",
    "YOLOConfig",
    "ModelType",
    "VisionSystemConfig",
    "ConfigManager",
    "DualModelValidator",
    "ValidationConfig",
    "ValidationStrategy",
    "AutomatedRollbackManager",
    "RollbackConfig",
    "RollbackTrigger",
    "PerformanceMonitor",
    "PerformanceMetrics",
    "MigrationTestFramework",
    "run_migration_validation_tests",
    "NISTComplianceManager",
    "SecurityLevel",
    "NISTFunction",
]