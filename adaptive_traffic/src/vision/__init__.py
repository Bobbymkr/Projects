"""Vision package for adaptive_traffic.

Exports:
- ROIConfig: Region of Interest configuration
- VehicleTracker: YOLOv8-based vehicle detection and queue estimation (dataclass)
- YOLOQueueEstimator: Main queue estimation class
- VideoInputStream, VideoConfig: Video capture pipeline
- ROIManager: ROI configuration manager
- VideoSourceType: Enum for video source types
"""

from .yolo_queue import ROIConfig, VehicleTracker, YOLOQueueEstimator, process_frame_for_queues, run_stream_queue_estimation
from .video_pipeline import VideoInputStream, VideoConfig, ROIManager, VideoSourceType

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
]