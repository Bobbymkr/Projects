"""
Video Input Pipeline for Adaptive Traffic System

This module provides comprehensive video processing capabilities including:
- Multi-source video input (webcam, file, RTSP)
- Real-time frame extraction and preprocessing
- ROI configuration and management
- Frame buffering and threading for performance
- Video recording and playback utilities
"""

from typing import List, Dict, Any, Optional, Tuple, Callable, Union
import cv2
import numpy as np
import threading
import queue
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque
import json
from enum import Enum

from .yolo_queue import ROIConfig

logger = logging.getLogger(__name__)


class VideoSourceType(Enum):
    """Types of video input sources."""
    WEBCAM = "webcam"
    FILE = "file"
    RTSP = "rtsp"
    HTTP = "http"
    DIRECTORY = "directory"  # For processing image sequences


@dataclass
class VideoConfig:
    """Configuration for video input and processing."""
    source: str  # Camera index, file path, or stream URL
    source_type: VideoSourceType
    target_fps: float = 30.0
    frame_width: int = 1280
    frame_height: int = 720
    buffer_size: int = 30
    auto_resize: bool = True
    flip_horizontal: bool = False
    flip_vertical: bool = False
    roi_configs: List[ROIConfig] = field(default_factory=list)
    recording_enabled: bool = False
    recording_path: Optional[str] = None
    recording_fps: float = 30.0


@dataclass
class FrameData:
    """Container for processed frame data."""
    frame: np.ndarray
    timestamp: float
    frame_number: int
    original_size: Tuple[int, int]
    processed_size: Tuple[int, int]
    metadata: Dict[str, Any] = field(default_factory=dict)


class VideoInputStream:
    """
    High-performance video input stream with threading and buffering.
    
    Features:
    - Thread-safe frame capture and buffering
    - Automatic reconnection for streams
    - Frame rate control and dropping
    - Multiple input source support
    - Real-time processing optimization
    """

    def __init__(self, config: VideoConfig):
        """
        Initialize video input stream.
        
        Args:
            config: Video configuration parameters
        """
        self.config = config
        self.cap = None
        self.frame_buffer = queue.Queue(maxsize=config.buffer_size)
        self.running = False
        self.capture_thread = None
        self.frame_count = 0
        self.dropped_frames = 0
        self.last_fps_check = time.time()
        self.fps_counter = deque(maxlen=30)
        
        # Recording setup
        self.video_writer = None
        if config.recording_enabled and config.recording_path:
            self._setup_recording()
        
        # Performance monitoring
        self.stats = {
            "frames_captured": 0,
            "frames_dropped": 0,
            "fps": 0.0,
            "buffer_utilization": 0.0,
            "connection_status": "disconnected"
        }
        
        logger.info(f"VideoInputStream initialized for {config.source_type.value}: {config.source}")

    def _setup_recording(self):
        """Setup video recording if enabled."""
        try:
            recording_path = Path(self.config.recording_path)
            recording_path.parent.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                str(recording_path),
                fourcc,
                self.config.recording_fps,
                (self.config.frame_width, self.config.frame_height)
            )
            logger.info(f"Recording enabled: {recording_path}")
        except Exception as e:
            logger.error(f"Failed to setup recording: {e}")
            self.video_writer = None

    def start(self) -> bool:
        """
        Start video capture stream.
        
        Returns:
            bool: True if started successfully
        """
        try:
            # Initialize video capture
            if self.config.source_type == VideoSourceType.WEBCAM:
                self.cap = cv2.VideoCapture(int(self.config.source))
            elif self.config.source_type in [VideoSourceType.FILE, VideoSourceType.HTTP, VideoSourceType.RTSP]:
                self.cap = cv2.VideoCapture(self.config.source)
            else:
                raise ValueError(f"Unsupported source type: {self.config.source_type}")
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {self.config.source}")
            
            # Configure capture properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
            
            # Set buffer size for better real-time performance
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            self.stats["connection_status"] = "connected"
            logger.info(f"Video stream started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start video stream: {e}")
            self.stats["connection_status"] = "error"
            return False

    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        target_interval = 1.0 / self.config.target_fps
        last_capture_time = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Frame rate control
                if current_time - last_capture_time < target_interval:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
                    continue
                
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    if self.config.source_type in [VideoSourceType.RTSP, VideoSourceType.HTTP]:
                        self._attempt_reconnection()
                    continue
                
                # Apply transformations
                frame = self._preprocess_frame(frame)
                
                # Create frame data
                frame_data = FrameData(
                    frame=frame,
                    timestamp=current_time,
                    frame_number=self.frame_count,
                    original_size=(frame.shape[1], frame.shape[0]),
                    processed_size=(frame.shape[1], frame.shape[0])
                )
                
                # Add to buffer (drop oldest if full)
                try:
                    self.frame_buffer.put_nowait(frame_data)
                    self.stats["frames_captured"] += 1
                except queue.Full:
                    # Drop oldest frame
                    try:
                        self.frame_buffer.get_nowait()
                        self.frame_buffer.put_nowait(frame_data)
                        self.dropped_frames += 1
                        self.stats["frames_dropped"] += 1
                    except queue.Empty:
                        pass
                
                # Record frame if enabled
                if self.video_writer is not None:
                    self.video_writer.write(frame)
                
                # Update performance metrics
                self.frame_count += 1
                last_capture_time = current_time
                self._update_fps_stats(current_time)
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)  # Brief pause before retry

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply preprocessing transformations to frame."""
        # Resize if needed
        if self.config.auto_resize:
            target_size = (self.config.frame_width, self.config.frame_height)
            if frame.shape[:2][::-1] != target_size:
                frame = cv2.resize(frame, target_size)
        
        # Apply flips
        if self.config.flip_horizontal:
            frame = cv2.flip(frame, 1)
        if self.config.flip_vertical:
            frame = cv2.flip(frame, 0)
        
        return frame

    def _attempt_reconnection(self):
        """Attempt to reconnect to stream source."""
        logger.info("Attempting to reconnect to stream...")
        try:
            if self.cap:
                self.cap.release()
            time.sleep(2)  # Wait before reconnection
            self.cap = cv2.VideoCapture(self.config.source)
            if self.cap.isOpened():
                logger.info("Reconnection successful")
                self.stats["connection_status"] = "connected"
            else:
                self.stats["connection_status"] = "reconnecting"
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            self.stats["connection_status"] = "error"

    def _update_fps_stats(self, current_time: float):
        """Update FPS performance statistics."""
        if current_time - self.last_fps_check >= 1.0:
            fps = len(self.fps_counter)
            self.stats["fps"] = fps
            self.stats["buffer_utilization"] = self.frame_buffer.qsize() / self.config.buffer_size
            self.last_fps_check = current_time
            self.fps_counter.clear()
        
        self.fps_counter.append(current_time)

    def get_frame(self, timeout: float = 0.1) -> Optional[FrameData]:
        """
        Get next available frame from buffer.
        
        Args:
            timeout: Maximum wait time for frame
            
        Returns:
            FrameData or None if no frame available
        """
        try:
            return self.frame_buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_latest_frame(self) -> Optional[FrameData]:
        """
        Get most recent frame, discarding older buffered frames.
        
        Returns:
            Most recent FrameData or None
        """
        latest_frame = None
        try:
            # Drain buffer to get latest frame
            while True:
                latest_frame = self.frame_buffer.get_nowait()
        except queue.Empty:
            pass
        return latest_frame

    def stop(self):
        """Stop video capture and cleanup resources."""
        self.running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        if self.video_writer:
            self.video_writer.release()
        
        # Clear buffer
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except queue.Empty:
                break
        
        self.stats["connection_status"] = "disconnected"
        logger.info("Video stream stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.stats.copy()


class ROIManager:
    """
    Manages Region of Interest (ROI) configurations for traffic lanes.
    
    Features:
    - Interactive ROI definition and editing
    - JSON serialization for configuration persistence
    - Validation and coordinate transformation
    - Visual overlay generation for debugging
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ROI manager.
        
        Args:
            config_path: Path to ROI configuration file
        """
        self.config_path = config_path
        self.rois: List[ROIConfig] = []
        self.frame_size: Optional[Tuple[int, int]] = None
        
        if config_path and Path(config_path).exists():
            self.load_config(config_path)

    def add_roi(self, roi: ROIConfig):
        """Add new ROI configuration."""
        # Validate ROI
        if len(roi.polygon) < 3:
            raise ValueError("ROI polygon must have at least 3 points")
        
        self.rois.append(roi)
        logger.info(f"Added ROI {roi.lane_id}: {roi.name}")

    def remove_roi(self, lane_id: int) -> bool:
        """
        Remove ROI by lane ID.
        
        Returns:
            bool: True if removed successfully
        """
        for i, roi in enumerate(self.rois):
            if roi.lane_id == lane_id:
                del self.rois[i]
                logger.info(f"Removed ROI {lane_id}")
                return True
        return False

    def get_roi(self, lane_id: int) -> Optional[ROIConfig]:
        """Get ROI configuration by lane ID."""
        for roi in self.rois:
            if roi.lane_id == lane_id:
                return roi
        return None

    def create_default_rois(self, frame_width: int, frame_height: int, num_lanes: int = 4) -> List[ROIConfig]:
        """
        Create default ROI configurations for a standard intersection.
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            num_lanes: Number of lanes to create
            
        Returns:
            List of ROIConfig objects
        """
        self.frame_size = (frame_width, frame_height)
        rois = []
        
        # Create ROIs for 4-way intersection
        if num_lanes == 4:
            # Top lane (North to South)
            rois.append(ROIConfig(
                lane_id=0,
                polygon=[(frame_width//2 - 100, 0), (frame_width//2, 0), 
                        (frame_width//2, frame_height//2 - 50), (frame_width//2 - 100, frame_height//2 - 50)],
                direction_vector=(0, 1),
                queue_line=((frame_width//2 - 50, frame_height//4), (frame_width//2 - 50, frame_height//2 - 50)),
                stop_line=((frame_width//2 - 100, frame_height//2 - 50), (frame_width//2, frame_height//2 - 50)),
                name="North Approach"
            ))
            
            # Right lane (East to West)
            rois.append(ROIConfig(
                lane_id=1,
                polygon=[(frame_width//2 + 50, frame_height//2 - 100), (frame_width, frame_height//2 - 100),
                        (frame_width, frame_height//2), (frame_width//2 + 50, frame_height//2)],
                direction_vector=(-1, 0),
                queue_line=((3*frame_width//4, frame_height//2 - 50), (frame_width//2 + 50, frame_height//2 - 50)),
                stop_line=((frame_width//2 + 50, frame_height//2 - 100), (frame_width//2 + 50, frame_height//2)),
                name="East Approach"
            ))
            
            # Bottom lane (South to North)
            rois.append(ROIConfig(
                lane_id=2,
                polygon=[(frame_width//2, frame_height//2 + 50), (frame_width//2 + 100, frame_height//2 + 50),
                        (frame_width//2 + 100, frame_height), (frame_width//2, frame_height)],
                direction_vector=(0, -1),
                queue_line=((frame_width//2 + 50, 3*frame_height//4), (frame_width//2 + 50, frame_height//2 + 50)),
                stop_line=((frame_width//2, frame_height//2 + 50), (frame_width//2 + 100, frame_height//2 + 50)),
                name="South Approach"
            ))
            
            # Left lane (West to East)
            rois.append(ROIConfig(
                lane_id=3,
                polygon=[(0, frame_height//2), (frame_width//2 - 50, frame_height//2),
                        (frame_width//2 - 50, frame_height//2 + 100), (0, frame_height//2 + 100)],
                direction_vector=(1, 0),
                queue_line=((frame_width//4, frame_height//2 + 50), (frame_width//2 - 50, frame_height//2 + 50)),
                stop_line=((frame_width//2 - 50, frame_height//2), (frame_width//2 - 50, frame_height//2 + 100)),
                name="West Approach"
            ))
        
        self.rois = rois
        logger.info(f"Created {len(rois)} default ROIs")
        return rois

    def save_config(self, path: Optional[str] = None):
        """Save ROI configuration to JSON file."""
        config_path = path or self.config_path
        if not config_path:
            raise ValueError("No configuration path specified")
        
        config_data = {
            "frame_size": self.frame_size,
            "rois": []
        }
        
        for roi in self.rois:
            roi_data = {
                "lane_id": roi.lane_id,
                "polygon": roi.polygon,
                "direction_vector": roi.direction_vector,
                "queue_line": roi.queue_line,
                "stop_line": roi.stop_line,
                "name": roi.name
            }
            config_data["rois"].append(roi_data)
        
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"ROI configuration saved to {config_path}")

    def load_config(self, path: str):
        """Load ROI configuration from JSON file."""
        with open(path, 'r') as f:
            config_data = json.load(f)
        
        self.frame_size = tuple(config_data.get("frame_size", (1280, 720)))
        self.rois = []
        
        for roi_data in config_data.get("rois", []):
            roi = ROIConfig(
                lane_id=roi_data["lane_id"],
                polygon=[tuple(pt) for pt in roi_data["polygon"]],
                direction_vector=tuple(roi_data["direction_vector"]),
                queue_line=(tuple(roi_data["queue_line"][0]), tuple(roi_data["queue_line"][1])),
                stop_line=(tuple(roi_data["stop_line"][0]), tuple(roi_data["stop_line"][1])),
                name=roi_data.get("name", f"Lane {roi_data['lane_id']}")
            )
            self.rois.append(roi)
        
        logger.info(f"Loaded {len(self.rois)} ROIs from {path}")

    def draw_rois_overlay(self, frame: np.ndarray, show_labels: bool = True) -> np.ndarray:
        """
        Draw ROI overlays on frame for visualization.
        
        Args:
            frame: Input frame
            show_labels: Whether to show lane labels
            
        Returns:
            Frame with ROI overlays
        """
        overlay = frame.copy()
        
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # Green, Blue, Red, Cyan
        
        for i, roi in enumerate(self.rois):
            color = colors[i % len(colors)]
            
            # Draw ROI polygon
            pts = np.array(roi.polygon, np.int32)
            cv2.polylines(overlay, [pts], True, color, 2)
            cv2.fillPoly(overlay, [pts], (*color, 50))  # Semi-transparent fill
            
            # Draw queue line
            cv2.line(overlay, roi.queue_line[0], roi.queue_line[1], color, 3)
            
            # Draw stop line
            cv2.line(overlay, roi.stop_line[0], roi.stop_line[1], (0, 255, 255), 3)
            
            # Draw labels
            if show_labels and roi.polygon:
                label_pos = roi.polygon[0]
                cv2.putText(overlay, roi.name, label_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Blend with original frame
        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        return result


def create_video_config_from_dict(config_dict: Dict[str, Any]) -> VideoConfig:
    """
    Create VideoConfig from dictionary (for JSON loading).
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        VideoConfig object
    """
    # Convert source type
    source_type = VideoSourceType(config_dict.get("source_type", "webcam"))
    
    # Convert ROI configurations
    roi_configs = []
    for roi_data in config_dict.get("roi_configs", []):
        roi = ROIConfig(
            lane_id=roi_data["lane_id"],
            polygon=[tuple(pt) for pt in roi_data["polygon"]],
            direction_vector=tuple(roi_data["direction_vector"]),
            queue_line=(tuple(roi_data["queue_line"][0]), tuple(roi_data["queue_line"][1])),
            stop_line=(tuple(roi_data["stop_line"][0]), tuple(roi_data["stop_line"][1])),
            name=roi_data.get("name", f"Lane {roi_data['lane_id']}")
        )
        roi_configs.append(roi)
    
    return VideoConfig(
        source=config_dict["source"],
        source_type=source_type,
        target_fps=config_dict.get("target_fps", 30.0),
        frame_width=config_dict.get("frame_width", 1280),
        frame_height=config_dict.get("frame_height", 720),
        buffer_size=config_dict.get("buffer_size", 30),
        auto_resize=config_dict.get("auto_resize", True),
        flip_horizontal=config_dict.get("flip_horizontal", False),
        flip_vertical=config_dict.get("flip_vertical", False),
        roi_configs=roi_configs,
        recording_enabled=config_dict.get("recording_enabled", False),
        recording_path=config_dict.get("recording_path"),
        recording_fps=config_dict.get("recording_fps", 30.0)
    )