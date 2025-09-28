from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import cv2
import os
import logging
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import threading
from pathlib import Path

# Import enhanced model management
from .model_manager import YOLOModelManager, ModelMetrics
from .config import YOLOConfig, ModelType


# Configure logging for vision module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VehicleTracker:
    """
    Tracks individual vehicles across frames for accurate queue estimation.
    Uses centroid tracking with Kalman-like motion prediction.
    """
    track_id: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    velocity: Tuple[float, float] = (0.0, 0.0)
    last_seen: float = field(default_factory=time.time)
    frames_missing: int = 0
    is_stationary: bool = False
    stationary_frames: int = 0
    
    def update_position(self, new_centroid: Tuple[float, float], new_bbox: Tuple[int, int, int, int], confidence: float):
        """Update tracker position with motion estimation."""
        # Calculate velocity
        dt = time.time() - self.last_seen
        if dt > 0:
            self.velocity = (
                (new_centroid[0] - self.centroid[0]) / dt,
                (new_centroid[1] - self.centroid[1]) / dt
            )
        
        # Update position and stats
        self.centroid = new_centroid
        self.bbox = new_bbox
        self.confidence = confidence
        self.last_seen = time.time()
        self.frames_missing = 0
        
        # Check if stationary (queue vehicle indicator)
        speed = np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        if speed < 2.0:  # pixels per second threshold
            self.stationary_frames += 1
            if self.stationary_frames > 5:
                self.is_stationary = True
        else:
            self.stationary_frames = 0
            self.is_stationary = False


@dataclass
class ROIConfig:
    """Configuration for Region of Interest (lane) monitoring."""
    lane_id: int
    polygon: List[Tuple[int, int]]  # ROI polygon vertices
    direction_vector: Tuple[float, float]  # Expected traffic direction
    queue_line: Tuple[Tuple[int, int], Tuple[int, int]]  # Start and end of queue measurement line
    stop_line: Tuple[Tuple[int, int], Tuple[int, int]]  # Stop line for traffic light
    name: str = ""


class YOLOQueueEstimator:
    """
    Production YOLOv8-based vehicle detection and queue estimation system.
    
    Features:
    - Real-time vehicle detection using YOLOv8
    - Multi-object tracking with centroid tracking
    - ROI-based lane monitoring
    - Queue length estimation based on stationary vehicles
    - Motion analysis for traffic flow measurement
    - Confidence-based filtering and temporal smoothing
    """

    def __init__(self, 
                 config: Optional[YOLOConfig] = None,
                 model_path: Optional[str] = None, 
                 rois: Optional[List[ROIConfig]] = None,
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4,
                 max_tracking_distance: float = 50.0,
                 max_missing_frames: int = 10,
                 queue_smoothing_window: int = 5):
        """
        Initialize enhanced YOLO-based queue estimator with YOLOv11n support.
        
        Args:
            config: YOLOConfig object with enhanced model management settings
            model_path: Legacy parameter for backward compatibility
            rois: List of ROI configurations for each lane
            confidence_threshold: Minimum detection confidence
            nms_threshold: Non-maximum suppression threshold
            max_tracking_distance: Maximum distance for track association
            max_missing_frames: Maximum frames before dropping a track
            queue_smoothing_window: Frames for temporal smoothing
        """
        
        # Handle configuration - prioritize config object over individual parameters
        if config is not None:
            self.config = config
        else:
            # Create config from legacy parameters for backward compatibility
            self.config = YOLOConfig(
                model_type=ModelType.YOLOV11_NANO,  # Default to YOLOv11n
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                max_tracking_distance=max_tracking_distance,
                max_missing_frames=max_missing_frames,
                queue_smoothing_window=queue_smoothing_window
            )
        
        # Initialize model manager
        self.model_manager = YOLOModelManager(self.config)
        
        # Initialize models
        if not self.model_manager.initialize_models():
            logger.error("Failed to initialize any YOLO models")
            self.model_manager = None
        
        # Legacy compatibility properties
        self.model_path = self.config.model_path
        self.confidence_threshold = self.config.confidence_threshold
        self.nms_threshold = self.config.nms_threshold
        self.max_tracking_distance = self.config.max_tracking_distance
        self.max_missing_frames = self.config.max_missing_frames
        
        self.rois = rois or []
        
        # Vehicle classes from COCO dataset
        self.vehicle_classes = set(self.config.vehicle_classes)
        
        # Tracking system
        self.trackers: Dict[int, VehicleTracker] = {}
        self.next_track_id = 0
        self.tracking_lock = threading.Lock()
        
        # Queue estimation with temporal smoothing
        self.queue_history = defaultdict(lambda: deque(maxlen=self.config.queue_smoothing_window))
        self.last_detection_time = 0
        
        # Performance monitoring (enhanced)
        self.fps_counter = deque(maxlen=30)
        self.detection_stats = {"total_detections": 0, "vehicles_detected": 0}
        self.shadow_detection_stats = {"total_detections": 0, "vehicles_detected": 0}
        
        # A/B testing state
        self.shadow_mode_active = self.config.enable_shadow_mode
        self.shadow_comparison_results = []
        
        logger.info(f"Enhanced YOLOQueueEstimator initialized with {len(self.rois)} ROIs")
        logger.info(f"Model configuration: {self.config.model_type.value}")
        logger.info(f"Shadow mode: {self.shadow_mode_active}")
        logger.info(f"Fallback chain: {[mt.value for mt in self.config.fallback_chain]}")

    def _compare_shadow_results(self, primary_detections: List[Dict], shadow_detections: List[Dict], 
                              primary_metrics: ModelMetrics, shadow_metrics: ModelMetrics):
        """Compare primary and shadow model results for A/B testing."""
        comparison = {
            "timestamp": time.time(),
            "primary_count": len(primary_detections),
            "shadow_count": len(shadow_detections),
            "primary_fps": primary_metrics.fps if primary_metrics else 0.0,
            "shadow_fps": shadow_metrics.fps if shadow_metrics else 0.0,
            "primary_inference_ms": primary_metrics.inference_time_ms if primary_metrics else 0.0,
            "shadow_inference_ms": shadow_metrics.inference_time_ms if shadow_metrics else 0.0,
            "detection_difference": abs(len(primary_detections) - len(shadow_detections)),
            "confidence_correlation": self._calculate_confidence_correlation(primary_detections, shadow_detections)
        }
        
        self.shadow_comparison_results.append(comparison)
        
        # Keep only recent comparisons (last 100)
        if len(self.shadow_comparison_results) > 100:
            self.shadow_comparison_results = self.shadow_comparison_results[-100:]
        
        # Log significant differences
        if comparison["detection_difference"] > 5:  # More than 5 vehicle difference
            logger.warning(f"Significant detection difference: Primary={comparison['primary_count']}, Shadow={comparison['shadow_count']}")
    
    def _calculate_confidence_correlation(self, primary_detections: List[Dict], shadow_detections: List[Dict]) -> float:
        """Calculate correlation between detection confidences."""
        if not primary_detections or not shadow_detections:
            return 0.0
        
        primary_confidences = [d["confidence"] for d in primary_detections]
        shadow_confidences = [d["confidence"] for d in shadow_detections]
        
        # Simple correlation based on mean confidence difference
        primary_mean = np.mean(primary_confidences)
        shadow_mean = np.mean(shadow_confidences)
        
        return 1.0 - abs(primary_mean - shadow_mean)  # Closer means higher correlation

    def estimate_queues(self, frame: np.ndarray) -> List[int]:
        """
        Estimate queue lengths for each configured ROI using enhanced model management.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            List of queue lengths for each ROI/lane
        """
        start_time = time.time()
        
        if self.model_manager is None:
            logger.warning("Model manager not available, returning dummy values")
            return [0] * len(self.rois)
        
        try:
            # Run primary detection
            detections, metrics = self.model_manager.predict(frame, use_shadow=False)
            
            # Run shadow detection for A/B testing if enabled
            shadow_detections = []
            shadow_metrics = None
            if self.shadow_mode_active:
                shadow_detections, shadow_metrics = self.model_manager.predict(frame, use_shadow=True)
                self._compare_shadow_results(detections, shadow_detections, metrics, shadow_metrics)
            
            # Update tracking system with primary detections
            self._update_trackers(detections)
            
            # Estimate queues for each ROI
            queue_lengths = []
            for roi in self.rois:
                queue_length = self._estimate_queue_for_roi(roi)
                queue_lengths.append(queue_length)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.fps_counter.append(1.0 / max(processing_time, 0.001))
            self.last_detection_time = time.time()
            
            # Update detection statistics
            self.detection_stats["total_detections"] += len(detections)
            self.detection_stats["vehicles_detected"] += len(detections)
            
            if self.shadow_mode_active and shadow_detections:
                self.shadow_detection_stats["total_detections"] += len(shadow_detections)
                self.shadow_detection_stats["vehicles_detected"] += len(shadow_detections)
            
            return queue_lengths
            
        except Exception as e:
            logger.error(f"Error in enhanced queue estimation: {e}")
            return [0] * len(self.rois)



    def _update_trackers(self, detections: List[Dict[str, Any]]):
        """Update vehicle trackers with new detections."""
        with self.tracking_lock:
            # Match detections to existing trackers
            matched_trackers = set()
            unmatched_detections = []
            
            for detection in detections:
                best_match = None
                best_distance = float('inf')
                
                for track_id, tracker in self.trackers.items():
                    if track_id in matched_trackers:
                        continue
                    
                    # Calculate distance between detection and tracker
                    distance = np.sqrt(
                        (detection["centroid"][0] - tracker.centroid[0])**2 +
                        (detection["centroid"][1] - tracker.centroid[1])**2
                    )
                    
                    if distance < self.max_tracking_distance and distance < best_distance:
                        best_match = track_id
                        best_distance = distance
                
                if best_match is not None:
                    # Update existing tracker
                    self.trackers[best_match].update_position(
                        detection["centroid"],
                        detection["bbox"],
                        detection["confidence"]
                    )
                    matched_trackers.add(best_match)
                else:
                    unmatched_detections.append(detection)
            
            # Create new trackers for unmatched detections
            for detection in unmatched_detections:
                self.trackers[self.next_track_id] = VehicleTracker(
                    track_id=self.next_track_id,
                    centroid=detection["centroid"],
                    bbox=detection["bbox"],
                    confidence=detection["confidence"],
                    class_id=detection["class_id"]
                )
                self.next_track_id += 1
            
            # Remove old/lost trackers
            current_time = time.time()
            to_remove = []
            for track_id, tracker in self.trackers.items():
                if track_id not in matched_trackers:
                    tracker.frames_missing += 1
                if (tracker.frames_missing > self.max_missing_frames or 
                    current_time - tracker.last_seen > 2.0):  # 2 second timeout
                    to_remove.append(track_id)
            
            for track_id in to_remove:
                del self.trackers[track_id]

    def _estimate_queue_for_roi(self, roi: ROIConfig) -> int:
        """Estimate queue length for a specific ROI."""
        try:
            # Find vehicles within ROI
            vehicles_in_roi = []
            
            with self.tracking_lock:
                for tracker in self.trackers.values():
                    if self._point_in_polygon(tracker.centroid, roi.polygon):
                        vehicles_in_roi.append(tracker)
            
            # Count stationary vehicles (queue indicators)
            stationary_count = sum(1 for v in vehicles_in_roi if v.is_stationary)
            
            # Apply temporal smoothing
            self.queue_history[roi.lane_id].append(stationary_count)
            smoothed_count = int(np.median(list(self.queue_history[roi.lane_id])))
            
            return max(0, smoothed_count)
            
        except Exception as e:
            logger.error(f"Error estimating queue for ROI {roi.lane_id}: {e}")
            return 0

    def _point_in_polygon(self, point: Tuple[float, float], polygon: List[Tuple[int, int]]) -> bool:
        """Check if point is inside polygon using ray casting algorithm."""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside

    def get_debug_visualization(self, frame: np.ndarray) -> np.ndarray:
        """
        Create visualization frame with detections, tracks, and ROIs.
        
        Args:
            frame: Original input frame
            
        Returns:
            Annotated frame with visualizations
        """
        vis_frame = frame.copy()
        
        try:
            # Draw ROIs
            for roi in self.rois:
                # Draw ROI polygon
                pts = np.array(roi.polygon, np.int32)
                cv2.polylines(vis_frame, [pts], True, (0, 255, 0), 2)
                
                # Draw lane label
                if roi.polygon:
                    label_pos = roi.polygon[0]
                    cv2.putText(vis_frame, f"Lane {roi.lane_id}", 
                               label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw vehicle tracks
            with self.tracking_lock:
                for tracker in self.trackers.values():
                    bbox = tracker.bbox
                    centroid = (int(tracker.centroid[0]), int(tracker.centroid[1]))
                    
                    # Color based on status
                    color = (0, 0, 255) if tracker.is_stationary else (255, 0, 0)  # Red for stationary, Blue for moving
                    
                    # Draw bounding box
                    cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    
                    # Draw centroid
                    cv2.circle(vis_frame, centroid, 5, color, -1)
                    
                    # Draw track ID and status
                    label = f"ID:{tracker.track_id}"
                    if tracker.is_stationary:
                        label += " (Q)"
                    
                    cv2.putText(vis_frame, label, (bbox[0], bbox[1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw performance info
            if self.fps_counter:
                fps = np.mean(self.fps_counter)
                cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(vis_frame, f"Vehicles: {len(self.trackers)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return vis_frame
            
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            return frame

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get enhanced performance statistics including model manager data."""
        base_stats = {
            "fps": np.mean(self.fps_counter) if self.fps_counter else 0,
            "active_tracks": len(self.trackers),
            "detection_stats": self.detection_stats.copy(),
            "last_detection_time": self.last_detection_time,
            "model_loaded": self.model_manager is not None
        }
        
        # Add model manager statistics
        if self.model_manager:
            model_summary = self.model_manager.get_performance_summary()
            base_stats.update({
                "model_manager": model_summary,
                "shadow_mode_active": self.shadow_mode_active,
                "shadow_detection_stats": self.shadow_detection_stats.copy() if self.shadow_mode_active else None,
                "shadow_comparison_count": len(self.shadow_comparison_results) if self.shadow_mode_active else 0
            })
            
            # Add recent shadow comparison summary
            if self.shadow_mode_active and self.shadow_comparison_results:
                recent_comparisons = self.shadow_comparison_results[-10:]  # Last 10 comparisons
                base_stats["shadow_comparison_summary"] = {
                    "avg_detection_difference": np.mean([c["detection_difference"] for c in recent_comparisons]),
                    "avg_confidence_correlation": np.mean([c["confidence_correlation"] for c in recent_comparisons]),
                    "avg_fps_difference": np.mean([abs(c["primary_fps"] - c["shadow_fps"]) for c in recent_comparisons])
                }
        
        return base_stats
    
    def get_shadow_comparison_results(self) -> List[Dict[str, Any]]:
        """Get detailed shadow mode comparison results for analysis."""
        return self.shadow_comparison_results.copy()
    
    def switch_to_shadow_model(self) -> bool:
        """Switch active model to shadow model (for manual testing)."""
        if not self.shadow_mode_active or not self.model_manager:
            logger.warning("Shadow mode not active or model manager not available")
            return False
        
        # This would require implementing model switching in model manager
        logger.info("Manual model switch requested - feature placeholder")
        return True
    
    def cleanup(self):
        """Clean up resources including model manager."""
        if self.model_manager:
            self.model_manager.cleanup()
            self.model_manager = None
        
        logger.info("YOLOQueueEstimator cleanup completed")


def process_frame_for_queues(detector: YOLOQueueEstimator, frame: 'np.ndarray', rois: 'List[ROIConfig]', min_stationary_seconds: float = 2.0, debug: bool = False) -> Dict[str, Any]:
    """
    Process a single frame to estimate queues across ROIs with optional visualization.

    Args:
        detector: Initialized YOLOQueueEstimator
        frame: BGR image frame
        rois: List of ROIConfig objects (order corresponds to output mapping)
        min_stationary_seconds: Deprecated. Kept for backward-compat; not used by estimator.
        debug: If True, returns annotated frame as well

    Returns:
        Dict with keys: 'queues' (dict lane_id->int), 'annotations' (optional annotated frame), 'detections'
    """
    # Run queue estimation
    queue_list = detector.estimate_queues(frame)

    # Build mapping lane_id -> queue length using provided rois or detector.rois
    active_rois = rois if rois is not None and len(rois) > 0 else getattr(detector, 'rois', [])
    queues: Dict[int, int] = {}
    for i, roi in enumerate(active_rois):
        val = int(queue_list[i]) if i < len(queue_list) else 0
        queues[roi.lane_id] = val

    # Build lightweight detections list from current trackers
    detections = []
    try:
        for t in getattr(detector, 'trackers', {}).values():
            detections.append({
                'track_id': t.track_id,
                'centroid': t.centroid,
                'bbox': t.bbox,
                'confidence': t.confidence,
                'class_id': t.class_id,
                'is_stationary': t.is_stationary,
            })
    except Exception:
        pass

    result: Dict[str, Any] = {
        'queues': queues,
        'detections': detections,
    }

    if debug:
        try:
            result['annotated'] = detector.get_debug_visualization(frame)
        except Exception:
            result['annotated'] = frame

    return result


def run_stream_queue_estimation(video_stream: 'VideoInputStream', detector: YOLOQueueEstimator, roi_manager: 'ROIManager', duration_sec: Optional[int] = None, debug: bool = False, display_window: bool = False):
    """
    Run queue estimation over a VideoInputStream, yielding results per frame.

    Args:
        video_stream: Initialized and started VideoInputStream
        detector: Initialized YOLOQueueEstimator
        roi_manager: ROIManager with configured ROIs
        duration_sec: Optional duration to run; if None runs until stream stops
        debug: If True, annotate frames
        display_window: If True, show live window (press 'q' to quit)

    Yields:
        Tuple(timestamp, frame_number, queues: Dict[int, int], detections: List[Dict], annotated_frame: Optional[np.ndarray])
    """
    start_time = time.time()
    try:
        while True:
            if duration_sec is not None and time.time() - start_time > duration_sec:
                break

            frame_data = video_stream.get_latest_frame()
            if frame_data is None:
                time.sleep(0.01)
                continue

            res = process_frame_for_queues(detector, frame_data.frame, roi_manager.rois, debug=debug)
            annotated = res.get('annotated') if debug else None

            if display_window and annotated is not None:
                cv2.imshow('Queues', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            yield (frame_data.timestamp, frame_data.frame_number, res.get('queues', {}), res.get('detections', []), annotated)
    finally:
        if display_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
