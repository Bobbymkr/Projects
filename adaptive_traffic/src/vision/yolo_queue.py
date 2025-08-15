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
                 model_path: Optional[str] = None, 
                 rois: Optional[List[ROIConfig]] = None,
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4,
                 max_tracking_distance: float = 50.0,
                 max_missing_frames: int = 10,
                 queue_smoothing_window: int = 5):
        """
        Initialize YOLO-based queue estimator.
        
        Args:
            model_path: Path to YOLOv8 model file (.pt)
            rois: List of ROI configurations for each lane
            confidence_threshold: Minimum detection confidence
            nms_threshold: Non-maximum suppression threshold
            max_tracking_distance: Maximum distance for track association
            max_missing_frames: Maximum frames before dropping a track
            queue_smoothing_window: Frames for temporal smoothing
        """
        self.model_path = model_path or self._download_default_model()
        self.rois = rois or []
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_tracking_distance = max_tracking_distance
        self.max_missing_frames = max_missing_frames
        
        # Initialize YOLO model
        self.model = self._load_yolo_model()
        
        # Vehicle classes from COCO dataset
        self.vehicle_classes = {2, 3, 5, 7}  # car, motorcycle, bus, truck
        
        # Tracking system
        self.trackers: Dict[int, VehicleTracker] = {}
        self.next_track_id = 0
        self.tracking_lock = threading.Lock()
        
        # Queue estimation with temporal smoothing
        self.queue_history = defaultdict(lambda: deque(maxlen=queue_smoothing_window))
        self.last_detection_time = 0
        
        # Performance monitoring
        self.fps_counter = deque(maxlen=30)
        self.detection_stats = {"total_detections": 0, "vehicles_detected": 0}
        
        logger.info(f"YOLOQueueEstimator initialized with {len(self.rois)} ROIs")

    def _download_default_model(self) -> str:
        """Download YOLOv8n model if not provided."""
        try:
            from ultralytics import YOLO
            model_path = "yolov8n.pt"
            # This will auto-download if not present
            YOLO(model_path)
            return model_path
        except ImportError:
            logger.warning("ultralytics not installed, using OpenCV DNN fallback")
            return None

    def _load_yolo_model(self):
        """Load YOLO model with error handling."""
        try:
            from ultralytics import YOLO
            if self.model_path and Path(self.model_path).exists():
                model = YOLO(self.model_path)
                logger.info(f"Loaded YOLO model from {self.model_path}")
                return model
            else:
                # Fallback to YOLOv8n
                model = YOLO("yolov8n.pt")
                logger.info("Using YOLOv8n model")
                return model
        except ImportError:
            logger.error("ultralytics package not installed. Install with: pip install ultralytics")
            return None
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return None

    def estimate_queues(self, frame: np.ndarray) -> List[int]:
        """
        Estimate queue lengths for each configured ROI.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            List of queue lengths for each ROI/lane
        """
        start_time = time.time()
        
        if self.model is None:
            logger.warning("YOLO model not loaded, returning dummy values")
            return [0] * len(self.rois)
        
        try:
            # Run YOLO detection
            detections = self._detect_vehicles(frame)
            
            # Update tracking system
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
            
            return queue_lengths
            
        except Exception as e:
            logger.error(f"Error in queue estimation: {e}")
            return [0] * len(self.rois)

    def _detect_vehicles(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLO detection on frame and filter for vehicles."""
        try:
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold, iou=self.nms_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        # Extract detection data
                        bbox = boxes.xyxy[i].cpu().numpy().astype(int)  # [x1, y1, x2, y2]
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        
                        # Filter for vehicles only
                        if class_id in self.vehicle_classes and confidence >= self.confidence_threshold:
                            centroid = (
                                (bbox[0] + bbox[2]) / 2,
                                (bbox[1] + bbox[3]) / 2
                            )
                            
                            detections.append({
                                "bbox": tuple(bbox),
                                "centroid": centroid,
                                "confidence": confidence,
                                "class_id": class_id
                            })
            
            self.detection_stats["total_detections"] += len(detections)
            self.detection_stats["vehicles_detected"] += len(detections)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []

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
        """Get current performance statistics."""
        return {
            "fps": np.mean(self.fps_counter) if self.fps_counter else 0,
            "active_tracks": len(self.trackers),
            "detection_stats": self.detection_stats.copy(),
            "last_detection_time": self.last_detection_time,
            "model_loaded": self.model is not None
        }


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
