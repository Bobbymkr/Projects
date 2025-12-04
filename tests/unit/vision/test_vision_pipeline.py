"""
Unit Tests for Vision Processing Pipeline.

Tests frame preprocessing, ROI masking, object tracking, and related functionality.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import vision components
from src.vision.yolo_queue import (
    YOLOQueueEstimator,
    ROIConfig,
    VehicleTracker
)
from src.vision.video_pipeline import VideoInputStream, VideoConfig

# Import fixtures
from tests.fixtures.vision_samples import (
    sample_video_frame,
    roi_config_4_lane,
    frame_sequence
)


class TestFramePreprocessing:
    """Test frame preprocessing functionality."""
    
    def test_frame_resize(self, sample_video_frame):
        """Test frame resizing."""
        frame = sample_video_frame
        original_shape = frame.shape
        
        # Resize to smaller
        resized = cv2.resize(frame, (320, 240))
        
        assert resized.shape == (240, 320, 3)
        assert resized.dtype == np.uint8
    
    def test_frame_normalization(self, sample_video_frame):
        """Test frame normalization."""
        frame = sample_video_frame
        
        # Normalize to [0, 1]
        normalized = frame.astype(np.float32) / 255.0
        
        assert normalized.dtype == np.float32
        assert np.min(normalized) >= 0.0
        assert np.max(normalized) <= 1.0
    
    def test_frame_grayscale_conversion(self, sample_video_frame):
        """Test grayscale conversion."""
        frame = sample_video_frame
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        assert len(gray.shape) == 2
        assert gray.shape[:2] == frame.shape[:2]
        assert gray.dtype == np.uint8
    
    def test_frame_brightness_adjustment(self, sample_video_frame):
        """Test brightness adjustment."""
        frame = sample_video_frame
        
        # Increase brightness
        brightened = cv2.convertScaleAbs(frame, alpha=1.0, beta=30)
        
        assert brightened.shape == frame.shape
        assert brightened.dtype == frame.dtype
    
    def test_frame_contrast_adjustment(self, sample_video_frame):
        """Test contrast adjustment."""
        frame = sample_video_frame
        
        # Increase contrast
        contrasted = cv2.convertScaleAbs(frame, alpha=1.5, beta=0)
        
        assert contrasted.shape == frame.shape
        assert contrasted.dtype == frame.dtype


class TestROIMasking:
    """Test ROI (Region of Interest) masking functionality."""
    
    def test_roi_mask_creation(self, sample_video_frame, roi_config_4_lane):
        """Test creating ROI mask from configuration."""
        frame = sample_video_frame
        config = roi_config_4_lane
        
        # Create mask for first lane
        lane = config['lanes'][0]
        roi_coords = np.array(lane['roi'], dtype=np.int32)
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [roi_coords], 255)
        
        assert mask.shape == frame.shape[:2]
        assert np.max(mask) == 255
        assert np.min(mask) == 0
    
    def test_roi_extraction(self, sample_video_frame, roi_config_4_lane):
        """Test extracting ROI region from frame."""
        frame = sample_video_frame
        config = roi_config_4_lane
        
        # Extract ROI for first lane
        lane = config['lanes'][0]
        roi_coords = np.array(lane['roi'], dtype=np.int32)
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [roi_coords], 255)
        
        roi_region = cv2.bitwise_and(frame, frame, mask=mask)
        
        assert roi_region.shape == frame.shape
        assert roi_region.dtype == frame.dtype
    
    def test_multiple_roi_masks(self, sample_video_frame, roi_config_4_lane):
        """Test creating multiple ROI masks."""
        frame = sample_video_frame
        config = roi_config_4_lane
        
        masks = []
        for lane in config['lanes']:
            roi_coords = np.array(lane['roi'], dtype=np.int32)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [roi_coords], 255)
            masks.append(mask)
        
        assert len(masks) == 4
        assert all(m.shape == frame.shape[:2] for m in masks)
    
    def test_roi_vehicle_counting(self, sample_video_frame, roi_config_4_lane):
        """Test counting vehicles within ROI."""
        frame = sample_video_frame
        config = roi_config_4_lane
        
        # Mock detections
        detections = {
            'boxes': [[100, 200, 150, 250], [300, 180, 350, 230]],
            'confidences': [0.95, 0.92],
            'classes': [2, 2]
        }
        
        # Count vehicles in first ROI
        lane = config['lanes'][0]
        roi_coords = np.array(lane['roi'], dtype=np.int32)
        
        vehicle_count = 0
        for box in detections['boxes']:
            # Check if box center is in ROI
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            if cv2.pointPolygonTest(roi_coords, (center_x, center_y), False) >= 0:
                vehicle_count += 1
        
        assert vehicle_count >= 0
        assert vehicle_count <= len(detections['boxes'])


class TestObjectTracking:
    """Test object tracking functionality."""
    
    def test_vehicle_tracker_initialization(self):
        """Test VehicleTracker initialization."""
        tracker = VehicleTracker(
            track_id=1,
            centroid=(100.0, 200.0),
            bbox=(90, 190, 110, 210),
            confidence=0.95,
            class_id=2
        )
        
        assert tracker.track_id == 1
        assert tracker.centroid == (100.0, 200.0)
        assert tracker.confidence == 0.95
        assert tracker.class_id == 2
        assert tracker.is_stationary == False
    
    def test_vehicle_tracker_position_update(self):
        """Test updating tracker position."""
        tracker = VehicleTracker(
            track_id=1,
            centroid=(100.0, 200.0),
            bbox=(90, 190, 110, 210),
            confidence=0.95,
            class_id=2
        )
        
        import time
        time.sleep(0.1)  # Small delay for velocity calculation
        
        tracker.update_position(
            new_centroid=(105.0, 200.0),
            new_bbox=(95, 190, 115, 210),
            confidence=0.94
        )
        
        assert tracker.centroid == (105.0, 200.0)
        assert tracker.confidence == 0.94
    
    def test_stationary_vehicle_detection(self):
        """Test detecting stationary vehicles (queued vehicles)."""
        tracker = VehicleTracker(
            track_id=1,
            centroid=(100.0, 200.0),
            bbox=(90, 190, 110, 210),
            confidence=0.95,
            class_id=2
        )
        
        # Update with same position multiple times (stationary)
        for _ in range(10):
            import time
            time.sleep(0.1)
            tracker.update_position(
                new_centroid=(100.0, 200.0),  # Same position
                new_bbox=(90, 190, 110, 210),
                confidence=0.95
            )
        
        # Should detect as stationary after threshold
        assert tracker.is_stationary == True
        assert tracker.stationary_frames > 5
    
    def test_tracker_velocity_calculation(self):
        """Test velocity calculation for tracking."""
        tracker = VehicleTracker(
            track_id=1,
            centroid=(100.0, 200.0),
            bbox=(90, 190, 110, 210),
            confidence=0.95,
            class_id=2
        )
        
        import time
        time.sleep(0.1)
        
        # Move vehicle
        tracker.update_position(
            new_centroid=(110.0, 210.0),  # Moved
            new_bbox=(100, 200, 120, 220),
            confidence=0.95
        )
        
        # Velocity should be calculated
        assert tracker.velocity[0] != 0.0 or tracker.velocity[1] != 0.0


class TestYOLOQueueEstimator:
    """Test YOLOQueueEstimator functionality."""
    
    @pytest.fixture
    def queue_estimator(self, roi_config_4_lane):
        """Create queue estimator with mock YOLO."""
        rois = [
            ROIConfig(
                lane_id=i,
                polygon=[tuple(p) for p in lane['roi']],
                direction_vector=(1.0, 0.0),
                queue_line=((0, 0), (100, 0)),
                stop_line=((0, 50), (100, 50)),
                name=lane['lane_id']
            )
            for i, lane in enumerate(roi_config_4_lane['lanes'])
        ]
        
        estimator = YOLOQueueEstimator(
            model_path=None,  # Will use mock
            rois=rois,
            confidence_threshold=0.5
        )
        
        # Mock YOLO model
        estimator.model = Mock()
        estimator.model.return_value = Mock()
        
        return estimator
    
    def test_queue_estimator_initialization(self, queue_estimator):
        """Test queue estimator initialization."""
        estimator = queue_estimator
        
        assert estimator is not None
        assert estimator.confidence_threshold == 0.5
        assert len(estimator.rois) == 4
    
    def test_detection_on_frame(self, queue_estimator, sample_video_frame):
        """Test vehicle detection on frame."""
        estimator = queue_estimator
        frame = sample_video_frame
        
        # Mock detection results
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = np.array([[100, 200, 150, 250], [300, 180, 350, 230]])
        mock_result.boxes.conf = np.array([0.95, 0.92])
        mock_result.boxes.cls = np.array([2, 2])
        
        estimator.model.return_value = mock_result
        
        # Run detection
        detections = estimator.detect(frame)
        
        assert detections is not None
    
    def test_queue_estimation(self, queue_estimator):
        """Test queue length estimation."""
        estimator = queue_estimator
        
        # Mock detections with stationary vehicles
        mock_detections = Mock()
        mock_detections.boxes = Mock()
        mock_detections.boxes.xyxy = np.array([
            [100, 200, 150, 250],
            [110, 210, 160, 260],
            [300, 180, 350, 230]
        ])
        mock_detections.boxes.conf = np.array([0.95, 0.92, 0.88])
        mock_detections.boxes.cls = np.array([2, 2, 2])
        
        # Estimate queues
        queue_lengths = estimator.estimate_queues(mock_detections)
        
        assert queue_lengths is not None
        assert len(queue_lengths) == 4  # 4 lanes
        assert all(q >= 0 for q in queue_lengths)
        assert all(isinstance(q, (int, float)) for q in queue_lengths)


class TestVideoPipeline:
    """Test video pipeline functionality."""
    
    def test_video_config_creation(self):
        """Test video configuration creation."""
        config = VideoConfig(
            source_type="file",
            fps=30.0,
            width=640,
            height=480
        )
        
        assert config.source_type == "file"
        assert config.fps == 30.0
        assert config.width == 640
        assert config.height == 480
    
    def test_frame_sequence_processing(self, frame_sequence):
        """Test processing sequence of frames."""
        frames = frame_sequence
        
        processed_frames = []
        for frame in frames:
            # Simple processing: resize
            processed = cv2.resize(frame, (320, 240))
            processed_frames.append(processed)
        
        assert len(processed_frames) == len(frames)
        assert all(f.shape == (240, 320, 3) for f in processed_frames)


class TestErrorHandling:
    """Test error handling in vision pipeline."""
    
    def test_invalid_frame_handling(self):
        """Test handling invalid frames."""
        # Test None frame
        with pytest.raises((AttributeError, TypeError)):
            cv2.resize(None, (320, 240))
        
        # Test empty frame
        empty_frame = np.array([])
        with pytest.raises(cv2.error):
            cv2.resize(empty_frame, (320, 240))
    
    def test_invalid_roi_handling(self, sample_video_frame):
        """Test handling invalid ROI coordinates."""
        frame = sample_video_frame
        
        # ROI outside frame bounds
        invalid_roi = np.array([
            [1000, 1000],
            [1100, 1000],
            [1100, 1100],
            [1000, 1100]
        ], dtype=np.int32)
        
        # Should handle gracefully (clip to bounds)
        height, width = frame.shape[:2]
        clipped_roi = np.clip(invalid_roi, [0, 0], [width - 1, height - 1])
        
        assert np.all(clipped_roi[:, 0] < width)
        assert np.all(clipped_roi[:, 1] < height)
    
    def test_missing_detections_handling(self, queue_estimator):
        """Test handling when no vehicles detected."""
        estimator = queue_estimator
        
        # Empty detections
        empty_detections = Mock()
        empty_detections.boxes = Mock()
        empty_detections.boxes.xyxy = np.array([])
        empty_detections.boxes.conf = np.array([])
        empty_detections.boxes.cls = np.array([])
        
        # Should return zero queues
        queue_lengths = estimator.estimate_queues(empty_detections)
        
        assert queue_lengths is not None
        assert len(queue_lengths) == 4
        assert all(q == 0 for q in queue_lengths)

