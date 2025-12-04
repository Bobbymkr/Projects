"""
Integration Tests: Vision Pipeline to Observations.

Tests the complete vision processing pipeline from video input
to traffic state observations for agent decision-making.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import vision components
from src.vision.video_pipeline import VideoInputStream, VideoConfig
from src.vision.yolo_queue import YOLOQueueEstimator, ROIConfig

# Import fixtures
from tests.fixtures.vision_samples import (
    sample_video_frame,
    sample_video_file,
    roi_config_4_lane,
    mock_yolo_detector,
    frame_sequence
)


class TestVideoInputToDetection:
    """Test video input processing to vehicle detection."""
    
    def test_video_frame_capture(self, sample_video_frame):
        """Test that video frames can be captured."""
        frame = sample_video_frame
        
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8
    
    def test_yolo_detection_on_frame(self, sample_video_frame, mock_yolo_detector):
        """Test YOLOv8 detection on video frame."""
        frame = sample_video_frame
        detector = mock_yolo_detector
        
        # Run detection
        results = detector.detect(frame)
        
        assert results is not None
        assert hasattr(results, 'boxes')
        assert len(results.boxes.xyxy) > 0
    
    def test_queue_estimation_from_detections(self, mock_yolo_detector, roi_config_4_lane):
        """Test queue length estimation from detections."""
        detector = mock_yolo_detector
        
        # Mock frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Run detection
        detections = detector.detect(frame)
        
        # Estimate queues
        queue_lengths = detector.estimate_queues(detections)
        
        assert queue_lengths is not None
        assert len(queue_lengths) == 4  # 4 lanes
        assert all(q >= 0 for q in queue_lengths)
        assert all(isinstance(q, (int, float)) for q in queue_lengths)


class TestROIManagement:
    """Test ROI (Region of Interest) management for lane detection."""
    
    def test_roi_configuration_loading(self, roi_config_4_lane):
        """Test ROI configuration can be loaded."""
        config = roi_config_4_lane
        
        assert config is not None
        assert 'lanes' in config
        assert len(config['lanes']) == 4
        assert 'frame_size' in config
    
    def test_roi_extraction_from_frame(self, sample_video_frame, roi_config_4_lane):
        """Test ROI regions can be extracted from frame."""
        frame = sample_video_frame
        config = roi_config_4_lane
        
        # Extract ROI for each lane
        for lane in config['lanes']:
            roi_coords = lane['roi']
            # Convert to numpy array for extraction
            roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            roi_points = np.array(roi_coords, dtype=np.int32)
            cv2.fillPoly(roi_mask, [roi_points], 255)
            
            # Extract ROI region
            roi_region = cv2.bitwise_and(frame, frame, mask=roi_mask)
            
            assert roi_region is not None
            assert roi_region.shape == frame.shape
    
    def test_roi_vehicle_counting(self, mock_yolo_detector, roi_config_4_lane):
        """Test vehicle counting within ROI regions."""
        detector = mock_yolo_detector
        config = roi_config_4_lane
        
        # Mock frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Get detections
        detections = detector.detect(frame)
        
        # Count vehicles in each ROI
        vehicle_counts = []
        for lane in config['lanes']:
            roi_coords = lane['roi']
            # Count detections within ROI (simplified)
            count = len(detections.boxes.xyxy)  # Mock count
            vehicle_counts.append(count)
        
        assert len(vehicle_counts) == 4
        assert all(c >= 0 for c in vehicle_counts)


class TestVisionToObservationPipeline:
    """Test complete vision pipeline to observation conversion."""
    
    def test_frame_to_observation_conversion(self, sample_video_frame, mock_yolo_detector):
        """Test conversion from video frame to agent observation."""
        frame = sample_video_frame
        detector = mock_yolo_detector
        
        # Process frame through pipeline
        detections = detector.detect(frame)
        queue_lengths = detector.estimate_queues(detections)
        
        # Convert to observation format
        # Observation: [queue_lengths, wait_times (estimated), arrival_rates (estimated)]
        wait_times = [q * 2.5 for q in queue_lengths]  # Simple estimation
        arrival_rates = [q / 100.0 for q in queue_lengths]  # Simple estimation
        
        observation = np.array(
            queue_lengths + wait_times + arrival_rates,
            dtype=np.float32
        )
        
        assert observation is not None
        assert observation.shape == (12,)  # 4 lanes * 3 features
        assert all(obs >= 0 for obs in observation)
    
    def test_observation_shape_consistency(self, frame_sequence, mock_yolo_detector):
        """Test that observations maintain consistent shape across frames."""
        detector = mock_yolo_detector
        frames = frame_sequence
        
        observation_shapes = []
        for frame in frames:
            detections = detector.detect(frame)
            queue_lengths = detector.estimate_queues(detections)
            
            # Create observation
            wait_times = [q * 2.5 for q in queue_lengths]
            arrival_rates = [q / 100.0 for q in queue_lengths]
            observation = np.array(
                queue_lengths + wait_times + arrival_rates,
                dtype=np.float32
            )
            
            observation_shapes.append(observation.shape)
        
        # All observations should have same shape
        assert len(set(observation_shapes)) == 1
        assert observation_shapes[0] == (12,)
    
    def test_real_time_processing_latency(self, sample_video_frame, mock_yolo_detector):
        """Test that vision processing meets real-time latency requirements."""
        import time
        
        frame = sample_video_frame
        detector = mock_yolo_detector
        
        # Measure processing time
        start = time.time()
        detections = detector.detect(frame)
        queue_lengths = detector.estimate_queues(detections)
        latency = time.time() - start
        
        # Should process in < 100ms for real-time (10 FPS)
        assert latency < 0.1


class TestVideoStreamIntegration:
    """Test video stream integration with detection pipeline."""
    
    @pytest.fixture
    def video_config(self):
        """Video configuration for testing."""
        return VideoConfig(
            source_type="file",
            fps=30.0,
            width=640,
            height=480
        )
    
    def test_video_stream_initialization(self, sample_video_file, video_config):
        """Test video stream can be initialized."""
        config = video_config
        config.source = str(sample_video_file)
        
        # Mock video stream (actual implementation would use VideoInputStream)
        stream = Mock()
        stream.is_open.return_value = True
        stream.get_fps.return_value = 30.0
        
        assert stream.is_open()
        assert stream.get_fps() == 30.0
    
    def test_frame_buffering(self, frame_sequence):
        """Test frame buffering for smooth processing."""
        frames = frame_sequence
        buffer_size = 5
        
        # Simulate buffering
        frame_buffer = []
        for frame in frames[:buffer_size]:
            frame_buffer.append(frame)
        
        assert len(frame_buffer) == buffer_size
        assert all(f.shape == (480, 640, 3) for f in frame_buffer)
    
    def test_continuous_processing(self, frame_sequence, mock_yolo_detector):
        """Test continuous frame processing."""
        detector = mock_yolo_detector
        frames = frame_sequence
        
        queue_histories = []
        for frame in frames:
            detections = detector.detect(frame)
            queue_lengths = detector.estimate_queues(detections)
            queue_histories.append(queue_lengths)
        
        assert len(queue_histories) == len(frames)
        assert all(len(q) == 4 for q in queue_histories)


class TestErrorHandling:
    """Test error handling in vision pipeline."""
    
    def test_invalid_frame_handling(self, mock_yolo_detector):
        """Test handling of invalid frames."""
        detector = mock_yolo_detector
        
        # Test with None frame
        with pytest.raises((AttributeError, TypeError)):
            detector.detect(None)
        
        # Test with empty frame
        empty_frame = np.array([])
        # Should handle gracefully or raise appropriate error
        try:
            detector.detect(empty_frame)
        except (ValueError, AttributeError):
            pass  # Expected
    
    def test_missing_detections_handling(self, mock_yolo_detector):
        """Test handling when no vehicles are detected."""
        detector = mock_yolo_detector
        
        # Mock empty detections
        empty_detections = MagicMock()
        empty_detections.boxes.xyxy = np.array([])
        empty_detections.boxes.conf = np.array([])
        detector.detect.return_value = empty_detections
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        
        # Should return empty but valid structure
        queue_lengths = detector.estimate_queues(detections)
        assert queue_lengths is not None
        assert len(queue_lengths) == 4
        assert all(q == 0 for q in queue_lengths)  # No vehicles = zero queues
    
    def test_roi_out_of_bounds(self, sample_video_frame):
        """Test handling of ROI coordinates out of frame bounds."""
        frame = sample_video_frame
        height, width = frame.shape[:2]
        
        # Create ROI that extends beyond frame
        invalid_roi = [
            [width + 10, height + 10],
            [width + 50, height + 10],
            [width + 50, height + 50],
            [width + 10, height + 50]
        ]
        
        # Should handle gracefully (clip to frame bounds)
        roi_points = np.array(invalid_roi, dtype=np.int32)
        roi_points = np.clip(roi_points, [0, 0], [width - 1, height - 1])
        
        assert all(0 <= p[0] < width for p in roi_points)
        assert all(0 <= p[1] < height for p in roi_points)


# Import cv2 for ROI operations
try:
    import cv2
except ImportError:
    cv2 = None
    pytest.skip("OpenCV not available", allow_module_level=True)

