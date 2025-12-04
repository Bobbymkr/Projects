"""
Vision Test Fixtures and Sample Data.

Provides video samples, frame sequences, ROI configurations, and ground truth data.
"""

import pytest
import numpy as np
import cv2
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json


@pytest.fixture
def sample_video_frame() -> np.ndarray:
    """Generate a sample video frame for testing."""
    # Create a 640x480 RGB frame with some "vehicles" (rectangles)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some colored rectangles to simulate vehicles
    cv2.rectangle(frame, (100, 200), (150, 250), (0, 255, 0), -1)  # Green "car"
    cv2.rectangle(frame, (300, 180), (350, 230), (255, 0, 0), -1)  # Blue "car"
    cv2.rectangle(frame, (500, 220), (550, 270), (0, 0, 255), -1)  # Red "car"
    
    return frame


@pytest.fixture
def sample_video_file(tmp_path) -> Path:
    """Create a sample video file for testing."""
    video_path = tmp_path / "test_traffic.mp4"
    
    # Create a simple video with 30 frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (640, 480))
    
    for i in range(30):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Add some moving rectangles
        x = (i * 10) % 600
        cv2.rectangle(frame, (x, 200), (x + 50, 250), (0, 255, 0), -1)
        out.write(frame)
    
    out.release()
    return video_path


@pytest.fixture
def roi_config_4_lane() -> Dict[str, Any]:
    """ROI configuration for 4-lane intersection."""
    return {
        "intersection_id": "test_intersection",
        "lanes": [
            {
                "lane_id": "lane_0",
                "roi": [[100, 100], [200, 100], [200, 300], [100, 300]],
                "direction": "north"
            },
            {
                "lane_id": "lane_1",
                "roi": [[300, 100], [400, 100], [400, 300], [300, 300]],
                "direction": "south"
            },
            {
                "lane_id": "lane_2",
                "roi": [[100, 300], [200, 300], [200, 500], [100, 500]],
                "direction": "east"
            },
            {
                "lane_id": "lane_3",
                "roi": [[300, 300], [400, 300], [400, 500], [300, 500]],
                "direction": "west"
            }
        ],
        "frame_size": [640, 480]
    }


@pytest.fixture
def roi_config_file(tmp_path, roi_config_4_lane) -> Path:
    """Save ROI configuration to file."""
    config_file = tmp_path / "roi_config.json"
    with open(config_file, 'w') as f:
        json.dump(roi_config_4_lane, f, indent=2)
    return config_file


@pytest.fixture
def ground_truth_detections() -> List[Dict[str, Any]]:
    """Ground truth vehicle detections for testing."""
    return [
        {
            "frame_id": 0,
            "detections": [
                {
                    "bbox": [100, 200, 150, 250],
                    "confidence": 0.95,
                    "class": "car",
                    "lane_id": "lane_0"
                },
                {
                    "bbox": [300, 180, 350, 230],
                    "confidence": 0.92,
                    "class": "car",
                    "lane_id": "lane_1"
                }
            ]
        },
        {
            "frame_id": 1,
            "detections": [
                {
                    "bbox": [110, 200, 160, 250],
                    "confidence": 0.94,
                    "class": "car",
                    "lane_id": "lane_0"
                },
                {
                    "bbox": [310, 180, 360, 230],
                    "confidence": 0.91,
                    "class": "car",
                    "lane_id": "lane_1"
                }
            ]
        }
    ]


@pytest.fixture
def mock_yolo_detector():
    """Mock YOLOv8 detector for testing."""
    from unittest.mock import Mock, MagicMock
    
    mock_detector = MagicMock()
    
    # Mock detection results
    mock_detection_result = MagicMock()
    mock_detection_result.boxes = MagicMock()
    mock_detection_result.boxes.xyxy = np.array([
        [100, 200, 150, 250],
        [300, 180, 350, 230]
    ])
    mock_detection_result.boxes.conf = np.array([0.95, 0.92])
    mock_detection_result.boxes.cls = np.array([2, 2])  # Car class
    
    mock_detector.detect.return_value = mock_detection_result
    mock_detector.estimate_queues.return_value = [2, 1, 0, 0]
    
    return mock_detector


@pytest.fixture
def frame_sequence() -> List[np.ndarray]:
    """Generate a sequence of frames for testing."""
    frames = []
    for i in range(10):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Add moving object
        x = (i * 20) % 600
        cv2.rectangle(frame, (x, 200), (x + 50, 250), (0, 255, 0), -1)
        frames.append(frame)
    return frames


@pytest.fixture
def video_input_stream_config() -> Dict[str, Any]:
    """Configuration for video input stream."""
    return {
        "source_type": "file",
        "fps": 30.0,
        "width": 640,
        "height": 480,
        "buffer_size": 10,
        "preprocessing": {
            "resize": True,
            "normalize": False
        }
    }

