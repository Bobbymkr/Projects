# Adaptive Traffic Vision

Computer vision pipeline for real-time vehicle detection and queue estimation.

## Overview

This package provides computer vision capabilities for traffic analysis:

- **Vehicle Detection**: YOLOv8-based object detection
- **Queue Estimation**: Real-time queue length estimation from video
- **Video Processing**: Multi-source video input support
- **ROI Management**: Region of Interest configuration and management

## Installation

```bash
pip install -e .
```

## Dependencies

- `adaptive-traffic-common` - Shared utilities

## Quick Start

```python
from adaptive_traffic_vision.vision import YOLOQueueEstimator, VideoInputStream

# Initialize video input
video = VideoInputStream(source=0)  # Webcam

# Initialize queue estimator
estimator = YOLOQueueEstimator(model_path="models/yolov8n.pt")

# Process frames
for frame in video:
    queues = estimator.estimate_queues(frame)
    print(f"Queue lengths: {queues}")
```

## Features

- Real-time vehicle detection using YOLOv8
- Queue length estimation from video feeds
- Support for webcam, video files, and IP cameras
- Configurable ROI (Region of Interest) management
- Performance optimization for real-time processing

## Project Structure

```
adaptive-traffic-vision/
├── src/
│   └── vision/           # Vision code
│       ├── yolo_queue.py
│       ├── video_pipeline.py
│       └── roi_helper.py
├── models/               # Model files
│   └── yolov8n.pt
└── README.md
```

## Model Files

The YOLOv8 model (`yolov8n.pt`) is included in the `models/` directory. For custom models, place them in this directory.

## License

MIT License

