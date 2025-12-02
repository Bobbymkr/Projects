# Video Inference Capability Assessment

## Executive Summary

**YES, your project IS ready to predict optimal green time based on video footage!** ✅

The system has complete video processing infrastructure and can analyze traffic footage to recommend optimal green light durations.

---

## Current Capabilities

### ✅ **What Works:**

1. **Video Input Processing**
   - Supports multiple video sources:
     - Video files (`.mp4`, `.avi`, etc.)
     - Webcam (USB camera)
     - Network streams (RTSP, HTTP)
   - Automatic frame buffering and preprocessing
   - Configurable FPS, resolution, and processing parameters

2. **Computer Vision Pipeline**
   - **YOLOv8** vehicle detection (auto-downloads model if missing)
   - **Queue length estimation** per lane
   - **ROI (Region of Interest)** management for lane segmentation
   - **Object tracking** for stationary vehicle detection

3. **Green Time Prediction**
   - Multiple algorithms available:
     - **DQN (Deep Q-Network)** - AI-based optimal timing
     - **Fuzzy Logic** - Rule-based control
     - **Genetic Algorithm** - Optimization-based
     - **PSO (Particle Swarm)** - Swarm intelligence
     - **GNN Forecasting** - Graph neural network
     - **Webster's Method** - Traditional traffic engineering

4. **Real-Time Processing**
   - Processes video frames in real-time
   - Aggregates observations over time windows
   - Provides continuous recommendations

---

## How to Use Video Inference

### **Basic Usage:**

```bash
# Process a video file
python src/rl/inference.py video \
    --model runs/dqn_traffic.npz \
    --video_source path/to/your/video.mp4 \
    --config configs/intersection.json

# Use webcam (camera index 0)
python src/rl/inference.py video \
    --model runs/dqn_traffic.npz \
    --video_source 0 \
    --config configs/intersection.json

# Process network stream
python src/rl/inference.py video \
    --model runs/dqn_traffic.npz \
    --video_source rtsp://camera-ip:554/stream \
    --config configs/intersection.json
```

### **Advanced Options:**

```bash
# With custom ROI configuration (for lane detection)
python src/rl/inference.py video \
    --model runs/dqn_traffic.npz \
    --video_source traffic.mp4 \
    --roi_config configs/roi_config.json \
    --fps 30.0 \
    --width 1280 \
    --height 720 \
    --warmup 3

# Visual inference with overlay (shows detections)
python src/rl/inference_viz.py \
    --model runs/dqn_traffic.npz \
    --video_source traffic.mp4 \
    --continuous
```

---

## System Workflow

```
Video Footage
    ↓
[VideoInputStream] → Frame Capture & Buffering
    ↓
[YOLOQueueEstimator] → Vehicle Detection & Tracking
    ↓
[ROIManager] → Lane Segmentation & Queue Counting
    ↓
[VideoTrafficEnv] → State Observation (Queue Lengths)
    ↓
[DQN Agent / Fuzzy Controller / etc.] → Action Selection
    ↓
Optimal Green Time Recommendation (5-60 seconds)
```

---

## Requirements & Prerequisites

### **1. Trained Model**
You need a trained model file:
- **DQN**: `runs/dqn_traffic.npz` (or `.zip` for Stable-Baselines3)
- **Fuzzy Logic**: No model needed (rule-based)
- **Other methods**: Check respective model paths

**To train a model:**
```bash
# Quick training (5 episodes)
python src/rl/train_dqn.py --episodes 5 --config configs/intersection.json

# Production training (6000 episodes)
python src/rl/train_dqn.py --episodes 6000 --out runs/production
```

### **2. Configuration Files**
- **Traffic Config**: `configs/intersection.json` (defines lanes, phases, timing constraints)
- **ROI Config** (optional): Defines lane regions in video frame

### **3. Video Requirements**
- **Format**: Any format supported by OpenCV (MP4, AVI, MOV, etc.)
- **Resolution**: Recommended 640x480 or higher
- **Content**: Should show traffic intersection with visible lanes
- **Camera Angle**: Top-down or angled view works best

---

## Current Limitations & Considerations

### ⚠️ **Important Notes:**

1. **ROI Configuration**
   - If no ROI config is provided, the system will auto-generate default ROIs
   - For best results, manually configure ROIs to match your video's lane layout
   - ROI config defines which regions of the frame correspond to which lanes

2. **Model Training**
   - Models are typically trained on simulated traffic
   - Real-world video may have different characteristics
   - Consider fine-tuning or training on video data for better accuracy

3. **Single Observation Mode**
   - Current `run_video_inference` processes one observation and exits
   - For continuous monitoring, use `inference_viz.py` with `--continuous` flag

4. **Method Selection**
   - ✅ **FIXED**: Now supports `--method` parameter in CLI
   - Defaults to Fuzzy Logic (no model required)
   - All methods (DQN, Fuzzy, GA, PSO, GNN, Webster) are now supported

---

## What Gets Predicted

The system analyzes video footage and outputs:

- **Queue Lengths**: Number of vehicles waiting in each lane
- **Recommended Green Time**: Optimal duration (5-60 seconds, in 5-second steps)
- **Confidence**: Based on queue differences and traffic patterns

**Example Output:**
```
Recommended green time (seconds): 25
```

This means the system recommends a 25-second green light duration based on the current traffic conditions observed in the video.

---

## Testing Your Setup

### **Quick Test (No Model Required):**
```bash
# Test with Fuzzy Logic (no training needed)
python src/rl/inference.py video \
    --model dummy \
    --video_source your_video.mp4 \
    --config configs/intersection.json
```

### **Full Test with DQN:**
1. Train a model first (see above)
2. Run inference on your video
3. Check the recommended green time output

---

## Next Steps & Recommendations

### **To Make It Production-Ready:**

1. **✅ Bugs Fixed**
   - ✅ Removed duplicate code in `run_video_inference`
   - ✅ Added `--method` parameter support to CLI
   - ✅ Added support for all methods (GA, PSO, GNN, Webster)
   - ✅ Improved error handling for model requirements

2. **ROI Configuration**
   - Create custom ROI config for your specific intersection
   - Calibrate lane regions to match your video angle

3. **Model Training**
   - Train on video data if available
   - Fine-tune hyperparameters for your use case
   - Consider ensemble methods for better accuracy

4. **Continuous Operation**
   - Use `inference_viz.py` for real-time monitoring
   - Set up logging and metrics collection
   - Integrate with traffic signal control hardware

---

## Summary

✅ **Your project CAN:**
- Process video footage (files, webcam, streams)
- Detect vehicles using YOLOv8
- Estimate queue lengths per lane
- Predict optimal green time using multiple algorithms
- Work in real-time

⚠️ **Needs attention:**
- ROI configuration may need customization for your videos
- Model training required for DQN/GNN methods
- Consider fine-tuning for your specific intersection layout

🎯 **Bottom Line:** The infrastructure is complete and functional. With a trained model and proper ROI configuration, you can analyze video footage and get optimal green time recommendations!