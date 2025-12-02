"""
Visual Inference Pipeline

This module provides real-time inference with visualization overlays showing:
- Vehicle detections and tracking
- Queue estimations per ROI
- Current observations and predicted actions
- Performance metrics (FPS, processing time)
"""

import json
import argparse
import time
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Optional

# Video-based environment and vision stack
from src.env.video_env import VideoTrafficEnv
from src.rl.dqn_agent import DQNAgent, DQNConfig
from src.vision import VideoInputStream, VideoConfig, ROIManager, YOLOQueueEstimator, VideoSourceType
from src.vision.yolo_queue import process_frame_for_queues


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration dictionary from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def create_video_config(video_source: str, fps: float = 15.0, width: int = 640, height: int = 480) -> VideoConfig:
    """Create a VideoConfig based on the provided source string."""
    if video_source.isdigit():
        source_type = VideoSourceType.WEBCAM
        source = int(video_source)
    elif video_source.startswith(('http://', 'https://', 'rtsp://')):
        source_type = VideoSourceType.RTSP if video_source.startswith('rtsp://') else VideoSourceType.HTTP
        source = video_source
    elif Path(video_source).exists():
        source_type = VideoSourceType.FILE
        source = video_source
    else:
        raise ValueError(f"Invalid video source: {video_source}")

    return VideoConfig(
        source=source,
        source_type=source_type,
        target_fps=fps,
        frame_width=width,
        frame_height=height,
        buffer_size=30,
        auto_resize=True,
        recording_enabled=False
    )


def setup_vision_system(video_config: VideoConfig, roi_config_path: Optional[str] = None):
    """Initialize the video input stream, ROI manager, and YOLO queue estimator."""
    video_stream = VideoInputStream(video_config)
    roi_manager = ROIManager(roi_config_path)
    detector = YOLOQueueEstimator(
        model_path=None,  # auto-download YOLOv8n if not present
        rois=roi_manager.rois,
        confidence_threshold=0.4,
        nms_threshold=0.5,
        max_tracking_distance=75.0,
        max_missing_frames=15,
        queue_smoothing_window=3
    )
    return video_stream, roi_manager, detector


def create_video_environment(traffic_config: Dict[str, Any],
                             video_stream: VideoInputStream,
                             roi_manager: ROIManager,
                             detector: YOLOQueueEstimator) -> VideoTrafficEnv:
    """Create a VideoTrafficEnv instance wired to the given vision components."""
    return VideoTrafficEnv(
        cfg=traffic_config,
        video_input=video_stream,
        roi_manager=roi_manager,
        detector=detector,
        sync_to_realtime=True
    )


def draw_inference_overlay(frame: np.ndarray, 
                          queues: Dict[int, int],
                          observation: np.ndarray,
                          action: int,
                          green_time: int,
                          processing_time_ms: float,
                          fps: float) -> np.ndarray:
    """Draw inference information overlay on the frame.
    
    Args:
        frame: Input frame
        queues: Queue counts per lane ID
        observation: Current environment observation
        action: Selected action index
        green_time: Recommended green time in seconds
        processing_time_ms: Processing time in milliseconds
        fps: Current FPS
        
    Returns:
        Frame with overlay information
    """
    overlay_frame = frame.copy()
    h, w = overlay_frame.shape[:2]
    
    # Semi-transparent overlay background
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Info panel background (top-left)
    cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay_frame, 0.8, overlay, 0.2, 0, overlay_frame)
    
    # Title
    cv2.putText(overlay_frame, "Traffic Signal AI", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Performance metrics
    y_offset = 65
    cv2.putText(overlay_frame, f"FPS: {fps:.1f}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_offset += 25
    cv2.putText(overlay_frame, f"Processing: {processing_time_ms:.1f}ms", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Queue information
    y_offset += 35
    cv2.putText(overlay_frame, "Queue Lengths:", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y_offset += 25
    
    for lane_id, queue_length in queues.items():
        color = (0, 0, 255) if queue_length > 5 else (0, 255, 0)  # Red if long queue
        cv2.putText(overlay_frame, f"  Lane {lane_id}: {queue_length} vehicles", 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += 20
    
    # Recommendation
    y_offset += 15
    cv2.putText(overlay_frame, f"Recommended Green Time:", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    y_offset += 25
    cv2.putText(overlay_frame, f"  {green_time} seconds (Action {action})", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Observation values (bottom-right)
    if len(observation) > 0:
        obs_panel_x = w - 300
        obs_panel_y = h - 150
        
        # Background for observation panel
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(overlay, (obs_panel_x - 10, obs_panel_y - 40), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay_frame, 0.8, overlay, 0.2, 0, overlay_frame)
        
        cv2.putText(overlay_frame, "Observation Vector:", (obs_panel_x, obs_panel_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Show first few observation values
        y_obs = obs_panel_y + 15
        for i, value in enumerate(observation[:6]):  # Show first 6 values
            cv2.putText(overlay_frame, f"  [{i}]: {value:.3f}", (obs_panel_x, y_obs),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_obs += 18
            
        if len(observation) > 6:
            cv2.putText(overlay_frame, f"  ... ({len(observation)} total)", (obs_panel_x, y_obs),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    
    return overlay_frame


def run_visual_inference(cfg_path: str,
                        model_path: str,
                        video_source: str,
                        roi_config: Optional[str] = None,
                        fps: float = 15.0,
                        width: int = 640,
                        height: int = 480,
                        warmup_sec: int = 2,
                        continuous: bool = False):
    """Run visual inference with real-time overlay showing detections and predictions.
    
    Args:
        cfg_path: Path to traffic configuration
        model_path: Path to trained DQN model
        video_source: Video input (webcam, file, or stream)
        roi_config: Optional ROI configuration file
        fps: Target processing FPS
        width: Frame width
        height: Frame height
        warmup_sec: Warm-up time before processing
        continuous: If True, run continuously; if False, process one observation and exit
    """
    traffic_config = load_config(cfg_path)

    # Build video config and vision components
    video_config = create_video_config(video_source, fps=fps, width=width, height=height)
    video_stream, roi_manager, detector = setup_vision_system(video_config, roi_config)

    # Start stream and wait briefly for frames to flow
    if not video_stream.start():
        raise RuntimeError("Failed to start video stream")
    
    print(f"Starting visual inference...")
    print(f"Video source: {video_source}")
    print(f"Model: {model_path}")
    print(f"ROI config: {roi_config or 'Auto-generated'}")
    print()
    
    if continuous:
        print("Running continuous inference. Press 'q' to quit, 'p' to pause/resume")
    else:
        print("Processing single observation. Press any key to continue after result...")
    
    time.sleep(max(0, warmup_sec))

    env = None
    try:
        env = create_video_environment(traffic_config, video_stream, roi_manager, detector)

        # Initialize agent and load model
        agent = DQNAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, cfg=DQNConfig())
        agent.load(model_path)

        # Performance tracking
        fps_counter = []
        paused = False
        
        # Create display window
        window_name = "Traffic AI Visual Inference"
        cv2.namedWindow(window_name, cv2.WINDOW_RESIZABLE)
        
        if not continuous:
            # Single observation mode
            obs, _ = env.reset()
            frame = video_stream.get_frame()
            
            if frame is not None:
                start_time = time.time()
                action = agent.select_action(obs.astype(np.float32), evaluate=True)
                processing_time = (time.time() - start_time) * 1000
                
                green_sec = env.green_values[action]
                
                # Get current queue information
                queue_result = process_frame_for_queues(detector, frame, roi_manager.rois, debug=True)
                queues = queue_result.get('queues', {})
                
                # Create visualization
                viz_frame = draw_inference_overlay(
                    queue_result.get('annotated', frame),
                    queues, obs, action, int(green_sec), processing_time, fps
                )
                
                cv2.imshow(window_name, viz_frame)
                print(f"Recommended green time: {int(green_sec)} seconds")
                print(f"Queue lengths: {queues}")
                print("Press any key to exit...")
                cv2.waitKey(0)
        else:
            # Continuous mode
            obs, _ = env.reset()
            last_inference_time = 0
            inference_interval = 1.0  # Update inference every 1 second
            
            while True:
                current_time = time.time()
                frame_start = current_time
                
                frame = video_stream.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Run inference periodically
                if current_time - last_inference_time > inference_interval and not paused:
                    obs, _ = env.reset()  # Get fresh observation
                    action = agent.select_action(obs.astype(np.float32), evaluate=True)
                    green_sec = env.green_values[action]
                    last_inference_time = current_time
                    
                    print(f"Updated recommendation: {int(green_sec)}s (Action {action})")
                
                # Get current queue information for visualization
                queue_result = process_frame_for_queues(detector, frame, roi_manager.rois, debug=True)
                queues = queue_result.get('queues', {})
                
                # Calculate processing time and FPS
                processing_time = (time.time() - frame_start) * 1000
                fps_counter.append(1.0 / max(time.time() - frame_start, 0.001))
                if len(fps_counter) > 30:
                    fps_counter.pop(0)
                current_fps = np.mean(fps_counter)
                
                # Create visualization
                try:
                    viz_frame = draw_inference_overlay(
                        queue_result.get('annotated', frame),
                        queues, obs, action if 'action' in locals() else 0, 
                        int(green_sec) if 'green_sec' in locals() else 0,
                        processing_time, current_fps
                    )
                    
                    # Add pause indicator if paused
                    if paused:
                        cv2.putText(viz_frame, "PAUSED", (viz_frame.shape[1]//2 - 50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    
                    cv2.imshow(window_name, viz_frame)
                except Exception as e:
                    print("Visualization error due to an internal error.")
                    cv2.imshow(window_name, frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("Paused" if paused else "Resumed")

    finally:
        # Cleanup
        if env is not None:
            env.close()
        try:
            video_stream.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual DQN inference with real-time overlay")
    parser.add_argument('--config', default='configs/intersection.json', help='Traffic config JSON')
    parser.add_argument('--model', required=True, help='Path to trained DQN .npz model')
    parser.add_argument('--video_source', required=True, help='Webcam index (e.g., 0), file path, or stream URL')
    parser.add_argument('--roi_config', default=None, help='Optional ROI config file')
    parser.add_argument('--fps', type=float, default=15.0, help='Target processing FPS')
    parser.add_argument('--width', type=int, default=640, help='Frame width')
    parser.add_argument('--height', type=int, default=480, help='Frame height')
    parser.add_argument('--warmup', type=int, default=2, help='Warm-up seconds before processing')
    parser.add_argument('--continuous', action='store_true', help='Run continuous inference (default: single shot)')

    args = parser.parse_args()

    run_visual_inference(
        cfg_path=args.config,
        model_path=args.model,
        video_source=args.video_source,
        roi_config=args.roi_config,
        fps=args.fps,
        width=args.width,
        height=args.height,
        warmup_sec=args.warmup,
        continuous=args.continuous
    )