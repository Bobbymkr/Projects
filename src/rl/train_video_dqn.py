"""
Video-based DQN Training for Adaptive Traffic Control

This script enables end-to-end training of DQN agents using real video feeds,
integrating the vision module (YOLOv8) with the RL environment for queue-based
traffic signal optimization.
"""

import json
import os
import argparse
import numpy as np
from tqdm import trange
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from src.env.video_env import VideoTrafficEnv, VideoEnvConfig
from src.rl.dqn_agent import DQNAgent, DQNConfig
from src.vision import VideoInputStream, VideoConfig, ROIManager, YOLOQueueEstimator, VideoSourceType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def create_video_config(video_source: str, fps: float = 30.0, width: int = 1280, height: int = 720) -> VideoConfig:
    """Create video configuration based on source type."""
    # Determine source type
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
        recording_enabled=False  # Disable recording during training
    )


def setup_vision_system(video_config: VideoConfig, roi_config_path: Optional[str] = None) -> tuple:
    """
    Initialize video input stream, ROI manager, and YOLO detector.
    
    Returns:
        Tuple of (video_stream, roi_manager, detector)
    """
    # Initialize video stream
    video_stream = VideoInputStream(video_config)
    
    # Initialize ROI manager
    roi_manager = ROIManager(roi_config_path)
    
    # Initialize YOLO detector
    detector = YOLOQueueEstimator(
        model_path=None,  # Will download YOLOv8n automatically
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
    """Create video-based traffic environment."""
    return VideoTrafficEnv(
        cfg=traffic_config,
        video_input=video_stream,
        roi_manager=roi_manager,
        detector=detector,
        sync_to_realtime=False  # Process frames as fast as available during training
    )


def train_video_dqn(traffic_config_path: str,
                   video_source: str,
                   episodes: int,
                   out_dir: str,
                   roi_config_path: Optional[str] = None,
                   video_fps: float = 30.0,
                   frame_width: int = 1280,
                   frame_height: int = 720,
                   episode_horizon_sec: int = 1800,
                   save_interval: int = 50):
    """
    Train DQN agent using video-based environment.
    
    Args:
        traffic_config_path: Path to traffic environment configuration
        video_source: Video source (webcam index, file path, or stream URL)
        episodes: Number of training episodes
        out_dir: Output directory for models and logs
        roi_config_path: Optional path to ROI configuration file
        video_fps: Target video processing FPS
        frame_width: Video frame width
        frame_height: Video frame height
        episode_horizon_sec: Episode duration in seconds
        save_interval: Save model every N episodes
    """
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Load traffic configuration
    traffic_config = load_config(traffic_config_path)
    traffic_config["episode_horizon_sec"] = episode_horizon_sec
    
    # Create video configuration
    video_config = create_video_config(video_source, video_fps, frame_width, frame_height)
    
    # Setup vision system
    logger.info("Initializing vision system...")
    video_stream, roi_manager, detector = setup_vision_system(video_config, roi_config_path)
    
    # Start video stream
    logger.info(f"Starting video stream from: {video_source}")
    if not video_stream.start():
        raise RuntimeError("Failed to start video stream")
    
    # Wait for video to stabilize
    time.sleep(2)
    
    try:
        # Create video-based environment
        logger.info("Creating video-based environment...")
        env = create_video_environment(traffic_config, video_stream, roi_manager, detector)
        
        # Initialize DQN agent
        logger.info("Initializing DQN agent...")
        dqn_config = DQNConfig(
            learning_rate=0.001,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            memory_size=10000,
            batch_size=32,
            target_update_freq=100,
            hidden_size=128
        )
        
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            cfg=dqn_config
        )
        
        # Training statistics
        rewards = []
        episode_lengths = []
        queue_stats = []
        
        logger.info(f"Starting training for {episodes} episodes...")
        
        # Training loop
        for episode in trange(episodes, desc="Training Episodes"):
            try:
                # Reset environment
                obs, info = env.reset()
                episode_reward = 0.0
                episode_steps = 0
                episode_queues = []
                
                terminated = truncated = False
                
                while not (terminated or truncated):
                    # Select action
                    action = agent.select_action(obs.astype(np.float32))
                    
                    # Execute action
                    next_obs, reward, terminated, truncated, step_info = env.step(action)
                    
                    # Store experience
                    agent.push(
                        obs.astype(np.float32),
                        action,
                        reward,
                        next_obs.astype(np.float32),
                        terminated or truncated
                    )
                    
                    # Train agent
                    loss = agent.train_step()
                    
                    # Update statistics
                    episode_reward += reward
                    episode_steps += 1
                    episode_queues.append(np.sum(next_obs))
                    
                    obs = next_obs
                
                # Episode completed
                rewards.append(episode_reward)
                episode_lengths.append(episode_steps)
                queue_stats.append(np.mean(episode_queues) if episode_queues else 0)
                
                # Log progress
                if (episode + 1) % 10 == 0:
                    recent_reward = np.mean(rewards[-10:])
                    recent_queue = np.mean(queue_stats[-10:])
                    current_epsilon = agent.epsilon
                    
                    logger.info(f"Episode {episode + 1}/{episodes}: "
                              f"Reward={recent_reward:.2f}, "
                              f"Avg Queue={recent_queue:.1f}, "
                              f"Epsilon={current_epsilon:.3f}, "
                              f"Steps={episode_steps}")
                
                # Save intermediate model
                if (episode + 1) % save_interval == 0:
                    model_path = os.path.join(out_dir, f'dqn_video_ep{episode + 1}.npz')
                    agent.save(model_path)
                    logger.info(f"Saved intermediate model: {model_path}")
                
            except Exception as e:
                logger.error(f"Error in episode {episode}: {e}")
                continue
        
        # Save final model and statistics
        final_model_path = os.path.join(out_dir, 'dqn_video_final.npz')
        agent.save(final_model_path)
        
        # Save training statistics
        stats = {
            'rewards': rewards,
            'episode_lengths': episode_lengths,
            'queue_stats': queue_stats,
            'config': {
                'traffic_config': traffic_config,
                'video_config': video_config.__dict__,
                'dqn_config': dqn_config.__dict__,
                'episodes': episodes
            }
        }
        
        stats_path = os.path.join(out_dir, 'training_stats.npz')
        np.savez(stats_path, **stats)
        
        # Performance summary
        logger.info(f"\n=== Training Complete ===")
        logger.info(f"Episodes: {episodes}")
        logger.info(f"Average reward: {np.mean(rewards):.2f}")
        logger.info(f"Best episode reward: {np.max(rewards):.2f}")
        logger.info(f"Average queue length: {np.mean(queue_stats):.1f}")
        logger.info(f"Model saved: {final_model_path}")
        logger.info(f"Stats saved: {stats_path}")
        
    finally:
        # Cleanup
        logger.info("Shutting down...")
        env.close()
        video_stream.stop()


def main():
    """Main training entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Train DQN agent with video-based traffic environment")
    
    # Required arguments
    parser.add_argument('--video_source', required=True, 
                       help='Video source: webcam index (0,1,2...), file path, or stream URL')
    
    # Optional arguments
    parser.add_argument('--config', default='configs/intersection.json',
                       help='Traffic environment configuration file')
    parser.add_argument('--roi_config', default=None,
                       help='ROI configuration file (optional)')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes')
    parser.add_argument('--out', default='runs/video_training',
                       help='Output directory for models and logs')
    parser.add_argument('--fps', type=float, default=15.0,
                       help='Video processing FPS (lower for training stability)')
    parser.add_argument('--width', type=int, default=640,
                       help='Video frame width')
    parser.add_argument('--height', type=int, default=480,
                       help='Video frame height')
    parser.add_argument('--episode_duration', type=int, default=300,
                       help='Episode duration in seconds (5 minutes default)')
    parser.add_argument('--save_interval', type=int, default=25,
                       help='Save model every N episodes')
    
    args = parser.parse_args()
    
    # Validate video source
    if not (args.video_source.isdigit() or 
            args.video_source.startswith(('http://', 'https://', 'rtsp://')) or
            Path(args.video_source).exists()):
        logger.error(f"Invalid video source: {args.video_source}")
        return
    
    # Start training
    train_video_dqn(
        traffic_config_path=args.config,
        video_source=args.video_source,
        episodes=args.episodes,
        out_dir=args.out,
        roi_config_path=args.roi_config,
        video_fps=args.fps,
        frame_width=args.width,
        frame_height=args.height,
        episode_horizon_sec=args.episode_duration,
        save_interval=args.save_interval
    )


if __name__ == "__main__":
    main()