import json
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Simulation (fallback) environment
from src.env.traffic_env import TrafficEnv
from src.env.sumo_env import SumoEnv
from src.env.marl_env import MarlEnv

# Video-based environment and vision stack
from src.env.video_env import VideoTrafficEnv
from src.rl.dqn_agent import DQNAgent, DQNConfig
from src.vision import VideoInputStream, VideoConfig, ROIManager, YOLOQueueEstimator, VideoSourceType


# ------------------------------
# Utility helpers
# ------------------------------

def load_config(path: str) -> Dict[str, Any]:
    """Load configuration dictionary from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def make_env(cfg: Dict[str, Any], use_sumo: bool = False, use_marl: bool = False):
    if use_marl:
        return MarlEnv()
    if use_sumo:
        return SumoEnv(cfg)
    return TrafficEnv(cfg)
    if use_sumo:
        return SumoEnv(cfg)
    return TrafficEnv(cfg)


def create_video_config(video_source: str, fps: float = 15.0, width: int = 640, height: int = 480) -> VideoConfig:
    """Create a VideoConfig based on the provided source string.

    - If the source is digits, treat as webcam index
    - If the source starts with http(s) or rtsp, treat as network stream
    - If it's an existing file path, treat as local file
    """
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
    """Initialize the video input stream, ROI manager, and YOLO queue estimator.

    Returns a tuple of (video_stream, roi_manager, detector).
    """
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


# ------------------------------
# Inference entry points
# ------------------------------

def run_inference(cfg_path: str, model_dir: str, use_sumo: bool = False, use_marl: bool = False):
    """Run inference on the simulated TrafficEnv (or SumoEnv if flagged) and print the recommended green time.
    This is the original inference path for the synthetic environment.
    """
    cfg = load_config(cfg_path)
    env = make_env(cfg, use_sumo, use_marl)
    if use_marl:
        for tl in env.intersections:
            forecaster_path = os.path.join(model_dir, f'forecaster_{tl}.h5')
            if os.path.exists(forecaster_path):
                env.forecaster[tl].load(forecaster_path)
        num_agents = env.num_agents
        agents = []
        for i in range(num_agents):
            agent = DQNAgent(state_dim=env.observation_space[i].shape[0], action_dim=env.action_space[i].n, cfg=DQNConfig())
            agent_path = os.path.join(model_dir, f'dqn_traffic_agent_{i}.npz')
            agent.load(agent_path)
            agents.append(agent)
        states = env.reset()
        total_rewards = [0.0] * num_agents
        done = False
        while not done:
            actions = [ag.select_action(st.astype(np.float32), evaluate=True) for ag, st in zip(agents, states)]
            next_states, rews, dones, _ = env.step(actions)
            for i in range(num_agents):
                total_rewards[i] += rews[i]
            states = next_states
            done = any(dones)
        avg_reward = np.mean(total_rewards)
        print(f"Average reward over the episode: {avg_reward:.2f}")
        return
    agent = DQNAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, cfg=DQNConfig())
    agent.load(model_path)

    obs, _ = env.reset()
    action = agent.select_action(obs.astype(np.float32), evaluate=True)
    green_sec = env.green_values[action]
    print(f"Recommended green time (seconds): {int(green_sec)}")


def run_video_inference(cfg_path: str,
                        model_path: str,
                        video_source: str,
                        roi_config: Optional[str] = None,
                        fps: float = 15.0,
                        width: int = 640,
                        height: int = 480,
                        warmup_sec: int = 2):
    """Run inference using real-time video to generate the current observation and
    print the recommended green time in seconds.

    Steps:
    1) Initialize the vision stack (VideoInputStream, ROIManager, YOLOQueueEstimator)
    2) Start the video, stabilize briefly, and build a VideoTrafficEnv
    3) Reset the env to collect an initial observation window from video
    4) Load the DQN model and choose the best action (evaluate=True)
    5) Print the corresponding green duration in seconds
    """
    traffic_config = load_config(cfg_path)

    # Build video config and vision components
    video_config = create_video_config(video_source, fps=fps, width=width, height=height)
    video_stream, roi_manager, detector = setup_vision_system(video_config, roi_config)

    # Start stream and wait briefly for frames to flow
    if not video_stream.start():
        raise RuntimeError("Failed to start video stream")
    time.sleep(max(0, warmup_sec))

    env = None
    try:
        env = create_video_environment(traffic_config, video_stream, roi_manager, detector)

        # Initialize agent and load model
        agent = DQNAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, cfg=DQNConfig())
        agent.load(model_path)

        # Gather initial observation from video
        obs, _ = env.reset()

        # Select action deterministically
        action = agent.select_action(obs.astype(np.float32), evaluate=True)
        green_sec = env.green_values[action]
        print(f"Recommended green time (seconds): {int(green_sec)}")
    finally:
        # Cleanup
        if env is not None:
            env.close()
        try:
            video_stream.stop()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN inference (simulated or real-time video)")
    subparsers = parser.add_subparsers(dest='mode', required=False)

    # Simulated inference (default)
    sim_parser = subparsers.add_parser('sim', help='Run inference on simulated TrafficEnv (default)')
    sim_parser.add_argument('--config', default='configs/intersection.json', help='Traffic config JSON')
    sim_parser.add_argument('--model', required=True, help='Path to trained DQN .npz model or directory for MARL')
    sim_parser.add_argument('--marl', action='store_true', help='Use MARL environment')
    sim_parser.add_argument('--use_sumo', action='store_true', help='Use SUMO-based environment')

    # Video-based inference
    vid_parser = subparsers.add_parser('video', help='Run inference using real-time video')
    vid_parser.add_argument('--config', default='configs/intersection.json', help='Traffic config JSON')
    vid_parser.add_argument('--model', required=True, help='Path to trained DQN .npz model')
    vid_parser.add_argument('--video_source', required=True, help='Webcam index (e.g., 0), file path, or stream URL')
    vid_parser.add_argument('--roi_config', default=None, help='Optional ROI config file')
    vid_parser.add_argument('--fps', type=float, default=15.0, help='Target processing FPS')
    vid_parser.add_argument('--width', type=int, default=640, help='Frame width')
    vid_parser.add_argument('--height', type=int, default=480, help='Frame height')
    vid_parser.add_argument('--warmup', type=int, default=2, help='Warm-up seconds before observation')

    args = parser.parse_args()

    # Default to sim mode if none provided for backward compatibility
    mode = args.mode or 'sim'

    if mode == 'video':
        run_video_inference(
            cfg_path=args.config,
            model_path=args.model,
            video_source=args.video_source,
            roi_config=args.roi_config,
            fps=args.fps,
            width=args.width,
            height=args.height,
            warmup_sec=args.warmup,
        )
    else:
        run_inference(args.config, args.model, args.use_sumo, args.marl)
