"""
Video-based Traffic Environment Wrapper

This environment wraps the simulated TrafficEnv API to consume observations from
real video via the vision subsystem (VideoInputStream + YOLOQueueEstimator + ROIManager).

Key design points:
- Observation: queue lengths per lane derived from vision (median over an interval)
- Action space: identical to TrafficEnv (discrete green durations)
- Step timing: step(action) consumes green_duration seconds of frames and an intergreen
  gap (yellow + all-red). The next observation is computed from the last segment.
- Reward: negative weighted sum of queue lengths and cumulative waiting approximation.
- Episode: time-bounded by a configured horizon (seconds). Termination is time-based.

Limitations:
- Without physical control of signal or a controllable simulator synchronized to the video,
  this environment approximates the reward from observed queues and their temporal aggregation.
- For offline videos (not live), processing runs as fast as frames are provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import time
import numpy as np

from src.vision import VideoInputStream, VideoConfig, ROIManager, ROIConfig, YOLOQueueEstimator
from src.env.traffic_env import Discrete, Box


@dataclass
class VideoEnvConfig:
    """Configuration for the video-based environment."""
    # Traffic control parameters (kept consistent with TrafficEnv)
    num_lanes: int = 4
    phase_lanes: List[List[int]] = None  # defaults in __post_init__
    min_green: int = 5
    max_green: int = 60
    green_step: int = 5
    cycle_yellow: int = 3
    cycle_all_red: int = 1

    # Reward weights
    reward_weights: Dict[str, float] = None  # defaults in __post_init__

    # Episode control
    episode_horizon_sec: int = 1800  # default 30 minutes

    # Vision processing
    min_stationary_seconds: float = 2.0
    observation_aggregate: str = "median"  # median or mean

    def __post_init__(self):
        if self.phase_lanes is None:
            self.phase_lanes = [[0, 1], [2, 3]]
        if self.reward_weights is None:
            self.reward_weights = {"queue": -1.0, "wait_penalty": -0.1}


class VideoTrafficEnv:
    """
    Environment interface driven by video-based queue estimation.

    Observation space: queue lengths per lane (int32 vector of size num_lanes)
    Action space: discrete green durations (min_green + k*green_step)
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        video_input: VideoInputStream,
        roi_manager: ROIManager,
        detector: YOLOQueueEstimator,
        sync_to_realtime: bool = True,
    ):
        """
        Initialize the video-based environment.

        Args:
            cfg: Dict of traffic params, see intersection.json compatible fields
            video_input: Initialized VideoInputStream
            roi_manager: ROIManager with ROIs defined (or empty to auto-init)
            detector: Initialized YOLOQueueEstimator
            sync_to_realtime: If True, waits wall-clock time for green_duration; otherwise consumes frames as available
        """
        self.cfg = VideoEnvConfig(
            num_lanes=int(cfg.get("num_lanes", 4)),
            phase_lanes=cfg.get("phase_lanes", [[0, 1], [2, 3]]),
            min_green=int(cfg.get("min_green", 5)),
            max_green=int(cfg.get("max_green", 60)),
            green_step=int(cfg.get("green_step", 5)),
            cycle_yellow=int(cfg.get("cycle_yellow", 3)),
            cycle_all_red=int(cfg.get("cycle_all_red", 1)),
            reward_weights=cfg.get("reward_weights", {"queue": -1.0, "wait_penalty": -0.1}),
            episode_horizon_sec=int(cfg.get("episode_horizon_sec", 1800)),
            min_stationary_seconds=float(cfg.get("min_stationary_seconds", 2.0)),
            observation_aggregate=str(cfg.get("observation_aggregate", "median")),
        )

        # Action and observation spaces compatible with TrafficEnv
        self.green_values = np.arange(self.cfg.min_green, self.cfg.max_green + 1, self.cfg.green_step, dtype=int)
        self.action_space = Discrete(len(self.green_values))
        self.observation_space = Box(low=0, high=cfg.get("queue_capacity", 100), shape=(self.cfg.num_lanes,), dtype=np.int32)

        self.video = video_input
        self.roi_manager = roi_manager
        self.detector = detector

        self.time = 0
        self.phase_index = 0
        self._episode_start_wall_time: Optional[float] = None
        self._wait_integral = np.zeros(self.cfg.num_lanes, dtype=float)

        # Internal buffer used to aggregate observations during a step
        self._queue_trace: List[np.ndarray] = []

    def _ensure_rois(self):
        """Ensure ROIs exist; if empty, create defaults based on current frame size."""
        if len(self.roi_manager.rois) == 0:
            # Peek a frame to get frame size
            fd = None
            for _ in range(50):
                fd = self.video.get_latest_frame()
                if fd is not None:
                    break
                time.sleep(0.02)
            if fd is None:
                raise RuntimeError("Cannot initialize ROIs without at least one frame")
            h, w = fd.frame.shape[:2]
            self.roi_manager.create_default_rois(w, h, num_lanes=self.cfg.num_lanes)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment and return the initial observation derived from video.
        """
        del seed  # not applicable
        if not self.video.running:
            ok = self.video.start()
            if not ok:
                raise RuntimeError("Video input failed to start")

        self._ensure_rois()

        self.time = 0
        self.phase_index = 0
        self._wait_integral[:] = 0.0
        self._queue_trace.clear()
        self._episode_start_wall_time = time.time()

        # Warm-up: gather a short window to compute initial observation
        obs = self._collect_observation(window_sec=2)
        info = {"time": self.time, "phase": self.phase_index}
        return obs.astype(np.int32), info

    def _collect_observation(self, window_sec: int) -> np.ndarray:
        """Collect queue estimates over a time window and aggregate to a single observation vector."""
        end_time = time.time() + window_sec
        queues_accum: List[np.ndarray] = []
        while time.time() < end_time:
            fd = self.video.get_latest_frame()
            if fd is None:
                time.sleep(0.01)
                continue
            # Estimate queues using YOLOQueueEstimator (returns list[int] per ROI)
            queue_list = self.detector.estimate_queues(fd.frame)
            # Ensure vector length matches number of lanes
            q = np.zeros(self.cfg.num_lanes, dtype=int)
            if len(queue_list) > 0:
                # Pad or trim to fit num_lanes
                arr = np.array(queue_list, dtype=int)
                if arr.shape[0] < self.cfg.num_lanes:
                    arr = np.pad(arr, (0, self.cfg.num_lanes - arr.shape[0]))
                elif arr.shape[0] > self.cfg.num_lanes:
                    arr = arr[: self.cfg.num_lanes]
                q = arr
            queues_accum.append(q)
            # approximate wait integral increment (sum of queues for this slice)
            self._wait_integral += q
            self._queue_trace.append(q)
            if self.cfg.observation_aggregate == "median":
                pass
        if not queues_accum:
            return np.zeros(self.cfg.num_lanes, dtype=int)
        stacks = np.stack(queues_accum, axis=0)
        if self.cfg.observation_aggregate == "mean":
            agg = stacks.mean(axis=0)
        else:
            agg = np.median(stacks, axis=0)
        return np.rint(agg).astype(int)

    def step(self, action: int):
        """
        Execute an action: wait for green_duration seconds (consuming frames), then intergreen,
        compute next observation and reward.
        """
        assert self.action_space.contains(action), "Invalid action"
        green_duration = int(self.green_values[action])

        # Consume frames during green interval
        green_obs = self._collect_observation(window_sec=green_duration)
        self.time += green_duration

        # Consume frames during intergreen (no departures expected, but queues still change)
        intergreen = self.cfg.cycle_yellow + self.cfg.cycle_all_red
        if intergreen > 0:
            _ = self._collect_observation(window_sec=intergreen)
            self.time += intergreen

        # Switch phase
        self.phase_index = 1 - self.phase_index

        # Observation is latest aggregated queues
        obs = green_obs.astype(np.int32)

        # Reward approximation mirrors TrafficEnv sign convention
        queue_cost = self.cfg.reward_weights.get("queue", -1.0) * float(np.sum(obs))
        wait_cost = self.cfg.reward_weights.get("wait_penalty", -0.1) * float(np.sum(self._wait_integral))
        reward = queue_cost + wait_cost

        # Episode time-based termination
        elapsed = time.time() - (self._episode_start_wall_time or time.time())
        truncated = elapsed >= self.cfg.episode_horizon_sec
        terminated = False

        info = {
            "time": self.time,
            "phase": self.phase_index,
            "green": green_duration,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        """Cleanup resources."""
        try:
            self.video.stop()
        except Exception:
            pass