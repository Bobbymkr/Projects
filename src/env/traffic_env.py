import numpy as np
import types
from typing import Tuple, Dict, Any
import gymnasium as gym
from gymnasium import spaces


class TrafficEnv(gym.Env):
    """
    Traffic environment simulating a single intersection with multiple lanes and two phases.

    Observation: queue lengths for each lane (num_lanes,)
    Action: discrete index selecting green duration = min_green + idx * green_step seconds for current phase

    The environment alternates phases automatically after serving chosen green time plus fixed yellow/all-red.
    Vehicles arrive via Poisson process per lane; departures during green follow a saturation flow rate.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.cfg = config
        self.num_lanes = int(self.cfg.get("num_lanes", 4))
        self.phase_lanes = self.cfg.get("phase_lanes", [[0, 1], [2, 3]])
        self.min_green = int(self.cfg.get("min_green", 5))
        self.max_green = int(self.cfg.get("max_green", 60))
        self.green_step = int(self.cfg.get("green_step", 5))
        self.yellow = int(self.cfg.get("cycle_yellow", 3))
        self.all_red = int(self.cfg.get("cycle_all_red", 1))
        self.queue_capacity = int(self.cfg.get("queue_capacity", 40))
        self.arrival_rates = np.array(self.cfg.get("arrival_rates", [0.3] * self.num_lanes), dtype=float)
        # Enhanced reward weights with multi-objective support
        default_reward_weights = {
            "queue": -0.4,
            "wait_penalty": -0.3,
            "throughput": 0.2,
            "efficiency": 0.1,
            "queue_reduction": 0.05,
            "safety": -1000.0
        }
        self.reward_weights = self.cfg.get("reward_weights", default_reward_weights)
        self.episode_horizon = int(self.cfg.get("episode_horizon", 3600))  # 1 hour default

        # Validate configuration
        self._validate_config()

        # Define action space: discrete green durations
        self.green_values = np.arange(self.min_green, self.max_green + 1, self.green_step, dtype=int)
        self.action_space = spaces.Discrete(len(self.green_values))

        # Observation space: queue length per lane (normalized)
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.num_lanes,), 
            dtype=np.float32
        )

        # Simulation parameters
        self.time = 0
        self.phase_index = 0  # 0 or 1 for which group of lanes gets green
        self.queues = np.zeros(self.num_lanes, dtype=int)
        self.wait_times = np.zeros(self.num_lanes, dtype=float)
        self.sat_flow_per_sec = 1.5  # vehicles per second per lane during green
        
        # Statistics tracking
        self.total_vehicles_processed = 0
        self.total_wait_time = 0.0
        self.max_queue_length = 0
        
        # Initialize reward tracking for enhanced reward function
        self._prev_queue_sum = 0.0
        self._accident_occurred = False

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.num_lanes <= 0:
            raise ValueError("num_lanes must be positive")
        if self.min_green >= self.max_green:
            raise ValueError("min_green must be less than max_green")
        if self.green_step <= 0:
            raise ValueError("green_step must be positive")
        if len(self.phase_lanes) != 2:
            raise ValueError("Must have exactly 2 phases")
        if len(self.arrival_rates) != self.num_lanes:
            raise ValueError("arrival_rates length must match num_lanes")
        # Arrival rates must be non-negative
        if np.any(self.arrival_rates < 0):
            raise ValueError("Arrival rates must be non-negative")

    def seed(self, seed: int | None = None):
        """Set random seed for reproducibility."""
        if seed is not None:
            np.random.seed(seed)

    def _poisson_arrivals(self, duration: int):
        """Simulate arrivals for each lane over 'duration' seconds using Poisson process."""
        lam = self.arrival_rates * duration
        arrivals = np.random.poisson(lam)
        self.queues = np.minimum(self.queues + arrivals, self.queue_capacity)

    def _departures_during_green(self, duration: int):
        """Simulate departures on lanes with green using a saturation flow rate."""
        can_depart_lanes = self.phase_lanes[self.phase_index]
        total_departed = 0
        
        for lane in can_depart_lanes:
            if lane < len(self.queues):
                depart = int(min(self.queues[lane], np.floor(self.sat_flow_per_sec * duration)))
                self.queues[lane] -= depart
                total_departed += depart
        
        self.total_vehicles_processed += total_departed

    def _advance_wait_times(self, duration: int):
        """Increase wait time proportional to queues over duration (approximate average)."""
        self.wait_times += self.queues * duration
        self.total_wait_time += np.sum(self.queues) * duration

    def _compute_reward(self, action: int = None, optimal_action: int = None) -> float:
        """
        Enhanced Multi-Objective Reward Function with proper normalization.
        
        Implements Phase 0.1 from OPTIMIZATION_ROADMAP.md:
        - Multi-scale, hierarchical rewards
        - Proper normalization for stability
        - Shaped rewards for learning guidance
        - Safety constraints with hard penalties
        
        Args:
            action: Current action taken (optional, for efficiency bonus)
            optimal_action: Optimal action for current state (optional, for efficiency bonus)
        
        Returns:
            Normalized reward value
        """
        # Primary Objectives (Weighted)
        # Queue penalty: normalized by capacity
        queue_penalty = self.reward_weights.get("queue", -0.4) * float(np.sum(self.queues)) / self.queue_capacity
        
        # Wait time penalty: normalized by episode horizon
        wait_penalty = self.reward_weights.get("wait_penalty", -0.3) * float(np.sum(self.wait_times)) / (self.episode_horizon * self.num_lanes)
        
        # Throughput bonus: reward for vehicles cleared
        throughput_bonus = self.reward_weights.get("throughput", 0.2) * self.total_vehicles_processed / max(1, self.time * self.num_lanes)
        
        # Efficiency bonus: reward for choosing good actions (if action info available)
        efficiency_bonus = 0.0
        if action is not None and optimal_action is not None:
            action_error = abs(action - optimal_action) / max(1, len(self.green_values) - 1)
            efficiency_bonus = self.reward_weights.get("efficiency", 0.1) * (1.0 - action_error)
        
        # Secondary Objectives (Shaping)
        # Queue reduction bonus: reward for reducing queues
        # This is computed per step, so we track previous queue state
        if not hasattr(self, '_prev_queue_sum'):
            self._prev_queue_sum = float(np.sum(self.queues))
        
        queue_reduction = self._prev_queue_sum - float(np.sum(self.queues))
        queue_reduction_bonus = self.reward_weights.get("queue_reduction", 0.05) * max(0.0, queue_reduction) / self.queue_capacity
        self._prev_queue_sum = float(np.sum(self.queues))
        
        # Safety Constraints (Hard penalties)
        # Penalty for maximum queue length exceeding capacity
        safety_penalty = 0.0
        if self.max_queue_length >= self.queue_capacity:
            safety_penalty = self.reward_weights.get("safety", -1000.0) * (self.max_queue_length - self.queue_capacity + 1) / self.queue_capacity
        
        # Penalty for accidents (if tracked in info)
        if hasattr(self, '_accident_occurred') and self._accident_occurred:
            safety_penalty += self.reward_weights.get("safety", -1000.0)
        
        # Combine all components
        raw_reward = (queue_penalty + wait_penalty + throughput_bonus + 
                     efficiency_bonus + queue_reduction_bonus + safety_penalty)
        
        # Normalization (Critical for stability)
        # Normalize by expected scale to prevent reward explosion
        normalized_reward = raw_reward / 100.0
        
        return normalized_reward

    def _normalize_observation(self, queues: np.ndarray) -> np.ndarray:
        """Normalize queue observations to [0, 1] range."""
        return np.clip(queues.astype(np.float32) / self.queue_capacity, 0.0, 1.0)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        if seed is not None:
            self.seed(seed)
        
        self.time = 0
        self.phase_index = 0
        self.queues = np.random.randint(0, self.queue_capacity // 4 + 1, size=self.num_lanes)
        self.wait_times = np.zeros(self.num_lanes, dtype=float)
        
        # Reset statistics
        self.total_vehicles_processed = 0
        self.total_wait_time = 0.0
        self.max_queue_length = np.max(self.queues)
        
        # Initialize reward tracking for enhanced reward function
        self._prev_queue_sum = float(np.sum(self.queues))
        self._accident_occurred = False
        
        obs = self._normalize_observation(self.queues)
        info = {
            "time": self.time, 
            "phase": self.phase_index,
            "queues": self.queues.copy(),
            "wait_times": self.wait_times.copy()
        }
        return obs, info

    def step(self, action: int):
        """Execute one step in the environment."""
        assert self.action_space.contains(action), f"Invalid action {action}"
        green_duration = int(self.green_values[action])

        # Arrivals during green and departures on the active phase
        self._poisson_arrivals(green_duration)
        self._departures_during_green(green_duration)
        self._advance_wait_times(green_duration)
        self.time += green_duration

        # Yellow + all red (no departures, only arrivals and waiting)
        intergreen = self.yellow + self.all_red
        self._poisson_arrivals(intergreen)
        self._advance_wait_times(intergreen)
        self.time += intergreen

        # Switch phase
        self.phase_index = 1 - self.phase_index

        # Update statistics
        self.max_queue_length = max(self.max_queue_length, int(np.max(self.queues)))

        obs = self._normalize_observation(self.queues)
        # Compute reward with action information (optimal_action not available, pass None)
        reward = self._compute_reward(action=action, optimal_action=None)
        terminated = False
        # Tests expect truncation when time reaches episode_horizon - 1 (inclusive)
        truncated = self.time >= max(1, self.episode_horizon - 1)

        info = {
            "time": self.time,
            "phase": self.phase_index,
            "queues": self.queues.copy(),
            "wait_times": self.wait_times.copy(),
            "total_vehicles_processed": self.total_vehicles_processed,
            "total_wait_time": self.total_wait_time,
            "max_queue_length": self.max_queue_length,
            "green_duration": green_duration,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the current state."""
        print(f"t={self.time}s phase={self.phase_index} queues={self.queues.tolist()} "
              f"processed={self.total_vehicles_processed} max_queue={self.max_queue_length}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get environment statistics."""
        return {
            "total_vehicles_processed": self.total_vehicles_processed,
            "total_wait_time": self.total_wait_time,
            "max_queue_length": self.max_queue_length,
            "average_wait_time": self.total_wait_time / max(1, self.total_vehicles_processed),
            "throughput_rate": self.total_vehicles_processed / max(1, self.time)
        }

    def simulate_episode(self, num_steps: int = 10, seed: int = 42, use_agent: bool = False):
        """Run a simulation for a given number of steps and print outputs.

        Args:
            num_steps: Number of steps to simulate.
            seed: Random seed for reproducibility.
            use_agent: If True, use DQN agent for action selection; else random.
        """
        obs, _ = self.reset(seed=seed)
        agent = None
        if use_agent:
            try:
                from src.rl.dqn_agent import DQNAgent, DQNConfig
                cfg = DQNConfig()
                agent = DQNAgent(
                    state_dim=self.observation_space.shape[0],
                    action_dim=self.action_space.n,
                    cfg=cfg
                )
                # For demo, we'll use a fresh agent; in practice, load a trained model
            except ImportError:
                print("Warning: DQN agent not available, using random actions")
                agent = None
        print(f"Initial: time={self.time}, phase={self.phase_index}, queues={obs.tolist()}")
        for step in range(num_steps):
            if agent:
                action = agent.select_action(obs.astype(np.float32))
            else:
                action = np.random.randint(0, self.action_space.n)  # Random action for demo
            obs, reward, _, _, info = self.step(action)
            green_duration = self.green_values[action]
            print(f"Step {step+1}: action={action} (green={green_duration}s), time={info['time']}, "
                  f"phase={info['phase']}, queues={obs.tolist()}, reward={reward:.3f}")


if __name__ == "__main__":
    config = {  # Sample config
        "num_lanes": 4,
        "phase_lanes": [[0, 1], [2, 3]],
        "min_green": 5,
        "max_green": 60,
        "green_step": 5,
        "cycle_yellow": 3,
        "cycle_all_red": 1,
        "arrival_rates": [0.3, 0.25, 0.35, 0.2],
        "queue_capacity": 40,
        "reward_weights": {"queue": -1.0, "wait_penalty": -0.1},
        "episode_horizon": 3600
    }
    env = TrafficEnv(config)
    print("Simulation with random actions:")
    env.simulate_episode(num_steps=5, use_agent=False)
    print("\nSimulation with DQN agent:")
    env.simulate_episode(num_steps=5, use_agent=True)
