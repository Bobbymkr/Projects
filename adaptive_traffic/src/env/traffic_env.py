import numpy as np
from typing import Tuple, Dict, Any
from src.rl.dqn_agent import DQNAgent, DQNConfig


class Discrete:
    """Minimal stand-in for Gymnasium's Discrete space."""
    def __init__(self, n: int):
        self.n = int(n)

    def contains(self, x: int) -> bool:
        return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n


class Box:
    """Minimal stand-in for Gymnasium's Box space."""
    def __init__(self, low, high, shape, dtype):
        self.low = low
        self.high = high
        self.shape = tuple(shape)
        self.dtype = dtype


class TrafficEnv:
    """
    Traffic environment simulating a single intersection with multiple lanes and two phases.

    Observation: queue lengths for each lane (num_lanes,)
    Action: discrete index selecting green duration = min_green + idx * green_step seconds for current phase

    The environment alternates phases automatically after serving chosen green time plus fixed yellow/all-red.
    Vehicles arrive via Poisson process per lane; departures during green follow a saturation flow rate.
    """

    def __init__(self, config: Dict[str, Any]):
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
        self.reward_weights = self.cfg.get("reward_weights", {"queue": -1.0, "wait_penalty": -0.1})

        # Define action space: discrete green durations
        self.green_values = np.arange(self.min_green, self.max_green + 1, self.green_step, dtype=int)
        self.action_space = Discrete(len(self.green_values))

        # Observation space: queue length per lane
        self.observation_space = Box(low=0, high=self.queue_capacity, shape=(self.num_lanes,), dtype=np.int32)

        # Simulation parameters
        self.time = 0
        self.phase_index = 0  # 0 or 1 for which group of lanes gets green
        self.queues = np.zeros(self.num_lanes, dtype=int)
        self.wait_times = np.zeros(self.num_lanes, dtype=float)
        self.sat_flow_per_sec = 1.5  # vehicles per second per lane during green

    def seed(self, seed: int | None = None):
        np.random.seed(seed)

    def _poisson_arrivals(self, duration: int):
        """Simulate arrivals for each lane over 'duration' seconds using Poisson process."""
        lam = self.arrival_rates * duration
        arrivals = np.random.poisson(lam)
        self.queues = np.minimum(self.queues + arrivals, self.queue_capacity)

    def _departures_during_green(self, duration: int):
        """Simulate departures on lanes with green using a saturation flow rate."""
        can_depart_lanes = self.phase_lanes[self.phase_index]
        for lane in can_depart_lanes:
            depart = int(min(self.queues[lane], np.floor(self.sat_flow_per_sec * duration)))
            self.queues[lane] -= depart

    def _advance_wait_times(self, duration: int):
        """Increase wait time proportional to queues over duration (approximate average)."""
        self.wait_times += self.queues * duration

    def _compute_reward(self) -> float:
        queue_cost = self.reward_weights.get("queue", -1.0) * float(np.sum(self.queues))
        wait_cost = self.reward_weights.get("wait_penalty", -0.1) * float(np.sum(self.wait_times))
        return queue_cost + wait_cost

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed(seed)
        self.time = 0
        self.phase_index = 0
        self.queues = np.random.randint(0, self.queue_capacity // 4 + 1, size=self.num_lanes)
        self.wait_times = np.zeros(self.num_lanes, dtype=float)
        obs = self.queues.copy().astype(np.int32)
        info = {"time": self.time, "phase": self.phase_index}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action"
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

        obs = self.queues.copy().astype(np.int32)
        reward = self._compute_reward()
        terminated = False
        truncated = self.time >= 3600  # 1-hour episode
        info = {"time": self.time, "phase": self.phase_index}
        return obs, reward, terminated, truncated, info

    def render(self):
        print(f"t={self.time}s phase={self.phase_index} queues={self.queues.tolist()}")

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
            agent = DQNAgent(
                state_dim=self.observation_space.shape[0],
                action_dim=self.action_space.n,
                cfg=DQNConfig()
            )
            # For demo, we'll use a fresh agent; in practice, load a trained model
        print(f"Initial: time={self.time}, phase={self.phase_index}, queues={obs.tolist()}")
        for step in range(num_steps):
            if agent:
                action = agent.select_action(obs.astype(np.float32))
            else:
                action = np.random.randint(0, self.action_space.n)  # Random action for demo
            obs, reward, _, _, info = self.step(action)
            green_duration = self.green_values[action]
            print(f"Step {step+1}: action={action} (green={green_duration}s), time={info['time']}, phase={info['phase']}, queues={obs.tolist()}, reward={reward}")

if __name__ == "__main__":
    config = {  # Sample config
        "num_lanes": 4,
        "phase_lanes": [[0, 1], [2, 3]],
        "min_green": 5,
        "max_green": 60,
        "green_step": 5,
        "cycle_yellow": 3,
        "cycle_all_red": 1,
        "arrival_rates": [0.3, 0.3, 0.3, 0.3],
        "queue_capacity": 40,
        "reward_weights": {"queue": -1.0, "wait_penalty": -0.1}
    }
    env = TrafficEnv(config)
    print("Simulation with random actions:")
    env.simulate_episode(num_steps=5, use_agent=False)
    print("\nSimulation with DQN agent:")
    env.simulate_episode(num_steps=5, use_agent=True)
