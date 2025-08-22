import numpy as np
import traci
import traci.constants as tc
from typing import Tuple, Dict, Any
import gymnasium as gym
from gymnasium import spaces

class SumoEnv(gym.Env):
    """SUMO-based traffic environment for a single intersection."""
    metadata = {'render_modes': ['human']}

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.cfg = config
        self.sumocfg = self.cfg.get("sumocfg", "configs/sumo.sumocfg")
        self.num_directions = 4  # North, South, East, West
        self.incoming_edges = ["north_in", "south_in", "east_in", "west_in"]
        self.tls_id = "center"  # Traffic light ID from network
        self.min_green = int(self.cfg.get("min_green", 5))
        self.max_green = int(self.cfg.get("max_green", 60))
        self.green_step = int(self.cfg.get("green_step", 5))
        self.yellow = int(self.cfg.get("cycle_yellow", 3))
        self.all_red = int(self.cfg.get("cycle_all_red", 1))
        self.queue_capacity = int(self.cfg.get("queue_capacity", 40))
        self.reward_weights = self.cfg.get("reward_weights", {"queue": -1.0, "wait_penalty": -0.1})

        # Action space
        self.green_values = np.arange(self.min_green, self.max_green + 1, self.green_step, dtype=int)
        self.action_space = spaces.Discrete(len(self.green_values))

        # Observation space: queue per direction
        self.observation_space = spaces.Box(low=0, high=self.queue_capacity * 4, shape=(self.num_directions,), dtype=np.float32)  # Change to float32

        # Phases: 0 for NS green, 1 for EW green (simplified)
        self.phase_defs = [
            "GGGGGggrrrrrrrGGGGGggrrrrrrr",  # NS green from net.xml
            "rrrrrrrGGGGGggrrrrrrrGGGGGgg"   # EW green from net.xml
        ]
        self.current_phase = 0

    def _get_queues(self):
        queues = np.zeros(self.num_directions, dtype=int)
        for i, edge in enumerate(self.incoming_edges):
            queues[i] = traci.edge.getLastStepHaltingNumber(edge)  # Better for queue length
        return np.minimum(queues, self.queue_capacity)

    def _get_wait_times(self):
        wait_times = np.zeros(self.num_directions, dtype=float)
        for i, edge in enumerate(self.incoming_edges):
            lanes = [f"{edge}_{i}" for i in range(4)]  # Assuming 4 lanes per edge
            edge_wait = sum(traci.lane.getWaitingTime(lane) for lane in lanes)
            wait_times[i] = edge_wait / max(1, len(lanes))  # Average per lane
        return wait_times

    def _compute_reward(self, queues, wait_times) -> float:
        queue_cost = self.reward_weights.get("queue", -1.0) * float(np.sum(queues))
        wait_cost = self.reward_weights.get("wait_penalty", -0.1) * float(np.sum(wait_times))
        return queue_cost + wait_cost

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        if traci.isLoaded():
            traci.close()
        traci.start(["sumo", "-c", self.sumocfg])
        traci.trafficlight.subscribe(self.tls_id, (tc.TL_RED_YELLOW_GREEN_STATE,))
        self.current_phase = 0
        self._set_phase(self.current_phase, self.min_green)  # Initial setup
        traci.simulationStep()
        queues = self._get_queues()
        info = {"phase": self.current_phase}
        return queues.astype(np.float32), info  # Change to float32 for SB3 compatibility

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action"
        green_duration = int(self.green_values[action])

        # Set green for current phase
        self._set_phase(self.current_phase, green_duration)
        for _ in range(green_duration):
            traci.simulationStep()

        # Yellow phase (simplified: set to yellow for all)
        traci.trafficlight.setRedYellowGreenState(self.tls_id, "yyyyyyyrrrrrrryyyyyyyrrrrrrr")
        for _ in range(self.yellow):
            traci.simulationStep()

        # All red
        
        traci.trafficlight.setRedYellowGreenState(self.tls_id, "rrrrrrrrrrrrrrrrrrrrrrrrrrrr")
        for _ in range(self.all_red):
            traci.simulationStep()

        # Switch phase
        self.current_phase = 1 - self.current_phase

        queues = self._get_queues()
        wait_times = self._get_wait_times()
        reward = self._compute_reward(queues, wait_times)
        terminated = False
        truncated = traci.simulation.getTime() >= 3600
        info = {"phase": self.current_phase}
        return queues.astype(np.int32), reward, terminated, truncated, info

    def _set_phase(self, phase_idx: int, duration: int):
        traci.trafficlight.setRedYellowGreenState(self.tls_id, self.phase_defs[phase_idx])
        # In real use, set phase duration if needed; here assuming state sets it

    def close(self):
        try:
            traci.close()
        except traci.exceptions.TraCIException:
            pass

    def render(self, mode='human'):
        pass  # GUI handled by SUMO if using sumo-gui

if __name__ == "__main__":
    config = {  # Sample config
        "sumocfg": "configs/sumo.sumocfg",
        "min_green": 5,
        "max_green": 60,
        "green_step": 5,
        "cycle_yellow": 3,
        "cycle_all_red": 1,
        "queue_capacity": 40,
        "reward_weights": {"queue": -1.0, "wait_penalty": -0.1}
    }
    env = SumoEnv(config)
    obs, _ = env.reset()
    print("Initial queues:", obs)
    for _ in range(5):
        action = np.random.randint(0, env.action_space.n)
        obs, reward, _, _, _ = env.step(action)
        print("Action:", action, "Queues:", obs, "Reward:", reward)
    env.close()