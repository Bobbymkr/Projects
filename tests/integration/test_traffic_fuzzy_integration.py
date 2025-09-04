import unittest
import numpy as np
from src.env.traffic_env import TrafficEnv
from src.control.fuzzy_control import FuzzyController, get_fuzzy_action

class TestTrafficFuzzyIntegration(unittest.TestCase):
    def setUp(self):
        # Configuration for a simple traffic environment
        env_config = {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "yellow": 3,
            "all_red": 1,
            "queue_capacity": 40,
            "arrival_rates": [0.5, 0.5, 0.5, 0.5],
            "episode_horizon": 100
        }
        self.env = TrafficEnv(env_config)
        self.fuzzy_controller = FuzzyController()

    def test_env_fuzzy_controller_interaction(self):
        obs, info = self.env.reset(seed=42)
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < self.env.episode_horizon:
            # Get action from fuzzy controller
            queue_lengths = info["queues"]
            # The get_fuzzy_action expects a state, which is typically the observation from the env.
            # However, the fuzzy controller directly uses queue_lengths, so we pass that.
            # The action needs to be an index for the discrete action space of the env.
            # We need to map the computed green time from fuzzy controller to the action space index.
            
            # Compute green time from fuzzy controller
            fuzzy_green_time = self.fuzzy_controller.compute_timing(queue_lengths)
            
            # Find the closest action index in the environment's green_values
            action_idx = np.argmin(np.abs(self.env.green_values - fuzzy_green_time))
            action = action_idx

            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        self.assertGreater(steps, 0, "Simulation should run for at least one step.")
        self.assertIsInstance(total_reward, float, "Total reward should be a float.")
        # Add more specific assertions based on expected behavior
        # For example, check if queues are managed, or if total vehicles processed is reasonable.
        self.assertGreaterEqual(self.env.total_vehicles_processed, 0, "Vehicles processed should be non-negative.")
        self.assertLessEqual(np.max(self.env.queues), self.env.queue_capacity, "Queue length should not exceed capacity.")

if __name__ == '__main__':
    unittest.main()