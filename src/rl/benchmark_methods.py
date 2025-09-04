import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the project root to sys.path for module discovery
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.env.traffic_env import TrafficEnv 
from src.rl.dqn_agent import DQNAgent, DQNConfig
from src.control.fuzzy_control import FuzzyController
from src.optimization.genetic_algo import GeneticAlgorithm
from src.optimization.pso import ParticleSwarmOptimizer
from src.forecast.gnn_forecast import GNNForecaster
from src.control.webster_method import WebsterMethod
import tensorflow as tf

class BenchmarkRunner:
    """Class to run benchmarks for different traffic control methods."""

    def __init__(self, env_config, num_episodes=10):
        self.env = TrafficEnv(env_config)
        self.num_episodes = num_episodes
        self.methods = ['dqn', 'fuzzy', 'genetic', 'pso', 'gnn', 'webster']
        self.metrics = {method: {'wait_time': [], 'queue_length': [], 'efficiency': []} for method in self.methods}

        # Safe defaults referenced later
        self.num_lanes = env_config.get('num_lanes', getattr(self.env, 'num_lanes', None))
        min_g = env_config.get('min_green', 5)
        max_g = env_config.get('max_green', 60)
        step = env_config.get('green_step', 5)
        self.green_values = np.arange(min_g, max_g + 1, step)

    def run_benchmark(self):
        """Run benchmark for all methods."""
        for method in self.methods:
            for _ in range(self.num_episodes):
                metrics = self._run_episode(method)
                self.metrics[method]['wait_time'].append(metrics['wait_time'])
                self.metrics[method]['queue_length'].append(metrics['queue_length'])
                self.metrics[method]['efficiency'].append(metrics['efficiency'])

    def _run_episode(self, method):
        """Run a single episode for a given method."""
        obs, info = self.env.reset()
        done = False

        if method == 'dqn':
            agent = DQNAgent(obs.shape[0], self.env.action_space.n, DQNConfig())
        elif method == 'fuzzy':
            controller = FuzzyController()
        elif method == 'genetic':
            optimizer = GeneticAlgorithm()
        elif method == 'pso':
            optimizer = ParticleSwarmOptimizer()
        elif method == 'gnn':
            forecaster = GNNForecaster(num_nodes=self.num_lanes, input_dim=1, time_steps=1)
        elif method == 'webster':
            controller = WebsterMethod()

        while not done:
            if method == 'dqn':
                action = agent.select_action(obs)
            elif method == 'fuzzy':
                action = controller.compute_timing(obs)
            elif method == 'genetic':
                queue_lengths = obs[:self.env.num_lanes]  # Assuming obs contains queue lengths first
                wait_times = obs[self.env.num_lanes:self.env.num_lanes*2] # Assuming obs contains wait times second
                action = optimizer.optimize(queue_lengths, wait_times)
            elif method == 'pso':
                queue_lengths = obs[:self.env.num_lanes]
                wait_times = obs[self.env.num_lanes:self.env.num_lanes*2]
                action = optimizer.optimize(queue_lengths, wait_times)
            elif method == 'gnn':
                adj = np.eye(self.num_lanes, dtype=np.float32)
                inputs = tf.convert_to_tensor(obs[None, None, :, None], dtype=tf.float32)
                prediction = forecaster.call(inputs, adj)
                action = self._gnn_to_action(prediction)
            elif method == 'webster':
                action = controller.get_action({'volumes': obs * self.env.queue_capacity * 100})

            # Convert to discrete action if necessary
            if not isinstance(action, (int, np.integer)):
                if isinstance(action, dict):
                    green_time = np.mean(action['green_times'])
                elif isinstance(action, (np.ndarray, tf.Tensor)):
                    if isinstance(action, tf.Tensor):
                        action = action.numpy()
                    green_time = np.mean(action)
                else:
                    green_time = float(action)
                action = int(np.argmin(np.abs(self.green_values - green_time)))

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            obs = next_obs

        return {
            'wait_time': info['total_wait_time'] / max(1, info['total_vehicles_processed']),
            'queue_length': info['max_queue_length'],
            'efficiency': info['total_vehicles_processed'] / max(1, info['time'])
        }

    def _gnn_to_action(self, prediction):
        """Convert GNN prediction to action."""
        # Placeholder: implement based on prediction
        return np.argmax(prediction)

    def generate_report(self):
        """Generate comparison report with visualizations."""
        for metric in ['wait_time', 'queue_length', 'efficiency']:
            plt.figure()
            values = [np.mean(self.metrics[m][metric]) for m in self.methods]
            plt.bar(self.methods, values)
            plt.title(f'Average {metric.capitalize()} Comparison')
            plt.savefig(f'{metric}_comparison.png')
        print("Average wait_time Comparison")
        print("Average queue_length Comparison")
        print("Average efficiency Comparison")

if __name__ == '__main__':
    config = {
        "num_lanes": 4,
        "phase_lanes": [[0, 1], [2, 3]],
        "min_green": 5,
        "max_green": 60,
        "green_step": 5,
        "cycle_yellow": 3,
        "cycle_all_red": 1,
        "queue_capacity": 40,
        "arrival_rates": [0.3] * 4,
        "reward_weights": {"queue": -1.0, "wait_penalty": -0.1},
        "episode_horizon": 300
    }
    runner = BenchmarkRunner(config)
    runner.run_benchmark()
    runner.generate_report()
