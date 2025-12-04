"""
Performance Microbenchmarks.

Comprehensive performance benchmarks for critical components.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock

# Import components
from src.rl.dqn_agent import DQNAgent, DQNConfig
from src.control.fuzzy_control import FuzzyLogicController
from src.forecast.traffic_forecast import TrafficForecaster


class TestAgentPerformance:
    """Benchmark agent performance."""
    
    def test_dqn_inference_latency(self, benchmark):
        """Benchmark DQN agent inference latency."""
        config = DQNConfig(
            state_dim=12,
            action_dim=4,
            learning_rate=0.001
        )
        agent = DQNAgent(config)
        state = np.random.rand(12)
        
        def inference():
            return agent.select_action(state)
        
        result = benchmark(inference)
        assert result is not None
        # Should be < 10ms for real-time
        assert benchmark.stats.mean < 0.01
    
    def test_fuzzy_controller_latency(self, benchmark):
        """Benchmark fuzzy controller latency."""
        controller = FuzzyLogicController()
        queue_lengths = [5, 3, 8, 2]
        wait_times = [12, 8, 15, 6]
        
        def compute_timing():
            return controller.compute_timing(queue_lengths, wait_times)
        
        result = benchmark(compute_timing)
        assert result is not None
        # Should be very fast (< 1ms)
        assert benchmark.stats.mean < 0.001
    
    def test_batch_inference_performance(self):
        """Test batch inference performance."""
        config = DQNConfig(state_dim=12, action_dim=4)
        agent = DQNAgent(config)
        
        # Batch of states
        batch_size = 100
        states = np.random.rand(batch_size, 12)
        
        start = time.time()
        actions = [agent.select_action(state) for state in states]
        elapsed = time.time() - start
        
        avg_latency = elapsed / batch_size
        
        assert len(actions) == batch_size
        assert avg_latency < 0.01  # < 10ms per inference


class TestForecastingPerformance:
    """Benchmark forecasting performance."""
    
    def test_forecast_latency(self, benchmark):
        """Benchmark forecast generation latency."""
        forecaster = TrafficForecaster(
            input_timesteps=20,
            output_timesteps=5,
            features=16
        )
        input_seq = np.random.randn(1, 20, 16)
        
        def predict():
            return forecaster.predict(input_seq)
        
        result = benchmark(predict)
        assert result is not None
        # Should be < 50ms for real-time
        assert benchmark.stats.mean < 0.05
    
    def test_batch_forecasting(self):
        """Test batch forecasting performance."""
        forecaster = TrafficForecaster(
            input_timesteps=20,
            output_timesteps=5,
            features=16
        )
        
        batch_size = 10
        input_seqs = np.random.randn(batch_size, 20, 16)
        
        start = time.time()
        forecasts = forecaster.predict(input_seqs)
        elapsed = time.time() - start
        
        avg_latency = elapsed / batch_size
        
        assert forecasts.shape[0] == batch_size
        assert avg_latency < 0.1  # < 100ms per forecast


class TestMemoryUsage:
    """Test memory usage of components."""
    
    def test_agent_memory_footprint(self):
        """Test agent memory footprint."""
        import sys
        
        config = DQNConfig(
            state_dim=12,
            action_dim=4,
            memory_size=10000
        )
        agent = DQNAgent(config)
        
        # Estimate memory usage
        size = sys.getsizeof(agent)
        # Should be reasonable (< 100MB)
        assert size < 100 * 1024 * 1024
    
    def test_replay_buffer_memory(self):
        """Test replay buffer memory usage."""
        from src.rl.dqn_agent import ReplayBuffer
        
        buffer = ReplayBuffer(capacity=10000, state_dim=12)
        
        # Fill buffer
        for _ in range(1000):
            state = np.random.rand(12)
            buffer.add(state, 0, 0.0, state, False)
        
        # Memory should be bounded
        assert buffer.size() <= buffer.capacity


class TestScalability:
    """Test system scalability."""
    
    def test_multi_agent_scalability(self):
        """Test scalability with multiple agents."""
        num_agents = 10
        agents = []
        
        for i in range(num_agents):
            config = DQNConfig(state_dim=12, action_dim=4)
            agents.append(DQNAgent(config))
        
        # All agents should work independently
        states = [np.random.rand(12) for _ in range(num_agents)]
        start = time.time()
        actions = [agent.select_action(state) for agent, state in zip(agents, states)]
        elapsed = time.time() - start
        
        assert len(actions) == num_agents
        assert elapsed < 1.0  # Should complete in < 1 second
    
    def test_throughput_under_load(self):
        """Test throughput under load."""
        controller = FuzzyLogicController()
        
        num_requests = 1000
        start = time.time()
        
        for _ in range(num_requests):
            queue_lengths = np.random.randint(0, 20, 4).tolist()
            wait_times = np.random.uniform(0, 60, 4).tolist()
            controller.compute_timing(queue_lengths, wait_times)
        
        elapsed = time.time() - start
        throughput = num_requests / elapsed
        
        # Should handle > 1000 req/s
        assert throughput > 1000


class TestConcurrency:
    """Test concurrent operation performance."""
    
    def test_concurrent_agent_inference(self):
        """Test concurrent agent inference."""
        import threading
        
        config = DQNConfig(state_dim=12, action_dim=4)
        agent = DQNAgent(config)
        
        results = []
        lock = threading.Lock()
        
        def inference_worker():
            state = np.random.rand(12)
            action = agent.select_action(state)
            with lock:
                results.append(action)
        
        # Run concurrent inferences
        threads = []
        for _ in range(10):
            t = threading.Thread(target=inference_worker)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(results) == 10
        assert all(0 <= a < 4 for a in results)

