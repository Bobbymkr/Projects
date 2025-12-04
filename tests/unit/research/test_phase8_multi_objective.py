"""
Comprehensive Unit Tests for Phase 8 Multi-Objective Optimization.

Tests all components as a top evaluator would:
- Multi-objective reward functions
- NSGA-II algorithm
- MO-PPO agent
- Constraint optimization
- CPO agent
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.research.multi_objective.phase8_multi_objective import (
    MultiObjectiveReward,
    MultiObjectiveWeights,
    NSGA2Solver,
    MOPPOAgent,
    MOPPOConfig,
    ConstraintOptimizer,
    ConstraintConfig,
    CPOAgent,
)


class TestMultiObjectiveReward:
    """Test multi-objective reward function."""
    
    @pytest.fixture
    def reward_fn(self):
        return MultiObjectiveReward()
    
    def test_reward_computation(self, reward_fn):
        """Test reward computation."""
        wait_times = np.array([10, 20, 30])
        queue_lengths = np.array([5, 10, 15])
        vehicles_served = 50
        phase_changes = 10
        emergency_stops = 0
        
        rewards = reward_fn.compute_rewards(
            wait_times, queue_lengths, vehicles_served, phase_changes, emergency_stops
        )
        
        assert 'total' in rewards
        assert 'wait_time' in rewards
        assert 'fuel_consumption' in rewards
        assert 'emissions' in rewards
        assert 'throughput' in rewards
        assert 'accidents' in rewards
        assert 'infrastructure_wear' in rewards
        
        assert isinstance(rewards['total'], float)
        assert isinstance(rewards['wait_time'], float)
    
    def test_scalar_reward(self, reward_fn):
        """Test scalar reward computation."""
        wait_times = np.array([10, 20, 30])
        queue_lengths = np.array([5, 10, 15])
        vehicles_served = 50
        phase_changes = 10
        
        scalar = reward_fn.compute_scalar_reward(
            wait_times, queue_lengths, vehicles_served, phase_changes
        )
        
        assert isinstance(scalar, float)
    
    def test_custom_weights(self):
        """Test custom objective weights."""
        weights = MultiObjectiveWeights(
            wait_time=0.5,
            fuel_consumption=0.2,
            emissions=0.1,
            throughput=0.1,
            accidents=0.05,
            infrastructure_wear=0.05,
        )
        
        reward_fn = MultiObjectiveReward(weights=weights)
        assert reward_fn.weights.wait_time == pytest.approx(0.5 / 1.0)


class TestNSGA2Solver:
    """Test NSGA-II solver."""
    
    @pytest.fixture
    def solver(self):
        return NSGA2Solver(
            population_size=50,
            num_generations=10,
            num_objectives=2,
        )
    
    def test_dominates(self, solver):
        """Test domination check."""
        obj1 = np.array([1.0, 2.0])  # Dominates obj2 and obj3
        obj2 = np.array([2.0, 3.0])  # Dominated by obj1
        obj3 = np.array([1.5, 2.5])  # Dominated by obj1, also dominates obj2
        
        assert solver.dominates(obj1, obj2) == True  # obj1 dominates obj2
        assert solver.dominates(obj2, obj1) == False  # obj2 doesn't dominate obj1
        assert solver.dominates(obj1, obj3) == True  # obj1 dominates obj3 (all values less)
        assert solver.dominates(obj3, obj2) == True  # obj3 dominates obj2 (all values less)
        # Test non-dominance case - obj4 = [1.0, 3.0] vs obj2 = [2.0, 3.0]
        # obj4 has 1.0 < 2.0 and 3.0 == 3.0, so obj4 DOES dominate obj2
        obj4 = np.array([1.0, 3.0])
        assert solver.dominates(obj4, obj2) == True  # obj4 dominates obj2
        # Test true non-dominance: obj5 = [1.0, 4.0] vs obj2 = [2.0, 3.0]
        # obj5 has 1.0 < 2.0 but 4.0 > 3.0, so neither dominates
        obj5 = np.array([1.0, 4.0])
        assert solver.dominates(obj5, obj2) == False  # obj5 doesn't dominate obj2
        assert solver.dominates(obj2, obj5) == False  # obj2 doesn't dominate obj5
    
    def test_non_dominated_sort(self, solver):
        """Test non-dominated sorting."""
        objectives = np.array([
            [1.0, 2.0],  # Dominates others
            [2.0, 3.0],
            [3.0, 1.0],
            [2.5, 2.5],
        ])
        
        fronts = solver.non_dominated_sort(objectives)
        
        assert len(fronts) > 0
        assert isinstance(fronts[0], list)
        assert len(fronts[0]) > 0
    
    def test_crowding_distance(self, solver):
        """Test crowding distance computation."""
        objectives = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ])
        front = [0, 1, 2]
        
        distances = solver.crowding_distance(objectives, front)
        
        assert len(distances) == len(front)
        assert np.all(distances >= 0)
        assert distances[0] == np.inf  # Boundary points
        assert distances[-1] == np.inf
    
    def test_solve(self, solver):
        """Test NSGA-II solving."""
        def evaluate_fn(x):
            # Simple 2-objective problem
            return np.array([np.sum(x**2), np.sum((x - 1)**2)])
        
        solutions, objectives = solver.solve(
            evaluate_fn,
            bounds=(0.0, 1.0),
            dim=5,
        )
        
        assert len(solutions) > 0
        assert len(objectives) > 0
        assert objectives.shape[1] == 2


class TestMOPPOAgent:
    """Test MO-PPO agent."""
    
    @pytest.fixture
    def agent(self):
        config = MOPPOConfig(
            batch_size=32,
            buffer_size=1000,
            num_objectives=6,
        )
        return MOPPOAgent(state_dim=4, action_dim=2, config=config)
    
    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.state_dim == 4
        assert agent.action_dim == 2
        assert agent.config.num_objectives == 6
    
    def test_select_action(self, agent):
        """Test action selection."""
        state = np.random.randn(4)
        action, log_prob, values = agent.select_action(state)
        
        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim
        assert isinstance(log_prob, float)
        assert values.shape == (6,)  # 6 objectives
    
    def test_store_transition(self, agent):
        """Test storing transitions."""
        state = np.random.randn(4)
        action = 0
        log_prob = -0.5
        values = torch.randn(6)
        rewards = np.random.randn(6)
        next_state = np.random.randn(4)
        done = False
        
        agent.store_transition(state, action, log_prob, values, rewards, next_state, done)
        
        assert len(agent.buffer) == 1
    
    def test_train_step(self, agent):
        """Test training step."""
        # Fill buffer
        for _ in range(agent.config.batch_size):
            state = np.random.randn(4)
            action = np.random.randint(0, agent.action_dim)
            log_prob = -0.5
            values = torch.randn(6)
            rewards = np.random.randn(6)
            next_state = np.random.randn(4)
            done = False
            
            agent.store_transition(state, action, log_prob, values, rewards, next_state, done)
        
        # Train
        metrics = agent.train_step()
        
        assert 'loss' in metrics
        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert 'entropy' in metrics


class TestConstraintOptimizer:
    """Test constraint optimizer."""
    
    @pytest.fixture
    def optimizer(self):
        config = ConstraintConfig(
            min_green_time=5.0,
            max_wait_time=300.0,
            max_queue_length=50.0,
        )
        return ConstraintOptimizer(config)
    
    def test_check_constraints(self, optimizer):
        """Test constraint checking."""
        action = 0
        green_time = 10.0
        wait_times = np.array([50, 100, 150])
        queue_lengths = np.array([10, 20, 30])
        
        satisfied, penalty = optimizer.check_constraints(
            action, green_time, wait_times, queue_lengths
        )
        
        assert isinstance(satisfied, bool)
        assert isinstance(penalty, float)
        assert penalty >= 0
    
    def test_constraint_violation(self, optimizer):
        """Test constraint violation detection."""
        # Violate minimum green time
        satisfied, penalty = optimizer.check_constraints(
            action=0,
            green_time=3.0,  # Below minimum
            wait_times=np.array([50]),
            queue_lengths=np.array([10]),
        )
        
        assert not satisfied or penalty > 0
    
    def test_apply_constraints(self, optimizer):
        """Test applying constraints to reward."""
        reward = 10.0
        action = 0
        green_time = 10.0
        wait_times = np.array([50])
        queue_lengths = np.array([10])
        
        constrained_reward = optimizer.apply_constraints(
            reward, action, green_time, wait_times, queue_lengths
        )
        
        assert isinstance(constrained_reward, float)


class TestCPOAgent:
    """Test CPO agent."""
    
    @pytest.fixture
    def agent(self):
        constraint_config = ConstraintConfig()
        ppo_config = MOPPOConfig(batch_size=32, buffer_size=1000)
        return CPOAgent(state_dim=4, action_dim=2, constraint_config=constraint_config, ppo_config=ppo_config)
    
    def test_initialization(self, agent):
        """Test CPO agent initialization."""
        assert agent.moppo is not None
        assert agent.lagrange_multipliers is not None
    
    def test_select_action(self, agent):
        """Test action selection."""
        state = np.random.randn(4)
        action, log_prob, values = agent.select_action(state)
        
        assert isinstance(action, int)
        assert 0 <= action < agent.moppo.action_dim
    
    def test_update_lagrange_multipliers(self, agent):
        """Test Lagrangian multiplier update."""
        constraint_violations = np.array([0.1, 0.2, 0.3])
        
        initial_multipliers = agent.lagrange_multipliers.clone()
        agent.update_lagrange_multipliers(constraint_violations)
        
        # Multipliers should be non-negative
        assert torch.all(agent.lagrange_multipliers >= 0)
    
    def test_train_step(self, agent):
        """Test CPO training step."""
        # Fill buffer
        for _ in range(agent.moppo.config.batch_size):
            state = np.random.randn(4)
            action = np.random.randint(0, agent.moppo.action_dim)
            log_prob = -0.5
            values = torch.randn(6)
            rewards = np.random.randn(6)
            next_state = np.random.randn(4)
            done = False
            
            agent.moppo.store_transition(state, action, log_prob, values, rewards, next_state, done)
        
        metrics = agent.train_step()
        
        assert 'loss' in metrics


class TestIntegration:
    """Integration tests."""
    
    def test_multi_objective_training_loop(self):
        """Test complete training loop with multi-objective rewards."""
        from src.env.traffic_env import TrafficEnv
        
        # Create environment
        config = {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "arrival_rates": [0.3, 0.3, 0.3, 0.3],
        }
        env = TrafficEnv(config=config)
        
        # Create agent
        agent = MOPPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            config=MOPPOConfig(batch_size=16, buffer_size=100),
        )
        
        # Create reward function
        reward_fn = MultiObjectiveReward()
        
        # Run episode
        obs, _ = env.reset()
        obs = np.array(obs, dtype=np.float32).flatten()
        done = False
        
        while not done:
            action, log_prob, values = agent.select_action(obs)
            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_obs = np.array(next_obs, dtype=np.float32).flatten()
            
            # Compute rewards
            rewards = reward_fn.compute_rewards(
                wait_times=env.wait_times,
                queue_lengths=env.queues,
                vehicles_served=10,
                phase_changes=1,
            )
            
            # Store transition
            agent.store_transition(
                obs, action, log_prob, values,
                np.array([
                    rewards['wait_time'],
                    rewards['fuel_consumption'],
                    rewards['emissions'],
                    rewards['throughput'],
                    rewards['accidents'],
                    rewards['infrastructure_wear'],
                ]),
                next_obs, done,
            )
            
            obs = next_obs
        
        # Train
        if len(agent.buffer) >= agent.config.batch_size:
            metrics = agent.train_step()
            assert 'loss' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

