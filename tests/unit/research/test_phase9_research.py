"""
Comprehensive Unit Tests for Phase 9 Research Improvements.

Tests:
- PC Algorithm for causal discovery
- Causal RL Agent
- Enhanced Neuro-Symbolic Agent
- Enhanced Federated Learning
- Phase 9 Integrated Agent
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.research.phase9_research_improvements import (
    PCAlgorithm,
    CausalRLAgent,
    EnhancedNeuroSymbolicAgent,
    EnhancedFederatedLearning,
    Phase9IntegratedAgent,
)
from src.research.novel_algorithms.neuro_symbolic import SymbolicRule


class TestPCAlgorithm:
    """Test PC Algorithm for causal discovery."""
    
    @pytest.fixture
    def pc_algorithm(self):
        return PCAlgorithm(alpha=0.05)
    
    def test_independence_test(self, pc_algorithm):
        """Test independence testing."""
        # Independent variables
        x = np.random.randn(100)
        y = np.random.randn(100)
        is_indep, p_value = pc_algorithm.test_independence(x, y)
        
        assert isinstance(is_indep, (bool, np.bool_))
        assert isinstance(p_value, (float, np.floating))
        assert 0 <= float(p_value) <= 1
    
    def test_causal_discovery(self, pc_algorithm):
        """Test causal graph discovery."""
        # Generate synthetic data
        n_samples = 200
        n_vars = 4
        data = np.random.randn(n_samples, n_vars)
        variable_names = [f"var_{i}" for i in range(n_vars)]
        
        graph = pc_algorithm.discover_causal_graph(data, variable_names)
        
        assert graph is not None
        assert len(graph.nodes) == n_vars


class TestCausalRLAgent:
    """Test Causal RL Agent."""
    
    @pytest.fixture
    def agent(self):
        return CausalRLAgent(state_dim=4, action_dim=2)
    
    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.state_dim == 4
        assert agent.action_dim == 2
        assert agent.causal_model is not None
    
    def test_select_action(self, agent):
        """Test action selection."""
        state = np.random.randn(4)
        action, log_prob = agent.select_action(state)
        
        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim
        assert isinstance(log_prob, float)
    
    def test_train_step(self, agent):
        """Test training step."""
        states = np.random.randn(10, 4)
        actions = np.random.randint(0, 2, 10)
        rewards = np.random.randn(10)
        next_states = np.random.randn(10, 4)
        
        metrics = agent.train_step(states, actions, rewards, next_states)
        
        assert 'loss' in metrics


class TestEnhancedNeuroSymbolicAgent:
    """Test Enhanced Neuro-Symbolic Agent."""
    
    @pytest.fixture
    def agent(self):
        return EnhancedNeuroSymbolicAgent(state_dim=4, action_dim=2)
    
    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.state_dim == 4
        assert agent.action_dim == 2
    
    def test_add_symbolic_rule(self, agent):
        """Test adding symbolic rules."""
        rule = SymbolicRule("var_0 > 0.5", action=0, confidence=0.8)
        agent.add_symbolic_rule(rule)
        
        assert len(agent.symbolic_rules) == 1
    
    def test_select_action(self, agent):
        """Test action selection."""
        state = np.array([0.8, 0.2, 0.3, 0.4])
        action, explanation = agent.select_action(state, explain=True)
        
        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim
        assert isinstance(explanation, dict)
        assert 'neural_prob' in explanation
    
    def test_train_step(self, agent):
        """Test training step."""
        states = np.random.randn(10, 4)
        actions = np.random.randint(0, 2, 10)
        rewards = np.random.randn(10)
        
        metrics = agent.train_step(states, actions, rewards)
        
        assert 'loss' in metrics


class TestEnhancedFederatedLearning:
    """Test Enhanced Federated Learning."""
    
    @pytest.fixture
    def fed_learning(self):
        initial_weights = {
            'layer1': np.random.randn(10, 10),
            'layer2': np.random.randn(5, 10),
        }
        return EnhancedFederatedLearning(initial_weights)
    
    def test_initialization(self, fed_learning):
        """Test initialization."""
        assert fed_learning.global_weights is not None
        assert fed_learning.round_number == 0
    
    def test_add_client_update(self, fed_learning):
        """Test adding client updates."""
        weights = {
            'layer1': np.random.randn(10, 10),
            'layer2': np.random.randn(5, 10),
        }
        fed_learning.add_client_update("client1", weights, num_samples=100)
        
        assert len(fed_learning.client_updates) == 1
    
    def test_aggregate_updates(self, fed_learning):
        """Test aggregating updates."""
        # Add multiple client updates
        for i in range(3):
            weights = {
                'layer1': np.random.randn(10, 10),
                'layer2': np.random.randn(5, 10),
            }
            fed_learning.add_client_update(f"client{i}", weights, num_samples=100)
        
        aggregated = fed_learning.aggregate_updates()
        
        assert aggregated is not None
        assert 'layer1' in aggregated
        assert fed_learning.round_number == 1


class TestPhase9IntegratedAgent:
    """Test Phase 9 Integrated Agent."""
    
    def test_initialization_causal_only(self):
        """Test initialization with causal only."""
        agent = Phase9IntegratedAgent(
            state_dim=4, action_dim=2,
            use_causal=True, use_neuro_symbolic=False
        )
        
        assert agent.causal_agent is not None
        assert agent.neuro_symbolic_agent is None
    
    def test_initialization_neuro_symbolic_only(self):
        """Test initialization with neuro-symbolic only."""
        agent = Phase9IntegratedAgent(
            state_dim=4, action_dim=2,
            use_causal=False, use_neuro_symbolic=True
        )
        
        assert agent.neuro_symbolic_agent is not None
        assert agent.causal_agent is None
    
    def test_select_action(self):
        """Test action selection."""
        agent = Phase9IntegratedAgent(
            state_dim=4, action_dim=2,
            use_neuro_symbolic=True
        )
        
        state = np.random.randn(4)
        action, metadata = agent.select_action(state)
        
        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim
        assert isinstance(metadata, dict)
    
    def test_train_step(self):
        """Test training step."""
        agent = Phase9IntegratedAgent(
            state_dim=4, action_dim=2,
            use_causal=True, use_neuro_symbolic=True
        )
        
        states = np.random.randn(10, 4)
        actions = np.random.randint(0, 2, 10)
        rewards = np.random.randn(10)
        
        metrics = agent.train_step(states, actions, rewards)
        
        assert isinstance(metrics, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

