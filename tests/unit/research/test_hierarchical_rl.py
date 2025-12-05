"""
Unit tests for Hierarchical RL Agent.

Tests for OptionDiscovery, HierarchicalPolicy, and HierarchicalRLAgent.
"""

import pytest
import numpy as np
from typing import List, Tuple

from src.research.novel_algorithms.hierarchical_rl import (
    OptionDiscovery,
    HierarchicalPolicy,
    HierarchicalRLAgent,
    Option,
    OptionType,
)


class TestOptionDiscovery:
    """Test suite for OptionDiscovery."""
    
    def test_option_discovery_initialization(self):
        """Test OptionDiscovery initialization."""
        discovery = OptionDiscovery(state_dim=4)
        
        assert discovery.state_dim == 4
        assert discovery.min_option_length > 0
        assert discovery.max_option_length > discovery.min_option_length
        assert len(discovery.discovered_options) == 0
    
    def test_create_domain_options(self):
        """Test creation of domain-specific options."""
        discovery = OptionDiscovery(state_dim=4)
        
        options = discovery.create_domain_options()
        
        assert len(options) > 0
        assert all(isinstance(opt, Option) for opt in options)
        assert all(opt.option_type in OptionType for opt in options)
    
    def test_discover_options_from_experience(self):
        """Test option discovery from trajectories."""
        discovery = OptionDiscovery(state_dim=4)
        
        # Create sample trajectories
        trajectories = [
            [
                (np.random.rand(4), np.random.randint(0, 12), 
                 np.random.randn(), np.random.rand(4))
                for _ in range(20)
            ]
            for _ in range(5)
        ]
        
        options = discovery.discover_options_from_experience(
            trajectories, num_options=4
        )
        
        assert len(options) == 4
        assert all(isinstance(opt, Option) for opt in options)
        assert len(discovery.discovered_options) == 4


class TestHierarchicalPolicy:
    """Test suite for HierarchicalPolicy."""
    
    def test_hierarchical_policy_initialization(self):
        """Test HierarchicalPolicy initialization."""
        discovery = OptionDiscovery(state_dim=4)
        options = discovery.create_domain_options()
        
        policy = HierarchicalPolicy(options, primitive_action_dim=12)
        
        assert len(policy.options) == len(options)
        assert policy.primitive_action_dim == 12
        assert policy.current_option is None
        assert policy.option_steps == 0
    
    def test_select_primitive_action(self):
        """Test primitive action selection."""
        discovery = OptionDiscovery(state_dim=4)
        options = discovery.create_domain_options()
        
        policy = HierarchicalPolicy(options, primitive_action_dim=12)
        
        state = np.array([0.1, 0.2, 0.3, 0.4])
        action = policy.select_primitive_action(state)
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 12
    
    def test_option_selection_and_execution(self):
        """Test option selection and execution."""
        discovery = OptionDiscovery(state_dim=4)
        options = discovery.create_domain_options()
        
        policy = HierarchicalPolicy(options, primitive_action_dim=12)
        
        state = np.array([0.1, 0.2, 0.3, 0.4])
        
        # Select action (should select an option first)
        action1 = policy.select_primitive_action(state)
        assert 0 <= action1 < 12
        
        # Option should be active
        if policy.current_option is not None:
            assert isinstance(policy.current_option, Option)
            assert policy.option_steps > 0


class TestHierarchicalRLAgent:
    """Test suite for HierarchicalRLAgent."""
    
    def test_agent_initialization(self):
        """Test HierarchicalRLAgent initialization."""
        agent = HierarchicalRLAgent(state_dim=4, action_dim=12, use_domain_options=True)
        
        assert agent.state_dim == 4
        assert agent.action_dim == 12
        assert agent.option_discovery is not None
        assert len(agent.options) > 0
        assert agent.hierarchical_policy is not None
    
    def test_agent_initialization_no_domain_options(self):
        """Test initialization without domain options."""
        agent = HierarchicalRLAgent(state_dim=4, action_dim=12, use_domain_options=False)
        
        assert len(agent.options) == 0
    
    def test_agent_discover_options(self):
        """Test option discovery."""
        agent = HierarchicalRLAgent(state_dim=4, action_dim=12, use_domain_options=False)
        
        # Create trajectories
        trajectories = [
            [
                (np.random.rand(4), np.random.randint(0, 12), 
                 np.random.randn(), np.random.rand(4))
                for _ in range(20)
            ]
            for _ in range(5)
        ]
        
        initial_option_count = len(agent.options)
        agent.discover_options(trajectories, num_options=4)
        
        assert len(agent.options) == 4
        assert len(agent.hierarchical_policy.options) == 4
    
    def test_agent_select_action(self):
        """Test action selection."""
        agent = HierarchicalRLAgent(state_dim=4, action_dim=12)
        
        state = np.array([0.1, 0.2, 0.3, 0.4])
        action = agent.select_action(state)
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 12
    
    def test_agent_get_active_option(self):
        """Test getting active option."""
        agent = HierarchicalRLAgent(state_dim=4, action_dim=12)
        
        state = np.array([0.1, 0.2, 0.3, 0.4])
        agent.select_action(state)
        
        active_option = agent.get_active_option()
        # May be None if no option is currently active
        if active_option is not None:
            assert isinstance(active_option, Option)
    
    def test_agent_reset(self):
        """Test agent reset."""
        agent = HierarchicalRLAgent(state_dim=4, action_dim=12)
        
        state = np.array([0.1, 0.2, 0.3, 0.4])
        agent.select_action(state)
        
        # Option may be active
        agent.reset()
        
        assert agent.hierarchical_policy.current_option is None
        assert agent.hierarchical_policy.option_steps == 0


class TestHierarchicalRLIntegration:
    """Integration tests for Hierarchical RL."""
    
    def test_full_training_cycle(self):
        """Test complete training cycle with option discovery."""
        agent = HierarchicalRLAgent(state_dim=4, action_dim=12, use_domain_options=False)
        
        # Collect trajectories
        trajectories = []
        for episode in range(3):
            trajectory = []
            state = np.random.rand(4)
            
            for step in range(10):
                action = agent.select_action(state)
                next_state = state + np.random.randn(4) * 0.1
                reward = -np.sum(np.abs(next_state))
                
                trajectory.append((state.copy(), action, reward, next_state.copy()))
                state = next_state
            
            trajectories.append(trajectory)
        
        # Discover options from experience
        agent.discover_options(trajectories, num_options=3)
        
        assert len(agent.options) == 3
        
        # Test action selection with discovered options
        test_state = np.random.rand(4)
        action = agent.select_action(test_state)
        assert 0 <= action < 12
    
    def test_multi_episode_option_usage(self):
        """Test option usage across multiple episodes."""
        agent = HierarchicalRLAgent(state_dim=4, action_dim=12)
        
        states = [np.random.rand(4) for _ in range(20)]
        actions = []
        
        for state in states:
            action = agent.select_action(state)
            actions.append(action)
            assert 0 <= action < 12
        
        # Verify actions are valid
        assert len(actions) == 20
        assert all(0 <= a < 12 for a in actions)

