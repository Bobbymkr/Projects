"""
Unit tests for deadline-aware agent.
"""

import pytest
from unittest.mock import Mock
from src.realtime.deadline_aware_agent import DeadlineAwareAgent


class TestDeadlineAwareAgent:
    """Test deadline-aware agent wrapper."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent."""
        agent = Mock()
        agent.select_action.return_value = 1
        return agent
    
    @pytest.fixture
    def mock_fallback_agent(self):
        """Create mock fallback agent."""
        agent = Mock()
        agent.select_action.return_value = 0
        return agent
    
    def test_initialization(self, mock_agent):
        """Test agent initialization."""
        wrapper = DeadlineAwareAgent(mock_agent, deadline_ms=100)
        assert wrapper.agent == mock_agent
        assert wrapper.deadline_ms == 100
        assert wrapper.deadline_misses == 0
        assert wrapper.fallback_activations == 0
    
    def test_select_action_normal(self, mock_agent):
        """Test normal action selection."""
        wrapper = DeadlineAwareAgent(mock_agent, deadline_ms=100)
        state = [1.0, 2.0, 3.0, 4.0]
        
        action = wrapper.select_action(state)
        assert action == 1
        mock_agent.select_action.assert_called_once()
    
    def test_select_action_with_fallback(self, mock_agent, mock_fallback_agent):
        """Test action selection with fallback."""
        wrapper = DeadlineAwareAgent(mock_agent, deadline_ms=100, fallback_agent=mock_fallback_agent)
        state = [1.0, 2.0, 3.0, 4.0]
        
        action = wrapper.select_action(state)
        assert action in [0, 1]
    
    def test_get_stats(self, mock_agent):
        """Test getting agent statistics."""
        wrapper = DeadlineAwareAgent(mock_agent, deadline_ms=100)
        stats = wrapper.get_stats()
        
        assert "deadline_misses" in stats
        assert "fallback_activations" in stats
        assert "deadline_ms" in stats
        assert stats["deadline_ms"] == 100

