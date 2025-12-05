"""
Unit tests for Transformer-based Traffic Control Agent.

Tests for TransformerTrafficController and TransformerAgent.
"""

import pytest
import numpy as np
import torch

try:
    from src.research.novel_algorithms.transformer_control import (
        TransformerTrafficController,
        TransformerAgent,
        PositionalEncoding,
    )
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    pytest.skip("Transformer agent not available", allow_module_level=True)


@pytest.mark.skipif(not TRANSFORMER_AVAILABLE, reason="Transformer agent not available")
class TestPositionalEncoding:
    """Test suite for PositionalEncoding."""
    
    def test_positional_encoding_initialization(self):
        """Test PositionalEncoding initialization."""
        pe = PositionalEncoding(d_model=128, max_len=100)
        
        assert pe.pe.shape == (100, 128)
    
    def test_positional_encoding_forward(self):
        """Test positional encoding forward pass."""
        pe = PositionalEncoding(d_model=64, max_len=50)
        
        x = torch.randn(10, 2, 64)  # [seq_len, batch, d_model]
        output = pe(x)
        
        assert output.shape == x.shape
        assert not torch.equal(output, x)  # Should modify input


@pytest.mark.skipif(not TRANSFORMER_AVAILABLE, reason="Transformer agent not available")
class TestTransformerTrafficController:
    """Test suite for TransformerTrafficController."""
    
    def test_controller_initialization(self):
        """Test TransformerTrafficController initialization."""
        controller = TransformerTrafficController(
            state_dim=4,
            action_dim=12,
            d_model=128,
            nhead=8,
            num_layers=2,
        )
        
        assert controller.state_dim == 4
        assert controller.action_dim == 12
        assert controller.d_model == 128
    
    def test_controller_forward(self):
        """Test controller forward pass."""
        controller = TransformerTrafficController(
            state_dim=4,
            action_dim=12,
            d_model=64,
            nhead=4,
            num_layers=2,
        )
        
        # Create state sequence [seq_len, batch, state_dim]
        state_sequence = torch.randn(10, 2, 4)
        
        action_logits = controller(state_sequence)
        
        assert action_logits.shape == (2, 12)  # [batch, action_dim]
        assert torch.all(torch.isfinite(action_logits))


@pytest.mark.skipif(not TRANSFORMER_AVAILABLE, reason="Transformer agent not available")
class TestTransformerAgent:
    """Test suite for TransformerAgent."""
    
    def test_agent_initialization(self):
        """Test TransformerAgent initialization."""
        agent = TransformerAgent(
            state_dim=4,
            action_dim=12,
            d_model=128,
            nhead=8,
            num_layers=2,
        )
        
        assert agent.state_dim == 4
        assert agent.action_dim == 12
        assert agent.controller is not None
        assert agent.optimizer is not None
    
    def test_agent_select_action(self):
        """Test action selection."""
        agent = TransformerAgent(state_dim=4, action_dim=12)
        
        # Create state history
        state_history = [np.random.rand(4) for _ in range(10)]
        
        action = agent.select_action(state_history)
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 12
    
    def test_agent_update(self):
        """Test agent update."""
        agent = TransformerAgent(state_dim=4, action_dim=12)
        
        # Create sample batch
        state_history = [np.random.rand(4) for _ in range(10)]
        action = 0
        reward = -1.0
        next_state_history = [np.random.rand(4) for _ in range(10)]
        done = False
        
        # Update agent
        loss = agent.update(
            state_history,
            action,
            reward,
            next_state_history,
            done
        )
        
        assert isinstance(loss, (float, np.floating))
        assert loss >= 0
    
    def test_agent_save_load(self, tmp_path):
        """Test agent save and load."""
        agent = TransformerAgent(state_dim=4, action_dim=12)
        
        # Save agent
        save_path = tmp_path / "transformer_agent.pt"
        agent.save(str(save_path))
        
        assert save_path.exists()
        
        # Load agent
        loaded_agent = TransformerAgent(state_dim=4, action_dim=12)
        loaded_agent.load(str(save_path))
        
        assert loaded_agent.controller is not None


@pytest.mark.skipif(not TRANSFORMER_AVAILABLE, reason="Transformer agent not available")
class TestTransformerAgentIntegration:
    """Integration tests for Transformer Agent."""
    
    def test_full_training_cycle(self):
        """Test complete training cycle."""
        agent = TransformerAgent(state_dim=4, action_dim=12)
        
        # Simulate training episodes
        state_history = [np.random.rand(4) for _ in range(10)]
        
        for step in range(20):
            # Select action
            action = agent.select_action(state_history)
            assert 0 <= action < 12
            
            # Simulate environment
            next_state = np.random.rand(4)
            reward = -np.sum(np.abs(next_state))
            done = step == 19
            
            # Update state history
            state_history.append(next_state)
            if len(state_history) > 10:
                state_history.pop(0)
            
            next_state_history = state_history.copy()
            
            # Update agent
            loss = agent.update(
                state_history[:-1] if len(state_history) > 1 else state_history,
                action,
                reward,
                next_state_history,
                done
            )
            
            assert isinstance(loss, (float, np.floating))

