"""
Unit tests for Phase 7 Transfer Learning Framework.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.research.transfer_learning.phase7_transfer_learning import (
    DiverseScenarioGenerator,
    PreTrainingFramework,
    PreTrainingConfig,
    FineTuningFramework,
    FineTuningConfig,
    ContinualLearningFramework,
    CrossDomainTransfer,
)
from src.research.novel_algorithms.phase6_advanced_rl import PPOAgent, PPOConfig


class TestDiverseScenarioGenerator:
    """Test scenario generator."""
    
    @pytest.fixture
    def base_config(self):
        return {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
        }
    
    @pytest.fixture
    def generator(self, base_config):
        return DiverseScenarioGenerator(base_config)
    
    def test_generate_scenario(self, generator):
        """Test scenario generation."""
        scenario = generator.generate_scenario("low_traffic")
        
        assert isinstance(scenario, dict)
        assert "arrival_rates" in scenario
        assert len(scenario["arrival_rates"]) == 4
        assert all(0 <= rate <= 1 for rate in scenario["arrival_rates"])
        assert "_scenario_type" in scenario
        assert scenario["_scenario_type"] == "low_traffic"
    
    def test_generate_batch(self, generator):
        """Test batch generation."""
        batch = generator.generate_batch(10)
        
        assert len(batch) == 10
        assert all(isinstance(s, dict) for s in batch)
        assert all("arrival_rates" in s for s in batch)
    
    def test_scenario_types(self, generator):
        """Test different scenario types."""
        types = ["low_traffic", "moderate_traffic", "high_traffic", "rush_hour"]
        
        for scenario_type in types:
            scenario = generator.generate_scenario(scenario_type)
            assert scenario["_scenario_type"] == scenario_type
            assert "arrival_rates" in scenario


class TestPreTrainingFramework:
    """Test pre-training framework."""
    
    @pytest.fixture
    def base_config(self):
        return {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "arrival_rates": [0.3, 0.3, 0.3, 0.3],
        }
    
    @pytest.fixture
    def temp_dir(self):
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def framework(self, base_config, temp_dir):
        config = PreTrainingConfig(
            total_episodes=100,  # Small for testing
            episodes_per_scenario=10,
            checkpoint_interval=50,
            save_dir=str(temp_dir),
        )
        return PreTrainingFramework(
            agent_class=PPOAgent,
            agent_config=PPOConfig(batch_size=16, buffer_size=100),
            base_env_config=base_config,
            pretraining_config=config,
        )
    
    def test_framework_initialization(self, framework):
        """Test framework initialization."""
        assert framework.agent_class == PPOAgent
        assert isinstance(framework.scenario_generator, DiverseScenarioGenerator)
        assert framework.config.total_episodes == 100
    
    def test_create_agent(self, framework):
        """Test agent creation."""
        agent = framework.create_agent(state_dim=4, action_dim=2)
        assert isinstance(agent, PPOAgent)
    
    def test_save_checkpoint(self, framework, temp_dir):
        """Test checkpoint saving."""
        from src.env.traffic_env import TrafficEnv
        
        env = TrafficEnv(config=framework.base_env_config)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        agent = framework.create_agent(state_dim, action_dim)
        framework.save_checkpoint(agent, 10)
        
        # Check checkpoint was saved
        checkpoints = list(temp_dir.glob("*.pt"))
        assert len(checkpoints) > 0


class TestFineTuningFramework:
    """Test fine-tuning framework."""
    
    @pytest.fixture
    def base_config(self):
        return {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "arrival_rates": [0.3, 0.3, 0.3, 0.3],
        }
    
    @pytest.fixture
    def temp_dir(self):
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def pretrained_model(self, base_config, temp_dir):
        """Create a dummy pre-trained model."""
        from src.env.traffic_env import TrafficEnv
        
        env = TrafficEnv(config=base_config)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        agent = PPOAgent(state_dim, action_dim, PPOConfig(batch_size=16, buffer_size=100))
        
        # Save model
        model_path = temp_dir / "pretrained_model.pt"
        torch.save({
            'policy_state_dict': agent.policy_net.state_dict(),
            'value_state_dict': agent.value_net.state_dict(),
            'episode': 100,
            'config': PPOConfig(),
        }, model_path)
        
        return str(model_path)
    
    @pytest.fixture
    def framework(self, base_config, pretrained_model, temp_dir):
        config = FineTuningConfig(
            episodes=100,  # Small for testing
            learning_rate=1e-4,
            checkpoint_interval=50,
            save_dir=str(temp_dir),
        )
        return FineTuningFramework(
            pretrained_model_path=pretrained_model,
            agent_class=PPOAgent,
            agent_config=PPOConfig(batch_size=16, buffer_size=100),
            target_env_config=base_config,
            finetuning_config=config,
        )
    
    def test_framework_initialization(self, framework):
        """Test framework initialization."""
        assert framework.agent_class == PPOAgent
        assert framework.config.episodes == 100
    
    def test_load_pretrained_model(self, framework):
        """Test loading pre-trained model."""
        from src.env.traffic_env import TrafficEnv
        
        env = TrafficEnv(config=framework.target_env_config)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        agent = framework.load_pretrained_model(state_dim, action_dim)
        assert isinstance(agent, PPOAgent)


class TestContinualLearningFramework:
    """Test continual learning framework."""
    
    @pytest.fixture
    def agent(self):
        return PPOAgent(4, 2, PPOConfig(batch_size=16, buffer_size=100))
    
    @pytest.fixture
    def continual_learner(self, agent):
        return ContinualLearningFramework(agent, replay_buffer_size=100)
    
    def test_add_experience(self, continual_learner):
        """Test adding experience."""
        state = np.random.randn(4)
        action = 1
        reward = 10.0
        next_state = np.random.randn(4)
        done = False
        
        continual_learner.add_experience(state, action, reward, next_state, done)
        
        assert len(continual_learner.replay_buffer) == 1
        assert continual_learner.adaptation_stats["samples_seen"] == 1
    
    def test_adapt(self, continual_learner):
        """Test adaptation."""
        # Add enough experiences
        for _ in range(50):
            state = np.random.randn(4)
            continual_learner.add_experience(state, 0, 1.0, state, False)
        
        initial_updates = continual_learner.adaptation_stats["updates"]
        continual_learner.adapt(batch_size=32)
        
        assert continual_learner.adaptation_stats["updates"] > initial_updates


class TestCrossDomainTransfer:
    """Test cross-domain transfer."""
    
    def test_transfer_weights_full(self):
        """Test full weight transfer."""
        # Create simple models
        source = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
        
        target = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
        
        # Transfer weights
        transferred = CrossDomainTransfer.transfer_weights(source, target, "full")
        
        # Check weights were transferred
        assert transferred is not None
    
    def test_transfer_weights_partial(self):
        """Test partial weight transfer."""
        source = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
        
        target = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
        
        transferred = CrossDomainTransfer.transfer_weights(source, target, "partial")
        assert transferred is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

