"""
Test script for Prioritized Experience Replay (PER).

Tests Phase 2.2 implementation.
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.traffic_env import TrafficEnv
from src.rl.dqn_with_per import DQNAgentWithPER, PERConfig
from src.rl.pytorch_dqn import DQNConfig

def test_per():
    """Test PER implementation."""
    print("="*80)
    print("Testing Prioritized Experience Replay (Phase 2.2)")
    print("="*80)
    
    # Create environment
    config = {
        "num_lanes": 4,
        "phase_lanes": [[0, 1], [2, 3]],
        "min_green": 5,
        "max_green": 60,
        "green_step": 5,
        "cycle_yellow": 3,
        "cycle_all_red": 1,
        "arrival_rates": [0.3, 0.25, 0.35, 0.2],
        "queue_capacity": 40,
        "episode_horizon": 3600
    }
    env = TrafficEnv(config=config)
    
    # Create DQN agent with PER
    dqn_cfg = DQNConfig()
    per_cfg = PERConfig(
        enabled=True,
        alpha=0.6,
        beta=0.4,
        adaptive=False
    )
    
    agent = DQNAgentWithPER(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        cfg=dqn_cfg,
        per_config=per_cfg
    )
    
    print(f"\n✅ PER Agent created")
    print(f"   Buffer type: {type(agent.memory).__name__}")
    print(f"   Alpha: {agent.memory.alpha}")
    print(f"   Beta: {agent.memory.beta}")
    
    # Collect some experiences
    print(f"\n📊 Collecting experiences...")
    obs, _ = env.reset()
    for step in range(100):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.push(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            obs, _ = env.reset()
    
    print(f"   Buffer size: {len(agent.memory)}")
    
    # Test training with PER
    print(f"\n🎓 Testing training with PER...")
    losses = []
    for i in range(50):
        if len(agent.memory) >= agent.cfg.batch_size:
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
    
    if losses:
        print(f"   ✅ Training successful")
        print(f"   Average loss: {np.mean(losses):.6f}")
        print(f"   Beta after training: {agent.memory.beta:.4f}")
        print(f"   Max priority: {agent.memory.max_priority:.4f}")
    else:
        print(f"   ⚠️  Not enough samples for training")
    
    # Test adaptive PER
    print(f"\n🔄 Testing Adaptive PER...")
    from src.rl.prioritized_replay import AdaptivePER
    adaptive_per = AdaptivePER(agent.memory)
    
    # Simulate learning progress
    for progress in [0.01, 0.02, 0.05, 0.1, 0.15]:
        adaptive_per.update_alpha(progress)
        print(f"   Progress: {progress:.3f}, Alpha: {adaptive_per.get_alpha():.4f}")
    
    print(f"\n✅ PER Test Complete!")
    return True

if __name__ == "__main__":
    test_per()

