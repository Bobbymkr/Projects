"""
Training Script for Distributional RL (C51, QR-DQN).

Implements Phase 2.3 from SCORE_IMPROVEMENT_ROADMAP.md:
- Distributional RL training with C51 or QR-DQN
- Integration with curriculum learning
- Performance validation
"""

import json
import os
import argparse
import numpy as np
from tqdm import trange
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Try to import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.env.traffic_env import TrafficEnv
from src.rl.curriculum_learning import TrafficCurriculum
from src.rl.convergence_monitor import ConvergenceMonitor

# Try to import distributional RL
try:
    import torch
    from src.rl.distributional_rl import (
        DistributionalDQNAgent,
        DistributionalRLConfig,
    )
    DISTRIBUTIONAL_RL_AVAILABLE = True
except ImportError as e:
    DISTRIBUTIONAL_RL_AVAILABLE = False
    print(f"Warning: Distributional RL not available: {e}")
    print("Install PyTorch to use distributional RL: pip install torch")


def load_config(path: str):
    """Load configuration from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def train(
    cfg_path: str,
    episodes: int,
    out_dir: str,
    algorithm: str = "C51",
    use_curriculum: bool = True,
):
    """
    Train Distributional RL agent.
    
    Args:
        cfg_path: Path to environment configuration file
        episodes: Number of training episodes
        out_dir: Output directory for saving models and checkpoints
        algorithm: Distributional RL algorithm ("C51" or "QR-DQN")
        use_curriculum: Whether to use curriculum learning
    """
    if not DISTRIBUTIONAL_RL_AVAILABLE or not TORCH_AVAILABLE:
        print("Error: Distributional RL requires PyTorch. Please install: pip install torch")
        return
    
    os.makedirs(out_dir, exist_ok=True)
    env_cfg = load_config(cfg_path)
    env = TrafficEnv(env_cfg)
    
    # Create distributional RL config
    dist_config = DistributionalRLConfig(
        algorithm=algorithm,
        num_atoms=51 if algorithm == "C51" else 0,
        num_quantiles=200 if algorithm == "QR-DQN" else 0,
        v_min=-10.0,
        v_max=10.0,
        risk_type="neutral",
    )
    
    # Initialize agent
    agent = DistributionalDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=dist_config,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Initialize convergence monitor
    convergence_monitor = ConvergenceMonitor(
        window=100,
        threshold=0.01,
        patience=500,
        min_episodes=200,
        mode="maximize"
    )
    
    # Initialize curriculum learning if enabled
    curriculum = None
    if use_curriculum:
        base_arrival_rates = env_cfg.get("arrival_rates", [0.3] * env.num_lanes)
        curriculum = TrafficCurriculum(
            base_arrival_rates=base_arrival_rates,
            performance_threshold=0.7,
            min_episodes_per_level=50,
            performance_window=100
        )
        print(f"Curriculum learning enabled with {len(curriculum.levels)} levels")
    
    # Training loop
    rewards = []
    checkpoint_path = os.path.join(out_dir, 'checkpoint')
    os.makedirs(checkpoint_path, exist_ok=True)
    
    print(f"Training {algorithm} agent for {episodes} episodes...")
    
    for ep in trange(episodes, desc="Training"):
        # Update environment with curriculum level if enabled
        if curriculum is not None:
            current_level = curriculum.get_current_level()
            env.arrival_rates = curriculum.get_arrival_rates()
            if ep % 50 == 0:
                print(f"Episode {ep}: Curriculum level {current_level.level_id} - {current_level.description}")
        
        obs, info = env.reset()
        episode_reward = 0.0
        terminated = truncated = False
        
        while not (terminated or truncated):
            action = agent.select_action(obs.astype(np.float32))
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            agent.push(
                obs.astype(np.float32),
                action,
                reward,
                next_obs.astype(np.float32),
                terminated or truncated
            )
            
            loss = agent.train_step()
            episode_reward += reward
            obs = next_obs
        
        rewards.append(episode_reward)
        
        # Update curriculum learning
        if curriculum is not None:
            curriculum.update_performance(episode_reward, ep)
        
        # Update convergence monitor
        monitor_status = convergence_monitor.update(episode_reward, ep)
        if monitor_status["should_stop"]:
            print(f"\nEarly stopping triggered at episode {ep}")
            break
        
        # Save checkpoint periodically
        if (ep + 1) % 50 == 0:
            # Save PyTorch model
            torch.save({
                'policy_net': agent.policy_net.state_dict(),
                'target_net': agent.target_net.state_dict(),
                'optimizer': agent.optimizer.state_dict(),
                'steps': agent.steps,
                'epsilon': agent.epsilon,
            }, os.path.join(checkpoint_path, f'{algorithm.lower()}_traffic_ep{ep+1}.pt'))
            np.save(os.path.join(checkpoint_path, 'rewards.npy'), np.array(rewards))
            
            # Log progress
            if ep % 50 == 0:
                print(f"Episode {ep}: Avg Reward={monitor_status['recent_avg']:.2f}, "
                      f"Std={monitor_status['recent_std']:.2f}")
    
    # Final save
    torch.save({
        'policy_net': agent.policy_net.state_dict(),
        'target_net': agent.target_net.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'steps': agent.steps,
        'epsilon': agent.epsilon,
    }, os.path.join(checkpoint_path, f'{algorithm.lower()}_traffic_final.pt'))
    np.save(os.path.join(checkpoint_path, 'rewards.npy'), np.array(rewards))
    
    print(f"\nTraining complete!")
    print(f"Final average reward: {np.mean(rewards[-100:]):.2f}")
    print(f"Final std: {np.std(rewards[-100:]):.2f}")
    if curriculum is not None:
        print(f"Final curriculum level: {curriculum.current_level}/{len(curriculum.levels)-1}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Distributional RL agent")
    parser.add_argument("--config", default="configs/intersection.json", help="Environment config")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--out", default="runs/distributional_rl", help="Output directory")
    parser.add_argument("--algorithm", choices=["C51", "QR-DQN"], default="C51", help="Algorithm")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    
    args = parser.parse_args()
    
    train(
        cfg_path=args.config,
        episodes=args.episodes,
        out_dir=args.out,
        algorithm=args.algorithm,
        use_curriculum=not args.no_curriculum,
    )


if __name__ == "__main__":
    main()

