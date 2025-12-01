#!/usr/bin/env python3
"""
Training script for Hierarchical Reinforcement Learning.

This completes the 70% partial implementation and trains the agent
to reduce waiting times.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import logging
from tqdm import tqdm
import json
from datetime import datetime

from src.research.novel_algorithms.hierarchical_rl_complete import HierarchicalRLAgent
from src.env.traffic_env import TrafficEnv
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_hierarchical_rl(
    config_path: str,
    episodes: int = 1000,
    output_dir: str = "./runs/hierarchical_rl",
    device: str = "cpu",
):
    """
    Train Hierarchical RL agent on traffic environment.
    
    Args:
        config_path: Path to traffic configuration file
        episodes: Number of training episodes
        output_dir: Directory to save models and results
        device: Device for training (cpu/cuda)
    """
    logger.info("="*80)
    logger.info("HIERARCHICAL RL TRAINING")
    logger.info("="*80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config(config_path)
    traffic_config = config.get('traffic', config)
    
    # Create environment
    env = TrafficEnv(traffic_config)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    logger.info(f"State dimension: {state_dim}")
    logger.info(f"Action dimension: {action_dim}")
    
    # Create agent
    agent = HierarchicalRLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        use_domain_options=True,
        device=device,
    )
    
    logger.info(f"Initialized agent with {len(agent.options)} options")
    
    # Training statistics
    episode_rewards = []
    episode_wait_times = []
    episode_queue_lengths = []
    episode_lengths = []
    
    # Training loop
    logger.info(f"\nStarting training for {episodes} episodes...")
    
    epsilon_start = 0.9
    epsilon_end = 0.05
    epsilon_decay = 0.995
    
    epsilon = epsilon_start
    
    for episode in tqdm(range(episodes), desc="Training"):
        # Reset environment
        obs, info = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_wait = 0.0
        episode_queues = []
        
        done = False
        agent.reset()
        
        active_option = None
        
        while not done:
            # Select action
            action = agent.select_action(obs, epsilon=epsilon)
            
            # Execute action
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            
            # Track active option
            current_option = agent.get_active_option()
            if current_option and current_option.option_id != active_option:
                active_option = current_option.option_id
                logger.debug(f"Episode {episode}, Step {episode_steps}: Active option = {active_option}")
            
            # Store experience for active option
            if active_option:
                agent.add_option_experience(
                    active_option,
                    obs,
                    action,
                    reward,
                    next_obs,
                )
            
            # Train option policies periodically
            if episode_steps % 10 == 0:
                for option in agent.options:
                    loss = agent.train_option_policy(option, batch_size=32)
            
            # Update statistics
            episode_reward += reward
            episode_steps += 1
            episode_wait += step_info.get('avg_wait_time', 0.0)
            episode_queues.append(np.sum(next_obs))
            
            obs = next_obs
        
        # Episode completed
        episode_rewards.append(episode_reward)
        episode_wait_times.append(episode_wait / episode_steps if episode_steps > 0 else 0.0)
        episode_queue_lengths.append(np.mean(episode_queues) if episode_queues else 0.0)
        episode_lengths.append(episode_steps)
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Log progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_wait = np.mean(episode_wait_times[-100:])
            avg_queue = np.mean(episode_queue_lengths[-100:])
            
            logger.info(
                f"Episode {episode + 1}/{episodes} | "
                f"Reward: {avg_reward:.2f} | "
                f"Wait Time: {avg_wait:.2f}s | "
                f"Queue: {avg_queue:.2f} | "
                f"Epsilon: {epsilon:.3f}"
            )
    
    # Save results
    logger.info("\nSaving results...")
    
    # Save agent
    agent_path = output_path / "hierarchical_rl_agent.pt"
    agent.save(str(agent_path))
    logger.info(f"Saved agent to {agent_path}")
    
    # Save training statistics
    results = {
        "episodes": episodes,
        "episode_rewards": [float(r) for r in episode_rewards],
        "episode_wait_times": [float(w) for w in episode_wait_times],
        "episode_queue_lengths": [float(q) for q in episode_queue_lengths],
        "episode_lengths": [int(l) for l in episode_lengths],
        "final_avg_reward": float(np.mean(episode_rewards[-100:])),
        "final_avg_wait_time": float(np.mean(episode_wait_times[-100:])),
        "final_avg_queue": float(np.mean(episode_queue_lengths[-100:])),
        "training_date": datetime.now().isoformat(),
    }
    
    results_path = output_path / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Final Average Reward: {results['final_avg_reward']:.2f}")
    logger.info(f"Final Average Wait Time: {results['final_avg_wait_time']:.2f}s")
    logger.info(f"Final Average Queue Length: {results['final_avg_queue']:.2f}")
    logger.info(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Hierarchical RL Agent")
    parser.add_argument('--config', type=str, default='configs/intersection.json',
                       help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--output', type=str, default='./runs/hierarchical_rl',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for training')
    
    args = parser.parse_args()
    
    # Check for CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    train_hierarchical_rl(
        config_path=args.config,
        episodes=args.episodes,
        output_dir=args.output,
        device=args.device,
    )

