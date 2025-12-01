#!/usr/bin/env python3
"""
Training script for Model-Based Reinforcement Learning.

This completes the 60% partial implementation and trains the agent
to reduce waiting times using world model and MPC.
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

from src.research.novel_algorithms.model_based_rl_complete import ModelBasedRLAgent
from src.env.traffic_env import TrafficEnv
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model_based_rl(
    config_path: str,
    episodes: int = 1000,
    output_dir: str = "./runs/model_based_rl",
    device: str = "cpu",
    world_model_train_interval: int = 50,
):
    """
    Train Model-Based RL agent on traffic environment.
    
    Args:
        config_path: Path to traffic configuration file
        episodes: Number of training episodes
        output_dir: Directory to save models and results
        device: Device for training (cpu/cuda)
        world_model_train_interval: Episodes between world model training
    """
    logger.info("="*80)
    logger.info("MODEL-BASED RL TRAINING")
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
    agent = ModelBasedRLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
    )
    
    logger.info("Initialized Model-Based RL Agent")
    
    # Training statistics
    episode_rewards = []
    episode_wait_times = []
    episode_queue_lengths = []
    episode_lengths = []
    world_model_losses = []
    
    # Training loop
    logger.info(f"\nStarting training for {episodes} episodes...")
    logger.info(f"World model will be trained every {world_model_train_interval} episodes")
    
    for episode in tqdm(range(episodes), desc="Training"):
        # Reset environment
        obs, info = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_wait = 0.0
        episode_queues = []
        
        done = False
        
        while not done:
            # Select action using MPC
            action = agent.select_action(obs)
            
            # Execute action
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.add_transition(obs, action, reward, next_obs, done)
            
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
        
        # Train world model periodically
        if (episode + 1) % world_model_train_interval == 0 and len(agent.transition_buffer) >= 100:
            logger.info(f"\nTraining world model at episode {episode + 1}...")
            training_metrics = agent.train_world_model(epochs=50, batch_size=32)
            if training_metrics:
                world_model_losses.append({
                    'episode': episode + 1,
                    'transition_loss': training_metrics.get('final_transition_loss', 0.0),
                    'reward_loss': training_metrics.get('final_reward_loss', 0.0),
                })
                logger.info(
                    f"World model trained | "
                    f"Transition Loss: {training_metrics.get('final_transition_loss', 0.0):.4f} | "
                    f"Reward Loss: {training_metrics.get('final_reward_loss', 0.0):.4f}"
                )
        
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
                f"Transitions: {len(agent.transition_buffer)}"
            )
    
    # Final world model training
    if len(agent.transition_buffer) >= 100:
        logger.info("\nPerforming final world model training...")
        training_metrics = agent.train_world_model(epochs=100, batch_size=32)
        if training_metrics:
            world_model_losses.append({
                'episode': episodes,
                'transition_loss': training_metrics.get('final_transition_loss', 0.0),
                'reward_loss': training_metrics.get('final_reward_loss', 0.0),
            })
    
    # Save results
    logger.info("\nSaving results...")
    
    # Save agent
    agent_path = output_path / "model_based_rl_agent.pt"
    agent.save(str(agent_path))
    logger.info(f"Saved agent to {agent_path}")
    
    # Save training statistics
    results = {
        "episodes": episodes,
        "episode_rewards": [float(r) for r in episode_rewards],
        "episode_wait_times": [float(w) for w in episode_wait_times],
        "episode_queue_lengths": [float(q) for q in episode_queue_lengths],
        "episode_lengths": [int(l) for l in episode_lengths],
        "world_model_losses": [
            {
                'episode': int(wm['episode']),
                'transition_loss': float(wm['transition_loss']),
                'reward_loss': float(wm['reward_loss']),
            }
            for wm in world_model_losses
        ],
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
    logger.info(f"World Model Trained: {len(world_model_losses)} times")
    logger.info(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Model-Based RL Agent")
    parser.add_argument('--config', type=str, default='configs/intersection.json',
                       help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--output', type=str, default='./runs/model_based_rl',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for training')
    parser.add_argument('--train-interval', type=int, default=50,
                       help='Episodes between world model training')
    
    args = parser.parse_args()
    
    # Check for CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    train_model_based_rl(
        config_path=args.config,
        episodes=args.episodes,
        output_dir=args.output,
        device=args.device,
        world_model_train_interval=args.train_interval,
    )

