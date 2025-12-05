"""
Automated Hyperparameter Optimization Script.

Uses Optuna to optimize DQN agent hyperparameters for best performance.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

from src.env.traffic_env import TrafficEnv
from src.rl.dqn_agent import DQNAgent, DQNConfig
from src.research.hyperparameter_optimization import (
    HyperparameterOptimizer,
    create_dqn_optimization_objective,
)


def train_agent_with_config(config: dict, episodes: int = 50) -> DQNAgent:
    """
    Train DQN agent with given hyperparameters.
    
    Args:
        config: Hyperparameter configuration
        episodes: Number of training episodes
        
    Returns:
        Trained agent
    """
    # Load environment config
    env_config_path = config.get("env_config", "configs/intersection.json")
    with open(env_config_path, 'r') as f:
        env_config = json.load(f)
    
    env = TrafficEnv(env_config)
    
    # Create DQN config from hyperparameters
    dqn_config = DQNConfig(
        lr=config.get("learning_rate", 1e-3),
        gamma=config.get("gamma", 0.99),
        eps_start=config.get("epsilon_start", 1.0),
        eps_end=config.get("epsilon_end", 0.05),
        eps_decay=config.get("epsilon_decay", 20000),
        batch_size=config.get("batch_size", 64),
        target_update=config.get("target_update", 1000),
        buffer_size=config.get("replay_buffer_size", 50000),
        grad_clip_norm=config.get("grad_clip_norm", 10.0),
        lr_schedule=config.get("lr_schedule", "cosine"),
        soft_update_tau=config.get("soft_update_tau", 0.005),
    )
    
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        cfg=dqn_config
    )
    
    # Training loop
    rewards = []
    for episode in range(episodes):
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
    
    return agent, rewards


def evaluate_agent(agent: DQNAgent, eval_episodes: int = 10) -> float:
    """
    Evaluate agent performance.
    
    Args:
        agent: Trained agent
        eval_episodes: Number of evaluation episodes
        
    Returns:
        Average reward
    """
    env_config_path = "configs/intersection.json"
    with open(env_config_path, 'r') as f:
        env_config = json.load(f)
    
    env = TrafficEnv(env_config)
    
    eval_rewards = []
    for episode in range(eval_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        terminated = truncated = False
        
        while not (terminated or truncated):
            action = agent.select_action(obs.astype(np.float32), evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
        
        eval_rewards.append(episode_reward)
    
    return np.mean(eval_rewards)


def main():
    """Main optimization function."""
    parser = argparse.ArgumentParser(description="Optimize DQN hyperparameters")
    parser.add_argument("--config", default="configs/intersection.json", help="Environment config")
    parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--episodes", type=int, default=50, help="Training episodes per trial")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--out", default="runs/hyperopt", help="Output directory")
    parser.add_argument("--study-name", default="dqn-hyperopt", help="Optuna study name")
    parser.add_argument("--timeout", type=float, default=None, help="Timeout in seconds")
    
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        study_name=args.study_name,
        direction="maximize",
        sampler="tpe"
    )
    
    if not optimizer.enabled:
        print("Optuna not available. Please install: pip install optuna")
        return
    
    # Create objective function
    def train_func(config):
        agent, rewards = train_agent_with_config(config, episodes=args.episodes)
        return agent
    
    def eval_func(agent):
        return evaluate_agent(agent, eval_episodes=args.eval_episodes)
    
    objective = create_dqn_optimization_objective(train_func, eval_func)
    
    # Wrap objective to include env config
    def wrapped_objective(trial):
        trial_config = {}
        # Add environment config path
        trial_config["env_config"] = args.config
        return objective(trial)
    
    print(f"Starting hyperparameter optimization with {args.trials} trials...")
    print(f"Training {args.episodes} episodes per trial, evaluating with {args.eval_episodes} episodes")
    
    # Run optimization
    results = optimizer.optimize(
        wrapped_objective,
        n_trials=args.trials,
        timeout=args.timeout,
        show_progress=True
    )
    
    # Save results
    if results:
        print(f"\nOptimization complete!")
        print(f"Best value: {results.get('best_value', 'N/A')}")
        print(f"Best parameters:")
        for key, value in results.get('best_params', {}).items():
            print(f"  {key}: {value}")
        
        # Save to file
        results_path = os.path.join(args.out, "best_params.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")
        
        # Save all trials
        trials = optimizer.get_trials()
        trials_path = os.path.join(args.out, "all_trials.json")
        with open(trials_path, 'w') as f:
            json.dump(trials, f, indent=2)
        print(f"All trials saved to {trials_path}")
    else:
        print("Optimization failed or returned no results.")


if __name__ == "__main__":
    main()
