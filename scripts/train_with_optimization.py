"""
Enhanced Training Script with Phase 0 Optimizations.

Implements:
- Enhanced reward function (Phase 0.1)
- Training stability framework (Phase 0.2)
- Convergence detection (Phase 0.3)
"""

import argparse
import logging
import json
import numpy as np
import torch
from pathlib import Path
import sys
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.traffic_env import TrafficEnv
from src.rl.training_stability import TrainingStabilityFramework
from src.rl.convergence_monitor import ConvergenceMonitor, PerformanceTracker
from src.rl.curriculum_learning import TrafficCurriculum, AdaptiveCurriculum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_with_optimizations(
    agent,
    env: TrafficEnv,
    episodes: int = 2000,
    convergence_config: Optional[Dict[str, Any]] = None,
    stability_config: Optional[Dict[str, Any]] = None,
    curriculum_config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Train agent with Phase 0 optimizations.
    
    Args:
        agent: RL agent to train
        env: Traffic environment
        episodes: Number of training episodes
        convergence_config: Configuration for convergence monitor
        stability_config: Configuration for training stability framework
        output_dir: Output directory for saving results
    
    Returns:
        Training results dictionary
    """
    # Initialize convergence monitor
    convergence_config = convergence_config or {}
    convergence_monitor = ConvergenceMonitor(
        window=convergence_config.get("window", 100),
        threshold=convergence_config.get("threshold", 0.01),
        patience=convergence_config.get("patience", 500),
        min_episodes=convergence_config.get("min_episodes", 200),
        mode=convergence_config.get("mode", "maximize")
    )
    
    # Initialize performance tracker
    performance_tracker = PerformanceTracker(metrics=["reward", "loss"])
    
    # Initialize training stability framework (if agent has optimizer)
    stability_framework = None
    if hasattr(agent, 'optimizer') and hasattr(agent, 'policy_net'):
        stability_config = stability_config or {}
        stability_framework = TrainingStabilityFramework(
            optimizer=agent.optimizer,
            policy_net=agent.policy_net,
            target_net=getattr(agent, 'target_net', None),
            config=stability_config
        )
        logger.info("Training stability framework initialized")
    
    # Initialize curriculum learning (Phase 2.1)
    curriculum = None
    if curriculum_config is not None and curriculum_config.get("enabled", False):
        base_arrival_rates = env.cfg.get("arrival_rates", [0.3] * env.num_lanes)
        curriculum_type = curriculum_config.get("type", "standard")
        
        if curriculum_type == "adaptive":
            curriculum = AdaptiveCurriculum(
                base_arrival_rates=base_arrival_rates,
                initial_level=curriculum_config.get("initial_level", 0),
                progression_rate=curriculum_config.get("progression_rate", 0.1),
                regression_threshold=curriculum_config.get("regression_threshold", 0.3),
            )
        else:
            curriculum = TrafficCurriculum(
                base_arrival_rates=base_arrival_rates,
                performance_threshold=curriculum_config.get("performance_threshold", 0.7),
                min_episodes_per_level=curriculum_config.get("min_episodes_per_level", 50),
                performance_window=curriculum_config.get("performance_window", 100),
            )
        logger.info("Curriculum learning initialized")
    
    # Training loop
    episode_rewards = []
    episode_losses = []
    
    logger.info(f"Starting training with Phase 0 optimizations for {episodes} episodes")
    
    for episode in range(episodes):
        # Update environment with curriculum if enabled
        if curriculum is not None:
            config_update = curriculum.get_config_update()
            # Update environment arrival rates
            env.arrival_rates = np.array(config_update["arrival_rates"])
            env.cfg["arrival_rates"] = config_update["arrival_rates"]
        
        obs, info = env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        steps = 0
        
        done = False
        while not done:
            # Select action with exploration
            if stability_framework is not None:
                epsilon = stability_framework.get_exploration_rate()
                if epsilon is not None and np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(obs)
            else:
                # Check if select_action needs epsilon parameter
                import inspect
                sig = inspect.signature(agent.select_action)
                if 'epsilon' in sig.parameters:
                    action = agent.select_action(obs, epsilon=0.0)
                else:
                    action = agent.select_action(obs)
            
            # Store action for agents that need it (e.g., Transformer)
            if hasattr(agent, 'action_history'):
                if not hasattr(agent, 'action_history'):
                    agent.action_history = []
                agent.action_history.append(action)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            
            # Store experience and train
            if hasattr(agent, 'push'):
                agent.push(obs, action, reward, next_obs, done)
            
            # Handle different agent training interfaces
            if hasattr(agent, 'train_step'):
                # Check if train_step requires arguments (like TransformerAgent)
                import inspect
                sig = inspect.signature(agent.train_step)
                if len(sig.parameters) == 0:
                    # No-argument train_step (DQN-style)
                    loss = agent.train_step()
                    if loss is not None:
                        episode_loss += loss
                        
                        # Apply training stability
                        if stability_framework is not None:
                            stability_framework.clip_gradients()
                            stability_framework.update_target_network()
                else:
                    # train_step requires arguments - skip during episode (train at end)
                    pass
            
            obs = next_obs
        
        # Train agents that require batch training (e.g., Transformer)
        if hasattr(agent, 'train_step'):
            import inspect
            sig = inspect.signature(agent.train_step)
            if len(sig.parameters) > 0:
                # Agent requires training data - collect and train
                if hasattr(agent, 'state_history') and len(agent.state_history) > 0:
                    # For TransformerAgent, train on collected sequences
                    # This is a simplified training - in practice, would batch properly
                    try:
                        # Get recent states and actions for training
                        if hasattr(agent, 'action_history'):
                            recent_states = agent.state_history[-min(32, len(agent.state_history)):]
                            recent_actions = agent.action_history[-min(32, len(agent.action_history)):]
                            if len(recent_states) > 0 and len(recent_actions) > 0:
                                loss = agent.train_step(recent_states, recent_actions)
                                if loss is not None:
                                    episode_loss += loss
                    except Exception as e:
                        # Skip training if not enough data or incompatible interface
                        pass
        
        # Update statistics
        episode_rewards.append(episode_reward)
        if episode_loss > 0:
            episode_losses.append(episode_loss / steps)
        else:
            episode_losses.append(0.0)
        
        # Update convergence monitor
        convergence_status = convergence_monitor.update(episode_reward, episode)
        
        # Update curriculum learning
        if curriculum is not None:
            curriculum.update_performance(episode_reward, episode)
        
        # Update performance tracker
        performance_tracker.log(
            episode,
            reward=episode_reward,
            loss=episode_loss / max(1, steps)
        )
        
        # Step learning rate scheduler
        if stability_framework is not None:
            stability_framework.step_scheduler(metric=episode_reward)
            stability_framework.step_exploration()
        
        # Logging
        if (episode + 1) % 100 == 0:
            stats = convergence_monitor.get_statistics()
            log_msg = (
                f"Episode {episode + 1}/{episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Avg Reward: {np.mean(episode_rewards[-100:]):.2f} | "
                f"Best: {stats['best_reward']:.2f} | "
                f"No Improvement: {stats['no_improvement_count']}/{convergence_config.get('patience', 500)}"
            )
            
            if curriculum is not None:
                curriculum_stats = curriculum.get_statistics()
                log_msg += f" | Curriculum Level: {curriculum_stats['current_level']}/{curriculum_stats['total_levels']-1}"
            
            logger.info(log_msg)
            
            if stability_framework is not None:
                logger.info(f"  LR: {stability_framework.get_current_lr():.6f} | "
                          f"Epsilon: {stability_framework.get_exploration_rate():.3f}")
        
        # Early stopping
        if convergence_status["should_stop"]:
            logger.info(f"Early stopping triggered at episode {episode + 1}")
            logger.info(f"Best reward: {convergence_status['best_reward']:.2f} at episode {convergence_status['best_episode']}")
            break
    
    # Final statistics
    final_stats = convergence_monitor.get_statistics()
    performance_stats = {
        metric: performance_tracker.get_statistics(metric)
        for metric in performance_tracker.metrics
    }
    
    results = {
        "episodes": len(episode_rewards),
        "avg_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "final_reward": episode_rewards[-1] if episode_rewards else 0.0,
        "best_reward": final_stats["best_reward"],
        "best_episode": final_stats["best_episode"],
        "converged": final_stats["converged"],
        "episode_rewards": episode_rewards,
        "episode_losses": episode_losses,
        "performance_stats": performance_stats,
    }
    
    # Add curriculum statistics if used
    if curriculum is not None:
        results["curriculum_stats"] = curriculum.get_statistics()
    
    # Save results
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train with Phase 0 optimizations")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/intersection.json",
        help="Environment config file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2000,
        help="Number of training episodes"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./runs/optimized_training",
        help="Output directory"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="DQN",
        help="Agent type (DQN, Transformer, etc.)"
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable curriculum learning (Phase 2.1)"
    )
    parser.add_argument(
        "--curriculum-type",
        type=str,
        default="standard",
        choices=["standard", "adaptive"],
        help="Curriculum learning type"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create environment
    env = TrafficEnv(config=config)
    
    # Create agent (example with DQN)
    if args.agent == "DQN":
        from src.rl.pytorch_dqn import DQNAgent, DQNConfig
        agent_cfg = DQNConfig()
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            cfg=agent_cfg
        )
    else:
        raise ValueError(f"Agent type {args.agent} not yet supported in this script")
    
    # Configure optimizations
    convergence_config = {
        "window": 100,
        "threshold": 0.01,
        "patience": 500,
        "min_episodes": 200,
        "mode": "maximize"
    }
    
    stability_config = {
        "grad_clip_norm": 10.0,
        "lr_scheduler": {
            "enabled": True,
            "type": "cosine",
            "params": {"T_max": args.episodes, "eta_min": 1e-6}
        },
        "target_update_tau": 0.005,
        "use_soft_update": True,
        "exploration": {
            "enabled": True,
            "initial_epsilon": 1.0,
            "final_epsilon": 0.01,
            "decay_type": "linear",
            "decay_steps": args.episodes // 2
        }
    }
    
    # Configure curriculum learning (Phase 2.1)
    curriculum_config = {
        "enabled": args.curriculum,
        "type": args.curriculum_type,
        "performance_threshold": 0.7,
        "min_episodes_per_level": 50,
        "performance_window": 100,
    }
    
    # Train
    output_dir = Path(args.output)
    results = train_with_optimizations(
        agent=agent,
        env=env,
        episodes=args.episodes,
        convergence_config=convergence_config,
        stability_config=stability_config,
        curriculum_config=curriculum_config,
        output_dir=output_dir
    )
    
    logger.info("\n" + "="*80)
    logger.info("Training Complete!")
    logger.info("="*80)
    logger.info(f"Episodes: {results['episodes']}")
    logger.info(f"Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
    logger.info(f"Best Reward: {results['best_reward']:.2f} at episode {results['best_episode']}")
    logger.info(f"Converged: {results['converged']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

