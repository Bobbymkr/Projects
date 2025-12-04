"""
Full-Scale Training Experiments for Phase 6 Advanced RL Algorithms.

Trains PPO, SAC, and Rainbow DQN with comprehensive logging, monitoring,
and result analysis. Designed for production-scale experiments.
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
import time
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.traffic_env import TrafficEnv
from src.research.novel_algorithms.phase6_advanced_rl import (
    PPOAgent, PPOConfig,
    SACAgent, SACConfig,
    RainbowDQNAgent, RainbowDQNConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase6_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Phase6Trainer:
    """Full-scale trainer for Phase 6 algorithms."""
    
    def __init__(
        self,
        config_path: str,
        output_dir: Path,
        episodes: int = 1000,
        eval_interval: int = 100,
        save_interval: int = 200,
        use_optimized_params: bool = False,
        optimized_params_path: Optional[str] = None,
    ):
        """
        Initialize Phase 6 trainer.
        
        Args:
            config_path: Path to environment config
            output_dir: Output directory for results
            episodes: Number of training episodes
            eval_interval: Episodes between evaluations
            save_interval: Episodes between model saves
            use_optimized_params: Whether to use optimized hyperparameters
            optimized_params_path: Path to optimized parameters JSON
        """
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.episodes = episodes
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.use_optimized_params = use_optimized_params
        
        # Load environment config
        with open(config_path, 'r') as f:
            self.env_config = json.load(f)
        
        # Create environment
        self.env = TrafficEnv(config=self.env_config)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        # Load optimized parameters if provided
        self.optimized_params = {}
        if use_optimized_params and optimized_params_path:
            with open(optimized_params_path, 'r') as f:
                self.optimized_params = json.load(f)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        
        # Training results storage
        self.results = {}
    
    def get_ppo_config(self) -> PPOConfig:
        """Get PPO configuration (optimized or default)."""
        if self.use_optimized_params and "PPO" in self.optimized_params:
            params = self.optimized_params["PPO"]
            return PPOConfig(
                lr=params.get("lr", 3e-4),
                gamma=params.get("gamma", 0.99),
                gae_lambda=params.get("gae_lambda", 0.95),
                clip_epsilon=params.get("clip_epsilon", 0.2),
                value_coef=params.get("value_coef", 0.5),
                entropy_coef=params.get("entropy_coef", 0.01),
                train_epochs=params.get("train_epochs", 4),
                batch_size=params.get("batch_size", 64),
                buffer_size=params.get("buffer_size", 2048),
            )
        else:
            # Default configuration from OPTIMIZATION_ROADMAP.md
            return PPOConfig(
                lr=3e-4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_epsilon=0.2,
                value_coef=0.5,
                entropy_coef=0.01,
                train_epochs=4,
                batch_size=64,
                buffer_size=2048,
            )
    
    def get_sac_config(self) -> SACConfig:
        """Get SAC configuration (optimized or default)."""
        if self.use_optimized_params and "SAC" in self.optimized_params:
            params = self.optimized_params["SAC"]
            return SACConfig(
                lr=params.get("lr", 3e-4),
                gamma=params.get("gamma", 0.99),
                tau=params.get("tau", 0.005),
                alpha=params.get("alpha", 0.2),
                batch_size=params.get("batch_size", 256),
                buffer_size=params.get("buffer_size", 100000),
            )
        else:
            # Default configuration from OPTIMIZATION_ROADMAP.md
            return SACConfig(
                lr=3e-4,
                gamma=0.99,
                tau=0.005,
                alpha=0.2,
                batch_size=256,
                buffer_size=100000,
            )
    
    def get_rainbow_config(self) -> RainbowDQNConfig:
        """Get Rainbow DQN configuration (optimized or default)."""
        if self.use_optimized_params and "Rainbow DQN" in self.optimized_params:
            params = self.optimized_params["Rainbow DQN"]
            return RainbowDQNConfig(
                lr=params.get("lr", 6.25e-5),
                gamma=params.get("gamma", 0.99),
                n_steps=params.get("n_steps", 3),
                batch_size=params.get("batch_size", 32),
                buffer_size=params.get("buffer_size", 100000),
                target_update_frequency=params.get("target_update_frequency", 8000),
                eps_start=params.get("eps_start", 1.0),
                eps_end=params.get("eps_end", 0.01),
                eps_decay=params.get("eps_decay", 25000),
                alpha=params.get("alpha", 0.6),
                beta=params.get("beta", 0.4),
                n_atoms=params.get("n_atoms", 51),
            )
        else:
            # Default configuration from OPTIMIZATION_ROADMAP.md
            return RainbowDQNConfig(
                lr=6.25e-5,
                gamma=0.99,
                n_steps=3,
                batch_size=32,
                buffer_size=100000,
                target_update_frequency=8000,
                eps_start=1.0,
                eps_end=0.01,
                eps_decay=25000,
                alpha=0.6,
                beta=0.4,
                n_atoms=51,
            )
    
    def train_algorithm(
        self,
        algorithm_name: str,
        agent,
        episodes: int,
    ) -> Dict[str, Any]:
        """
        Train a single algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            agent: Agent instance
            episodes: Number of episodes to train
            
        Returns:
            Training results dictionary
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Training: {algorithm_name}")
        logger.info(f"{'='*80}")
        logger.info(f"Episodes: {episodes}")
        logger.info(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        logger.info(f"{'='*80}\n")
        
        episode_rewards = []
        episode_lengths = []
        training_losses = []
        eval_rewards = []
        
        start_time = time.time()
        
        try:
            for episode in range(episodes):
                obs, info = self.env.reset()
                obs = np.array(obs, dtype=np.float32).flatten()
                episode_reward = 0.0
                episode_length = 0
                done = False
                
                # Episode loop
                while not done:
                    # Select action
                    if algorithm_name == "PPO":
                        result = agent.select_action(obs, deterministic=False)
                        if isinstance(result, tuple):
                            action = result[0]
                        else:
                            action = result
                    else:
                        action = agent.select_action(obs, evaluate=False)
                    
                    # Ensure valid action
                    action = int(action)
                    if action < 0 or action >= self.env.action_space.n:
                        action = 0
                    
                    # Step environment
                    next_obs, reward, terminated, truncated, step_info = self.env.step(action)
                    done = terminated or truncated
                    next_obs = np.array(next_obs, dtype=np.float32).flatten()
                    
                    # Store experience
                    agent.push(obs, action, reward, next_obs, done)
                    
                    # Train (algorithm-specific)
                    if algorithm_name == "PPO":
                        # PPO trains on full episodes
                        if done and len(agent.buffer['states']) >= agent.config.batch_size:
                            metrics = agent.train_step()
                            if metrics:
                                training_losses.append(metrics.get('loss', 0.0))
                    else:
                        # SAC and Rainbow DQN train periodically
                        if len(agent.buffer) >= getattr(agent.config, 'batch_size', 32):
                            if episode % 10 == 0:  # Train every 10 episodes
                                metrics = agent.train_step()
                                if metrics:
                                    training_losses.append(metrics.get('loss', 0.0))
                    
                    episode_reward += reward
                    episode_length += 1
                    obs = next_obs
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Evaluation
                if (episode + 1) % self.eval_interval == 0:
                    eval_reward = self.evaluate(agent, algorithm_name, n_episodes=10)
                    eval_rewards.append(eval_reward)
                    logger.info(
                        f"Episode {episode + 1}/{episodes} | "
                        f"Avg Reward: {np.mean(episode_rewards[-self.eval_interval:]):.2f} | "
                        f"Eval Reward: {eval_reward:.2f}"
                    )
                
                # Save checkpoint
                if (episode + 1) % self.save_interval == 0:
                    self.save_checkpoint(agent, algorithm_name, episode + 1)
            
            training_time = time.time() - start_time
            
            # Final evaluation
            final_eval_reward = self.evaluate(agent, algorithm_name, n_episodes=50)
            
            results = {
                "algorithm": algorithm_name,
                "episodes": episodes,
                "avg_reward": float(np.mean(episode_rewards)),
                "std_reward": float(np.std(episode_rewards)),
                "final_reward": float(np.mean(episode_rewards[-100:])),
                "best_reward": float(np.max(episode_rewards)),
                "avg_length": float(np.mean(episode_lengths)),
                "eval_rewards": [float(r) for r in eval_rewards],
                "final_eval_reward": float(final_eval_reward),
                "training_losses": [float(l) for l in training_losses[-1000:]],  # Last 1000
                "training_time": training_time,
                "episode_rewards": [float(r) for r in episode_rewards],
                "episode_lengths": [int(l) for l in episode_lengths],
            }
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Training Complete: {algorithm_name}")
            logger.info(f"{'='*80}")
            logger.info(f"Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
            logger.info(f"Final Reward: {results['final_reward']:.2f}")
            logger.info(f"Best Reward: {results['best_reward']:.2f}")
            logger.info(f"Final Eval Reward: {results['final_eval_reward']:.2f}")
            logger.info(f"Training Time: {training_time:.2f}s")
            logger.info(f"{'='*80}\n")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training {algorithm_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "algorithm": algorithm_name,
                "status": "error",
                "error": str(e),
            }
    
    def evaluate(self, agent, algorithm_name: str, n_episodes: int = 10) -> float:
        """Evaluate agent performance."""
        eval_rewards = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            obs = np.array(obs, dtype=np.float32).flatten()
            done = False
            episode_reward = 0.0
            
            while not done:
                if algorithm_name == "PPO":
                    result = agent.select_action(obs, deterministic=True)
                    if isinstance(result, tuple):
                        action = result[0]
                    else:
                        action = result
                else:
                    action = agent.select_action(obs, evaluate=True)
                
                action = int(action)
                if action < 0 or action >= self.env.action_space.n:
                    action = 0
                
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_obs = np.array(next_obs, dtype=np.float32).flatten()
                
                episode_reward += reward
                obs = next_obs
            
            eval_rewards.append(episode_reward)
        
        return np.mean(eval_rewards)
    
    def save_checkpoint(self, agent, algorithm_name: str, episode: int):
        """Save model checkpoint."""
        try:
            checkpoint_dir = self.output_dir / "models" / algorithm_name.lower().replace(' ', '_')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save PyTorch models
            if hasattr(agent, 'policy_net'):
                torch.save(agent.policy_net.state_dict(), 
                          checkpoint_dir / f"policy_net_ep{episode}.pt")
            if hasattr(agent, 'value_net'):
                torch.save(agent.value_net.state_dict(), 
                          checkpoint_dir / f"value_net_ep{episode}.pt")
            if hasattr(agent, 'actor'):
                torch.save(agent.actor.state_dict(), 
                          checkpoint_dir / f"actor_ep{episode}.pt")
            if hasattr(agent, 'critic1'):
                torch.save(agent.critic1.state_dict(), 
                          checkpoint_dir / f"critic1_ep{episode}.pt")
            if hasattr(agent, 'critic2'):
                torch.save(agent.critic2.state_dict(), 
                          checkpoint_dir / f"critic2_ep{episode}.pt")
            if hasattr(agent, 'q_net'):
                torch.save(agent.q_net.state_dict(), 
                          checkpoint_dir / f"q_net_ep{episode}.pt")
            if hasattr(agent, 'target_net'):
                torch.save(agent.target_net.state_dict(), 
                          checkpoint_dir / f"target_net_ep{episode}.pt")
        except Exception as e:
            logger.warning(f"Could not save checkpoint: {e}")
    
    def train_all(self) -> Dict[str, Any]:
        """Train all Phase 6 algorithms."""
        logger.info(f"\n{'='*80}")
        logger.info("Phase 6 Full-Scale Training Experiments")
        logger.info(f"{'='*80}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Episodes per algorithm: {self.episodes}")
        logger.info(f"Using optimized params: {self.use_optimized_params}")
        logger.info(f"{'='*80}\n")
        
        algorithms = {
            "PPO": lambda: PPOAgent(self.state_dim, self.action_dim, self.get_ppo_config()),
            "SAC": lambda: SACAgent(self.state_dim, self.action_dim, self.get_sac_config()),
            "Rainbow DQN": lambda: RainbowDQNAgent(self.state_dim, self.action_dim, self.get_rainbow_config()),
        }
        
        all_results = {}
        
        for alg_name, agent_factory in algorithms.items():
            try:
                agent = agent_factory()
                results = self.train_algorithm(alg_name, agent, self.episodes)
                all_results[alg_name] = results
                
                # Save final model
                self.save_checkpoint(agent, alg_name, self.episodes)
                
            except Exception as e:
                logger.error(f"Failed to train {alg_name}: {e}")
                all_results[alg_name] = {"status": "error", "error": str(e)}
        
        # Save all results
        results_path = self.output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Generate summary
        self.generate_summary(all_results)
        
        return all_results
    
    def generate_summary(self, results: Dict[str, Any]):
        """Generate training summary."""
        summary_path = self.output_dir / "training_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Phase 6 Full-Scale Training Summary\n")
            f.write("="*80 + "\n\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Episodes per algorithm: {self.episodes}\n")
            f.write(f"Using optimized params: {self.use_optimized_params}\n\n")
            
            f.write("="*80 + "\n")
            f.write("Results\n")
            f.write("="*80 + "\n\n")
            
            for alg_name, result in results.items():
                if result.get("status") == "error":
                    f.write(f"{alg_name}: ERROR - {result.get('error', 'Unknown error')}\n\n")
                else:
                    f.write(f"{alg_name}:\n")
                    f.write(f"  Average Reward: {result.get('avg_reward', 0):.2f} ± {result.get('std_reward', 0):.2f}\n")
                    f.write(f"  Final Reward: {result.get('final_reward', 0):.2f}\n")
                    f.write(f"  Best Reward: {result.get('best_reward', 0):.2f}\n")
                    f.write(f"  Final Eval Reward: {result.get('final_eval_reward', 0):.2f}\n")
                    f.write(f"  Training Time: {result.get('training_time', 0):.2f}s\n")
                    f.write(f"  Episodes: {result.get('episodes', 0)}\n\n")
            
            # Comparison
            f.write("="*80 + "\n")
            f.write("Algorithm Comparison\n")
            f.write("="*80 + "\n\n")
            
            valid_results = {k: v for k, v in results.items() if v.get("status") != "error"}
            if valid_results:
                best_avg = max(valid_results.items(), key=lambda x: x[1].get('avg_reward', -1000))
                best_final = max(valid_results.items(), key=lambda x: x[1].get('final_reward', -1000))
                best_eval = max(valid_results.items(), key=lambda x: x[1].get('final_eval_reward', -1000))
                
                f.write(f"Best Average Reward: {best_avg[0]} ({best_avg[1].get('avg_reward', 0):.2f})\n")
                f.write(f"Best Final Reward: {best_final[0]} ({best_final[1].get('final_reward', 0):.2f})\n")
                f.write(f"Best Eval Reward: {best_eval[0]} ({best_eval[1].get('final_eval_reward', 0):.2f})\n")
        
        logger.info(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Full-scale Phase 6 training experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/intersection.json",
        help="Environment config file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of episodes per algorithm"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./runs/phase6_full_scale",
        help="Output directory"
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="Episodes between evaluations"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=200,
        help="Episodes between model saves"
    )
    parser.add_argument(
        "--use-optimized",
        action="store_true",
        help="Use optimized hyperparameters"
    )
    parser.add_argument(
        "--optimized-params",
        type=str,
        default=None,
        help="Path to optimized parameters JSON file"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["PPO", "SAC", "Rainbow DQN", "all"],
        default="all",
        help="Algorithm to train (default: all)"
    )
    
    args = parser.parse_args()
    
    trainer = Phase6Trainer(
        config_path=args.config,
        output_dir=Path(args.output),
        episodes=args.episodes,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        use_optimized_params=args.use_optimized,
        optimized_params_path=args.optimized_params,
    )
    
    if args.algorithm == "all":
        results = trainer.train_all()
    else:
        # Train single algorithm
        if args.algorithm == "PPO":
            agent = PPOAgent(trainer.state_dim, trainer.action_dim, trainer.get_ppo_config())
        elif args.algorithm == "SAC":
            agent = SACAgent(trainer.state_dim, trainer.action_dim, trainer.get_sac_config())
        elif args.algorithm == "Rainbow DQN":
            agent = RainbowDQNAgent(trainer.state_dim, trainer.action_dim, trainer.get_rainbow_config())
        
        results = {args.algorithm: trainer.train_algorithm(args.algorithm, agent, args.episodes)}
        trainer.save_checkpoint(agent, args.algorithm, args.episodes)
        
        # Save results
        results_path = trainer.output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        trainer.generate_summary(results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

