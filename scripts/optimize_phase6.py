"""
Hyperparameter Optimization for Phase 6 Advanced RL Algorithms.

Optimizes hyperparameters for:
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- Rainbow DQN

Uses Optuna with TPE sampler and median pruner for efficient optimization.
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, Optional
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import optuna.visualization as vis

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.traffic_env import TrafficEnv
from scripts.train_all_technologies import train_technology

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phase6Optimizer:
    """Hyperparameter optimizer for Phase 6 algorithms."""
    
    def __init__(
        self,
        algorithm_name: str,
        config_path: str,
        n_trials: int = 50,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        episodes_per_trial: int = 100,  # Reduced for faster optimization
    ):
        self.algorithm_name = algorithm_name
        self.config_path = config_path
        self.n_trials = n_trials
        self.episodes_per_trial = episodes_per_trial
        self.study_name = study_name or f"phase6_{algorithm_name.lower().replace(' ', '_')}_optimization"
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Create environment to get dimensions
        self.env = TrafficEnv(config=self.config)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        # Create study with TPE sampler and median pruner
        sampler = TPESampler(seed=42, n_startup_trials=10)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=20)
        
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize",  # Maximize average reward
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )
    
    def get_ppo_config(self, trial: optuna.Trial):
        """Get optimized PPO configuration."""
        from src.research.novel_algorithms.phase6_advanced_rl import PPOConfig
        
        return PPOConfig(
            lr=trial.suggest_loguniform("lr", 1e-5, 1e-2),
            gamma=trial.suggest_float("gamma", 0.90, 0.99),
            gae_lambda=trial.suggest_float("gae_lambda", 0.90, 0.99),
            clip_epsilon=trial.suggest_float("clip_epsilon", 0.1, 0.3),
            value_coef=trial.suggest_float("value_coef", 0.1, 1.0),
            entropy_coef=trial.suggest_loguniform("entropy_coef", 1e-4, 1e-1),
            train_epochs=trial.suggest_int("train_epochs", 4, 10),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            buffer_size=trial.suggest_categorical("buffer_size", [1024, 2048, 4096]),
        )
    
    def get_sac_config(self, trial: optuna.Trial):
        """Get optimized SAC configuration."""
        from src.research.novel_algorithms.phase6_advanced_rl import SACConfig
        
        return SACConfig(
            lr=trial.suggest_loguniform("lr", 1e-5, 1e-2),
            gamma=trial.suggest_float("gamma", 0.90, 0.99),
            tau=trial.suggest_float("tau", 0.001, 0.01),
            alpha=trial.suggest_loguniform("alpha", 0.01, 1.0),
            batch_size=trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            buffer_size=trial.suggest_categorical("buffer_size", [50000, 100000, 200000]),
        )
    
    def get_rainbow_config(self, trial: optuna.Trial):
        """Get optimized Rainbow DQN configuration."""
        from src.research.novel_algorithms.phase6_advanced_rl import RainbowDQNConfig
        
        return RainbowDQNConfig(
            lr=trial.suggest_loguniform("lr", 1e-5, 1e-3),
            gamma=trial.suggest_float("gamma", 0.90, 0.99),
            n_steps=trial.suggest_int("n_steps", 1, 5),
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            buffer_size=trial.suggest_categorical("buffer_size", [50000, 100000, 200000]),
            target_update_frequency=trial.suggest_categorical("target_update_frequency", [1000, 2000, 4000, 8000]),
            eps_start=trial.suggest_float("eps_start", 0.9, 1.0),
            eps_end=trial.suggest_float("eps_end", 0.01, 0.1),
            eps_decay=trial.suggest_int("eps_decay", 10000, 50000),
            alpha=trial.suggest_float("alpha", 0.4, 0.8),
            beta=trial.suggest_float("beta", 0.2, 0.6),
            n_atoms=trial.suggest_categorical("n_atoms", [51, 101, 201]),
        )
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization."""
        try:
            # Create agent with trial hyperparameters
            if self.algorithm_name == "PPO":
                from src.research.novel_algorithms.phase6_advanced_rl import PPOAgent
                config = self.get_ppo_config(trial)
                agent = PPOAgent(self.state_dim, self.action_dim, config)
            elif self.algorithm_name == "SAC":
                from src.research.novel_algorithms.phase6_advanced_rl import SACAgent
                config = self.get_sac_config(trial)
                agent = SACAgent(self.state_dim, self.action_dim, config)
            elif self.algorithm_name == "Rainbow DQN":
                from src.research.novel_algorithms.phase6_advanced_rl import RainbowDQNAgent
                config = self.get_rainbow_config(trial)
                agent = RainbowDQNAgent(self.state_dim, self.action_dim, config)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm_name}")
            
            # Train with limited episodes for optimization
            result = train_technology(
                self.algorithm_name,
                agent,
                self.env,
                episodes=self.episodes_per_trial,
                output_dir=None,  # Don't save during optimization
            )
            
            # Return average reward (higher is better)
            avg_reward = result.get("avg_reward", -1000.0)
            
            # Report intermediate value for pruning
            trial.report(avg_reward, step=self.episodes_per_trial)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return avg_reward
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return -1000.0  # Return very bad score on failure
    
    def optimize(self) -> Dict[str, Any]:
        """Run optimization."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting hyperparameter optimization for {self.algorithm_name}")
        logger.info(f"{'='*80}")
        logger.info(f"Number of trials: {self.n_trials}")
        logger.info(f"Episodes per trial: {self.episodes_per_trial}")
        logger.info(f"Study name: {self.study_name}")
        logger.info(f"{'='*80}\n")
        
        self.study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        
        # Get best trial
        best_trial = self.study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Optimization Complete!")
        logger.info(f"{'='*80}")
        logger.info(f"Best Average Reward: {best_value:.2f}")
        logger.info(f"Best Trial Number: {best_trial.number}")
        logger.info(f"\nBest Parameters:")
        for key, value in sorted(best_params.items()):
            logger.info(f"  {key}: {value}")
        logger.info(f"{'='*80}\n")
        
        return {
            "best_value": best_value,
            "best_params": best_params,
            "best_trial": best_trial.number,
            "n_trials": len(self.study.trials),
            "study_name": self.study_name,
        }
    
    def save_results(self, output_path: Path):
        """Save optimization results."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save best parameters
        best_params = self.study.best_trial.params
        with open(output_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        # Save full study results
        study_path = output_path.parent / f"{self.study_name}_study.json"
        study_data = {
            "study_name": self.study_name,
            "algorithm": self.algorithm_name,
            "best_value": self.study.best_trial.value,
            "best_trial": self.study.best_trial.number,
            "best_params": best_params,
            "n_trials": len(self.study.trials),
            "all_trials": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "state": t.state.name,
                }
                for t in self.study.trials
            ],
        }
        with open(study_path, 'w') as f:
            json.dump(study_data, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"Study saved to: {study_path}")
    
    def visualize(self, output_dir: Path):
        """Generate visualization plots."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Optimization history
            fig = vis.plot_optimization_history(self.study)
            fig.write_image(str(output_dir / "optimization_history.png"))
            
            # Parameter importance
            try:
                fig = vis.plot_param_importances(self.study)
                fig.write_image(str(output_dir / "param_importances.png"))
            except Exception as e:
                logger.warning(f"Could not generate param importance plot: {e}")
            
            # Parallel coordinate plot
            try:
                fig = vis.plot_parallel_coordinate(self.study)
                fig.write_image(str(output_dir / "parallel_coordinate.png"))
            except Exception as e:
                logger.warning(f"Could not generate parallel coordinate plot: {e}")
            
            logger.info(f"Visualizations saved to: {output_dir}")
        except Exception as e:
            logger.warning(f"Could not generate visualizations: {e}")


def main():
    parser = argparse.ArgumentParser(description="Optimize Phase 6 algorithm hyperparameters")
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["PPO", "SAC", "Rainbow DQN"],
        help="Algorithm to optimize"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/intersection.json",
        help="Config file"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of optimization trials"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes per trial"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./runs/phase6_optimization",
        help="Output directory"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    optimizer = Phase6Optimizer(
        algorithm_name=args.algorithm,
        config_path=args.config,
        n_trials=args.trials,
        episodes_per_trial=args.episodes,
    )
    
    results = optimizer.optimize()
    
    # Save results
    output_path = output_dir / f"{args.algorithm.lower().replace(' ', '_')}_best_params.json"
    optimizer.save_results(output_path)
    
    # Generate visualizations
    if args.visualize:
        optimizer.visualize(output_dir / "visualizations")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

