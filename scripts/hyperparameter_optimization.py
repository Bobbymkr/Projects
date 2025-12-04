"""
Hyperparameter Optimization Framework.

Uses Optuna for automated hyperparameter tuning across all algorithms.
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.traffic_env import TrafficEnv
from scripts.train_all_technologies import train_technology

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """Optimize hyperparameters for traffic control algorithms."""
    
    def __init__(
        self,
        algorithm_name: str,
        config_path: str,
        n_trials: int = 100,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
    ):
        self.algorithm_name = algorithm_name
        self.config_path = config_path
        self.n_trials = n_trials
        self.study_name = study_name or f"{algorithm_name}_optimization"
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Create environment to get dimensions
        self.env = TrafficEnv(config=self.config)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        # Create study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize",  # Maximize average reward
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )
    
    def get_algorithm_factory(self, trial: optuna.Trial):
        """Get algorithm factory with trial hyperparameters."""
        if self.algorithm_name == "Transformer":
            return self._get_transformer_factory(trial)
        elif self.algorithm_name == "Hierarchical RL":
            return self._get_hrl_factory(trial)
        elif self.algorithm_name == "Model-Based RL":
            return self._get_mbrl_factory(trial)
        elif self.algorithm_name == "Imitation Learning (BC)":
            return self._get_il_factory(trial)
        else:
            # Default factory
            return lambda: self._create_default_agent(trial)
    
    def _get_transformer_factory(self, trial: optuna.Trial):
        """Get Transformer factory with optimized hyperparameters."""
        from src.research.novel_algorithms.transformer_control import TransformerAgent
        
        num_layers = trial.suggest_int("num_layers", 2, 8)
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8, 16])
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        dropout = trial.suggest_uniform("dropout", 0.0, 0.5)
        
        return lambda: TransformerAgent(
            self.state_dim,
            self.action_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            dropout=dropout,
        )
    
    def _get_hrl_factory(self, trial: optuna.Trial):
        """Get HRL factory with optimized hyperparameters."""
        from src.research.novel_algorithms.hierarchical_rl_complete import CompleteHierarchicalRLAgent
        
        hidden_dims_1 = trial.suggest_categorical("hidden_dims_1", [32, 64, 128, 256])
        hidden_dims_2 = trial.suggest_categorical("hidden_dims_2", [32, 64, 128, 256])
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        
        return lambda: CompleteHierarchicalRLAgent(
            self.state_dim,
            self.action_dim,
            use_domain_options=True,
        )
    
    def _get_mbrl_factory(self, trial: optuna.Trial):
        """Get MBRL factory with optimized hyperparameters."""
        from src.research.novel_algorithms.model_based_rl_complete import CompleteModelBasedRLAgent
        
        world_model_hidden = trial.suggest_categorical("world_model_hidden", [64, 128, 256])
        mpc_horizon = trial.suggest_int("mpc_horizon", 5, 30)
        mpc_candidates = trial.suggest_categorical("mpc_candidates", [50, 100, 200, 500])
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        
        return lambda: CompleteModelBasedRLAgent(
            self.state_dim,
            self.action_dim,
            world_model_config={
                "hidden_dims": [world_model_hidden, world_model_hidden],
                "learning_rate": learning_rate,
            },
            mpc_config={
                "horizon": mpc_horizon,
                "num_candidates": mpc_candidates,
            },
        )
    
    def _get_il_factory(self, trial: optuna.Trial):
        """Get Imitation Learning factory with optimized hyperparameters."""
        from src.research.novel_algorithms.imitation_learning import BehavioralCloningAgent
        
        hidden_dims_1 = trial.suggest_categorical("hidden_dims_1", [64, 128, 256])
        hidden_dims_2 = trial.suggest_categorical("hidden_dims_2", [64, 128, 256])
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        
        return lambda: BehavioralCloningAgent(
            self.state_dim,
            self.action_dim,
            hidden_dims=[hidden_dims_1, hidden_dims_2],
            learning_rate=learning_rate,
        )
    
    def _create_default_agent(self, trial: optuna.Trial):
        """Create default agent (fallback)."""
        # This would need to be implemented based on available algorithms
        raise NotImplementedError(f"Optimization for {self.algorithm_name} not yet implemented")
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization."""
        try:
            # Get algorithm factory with trial hyperparameters
            agent_factory = self.get_algorithm_factory(trial)
            agent = agent_factory()
            
            # Train with limited episodes for optimization
            episodes = 200  # Reduced for faster optimization
            
            result = train_technology(
                self.algorithm_name,
                agent,
                self.env,
                episodes=episodes,
                output_dir=None,  # Don't save during optimization
            )
            
            # Return average reward (higher is better)
            avg_reward = result.get("avg_reward", -1000.0)
            
            # Report intermediate value for pruning
            trial.report(avg_reward, step=episodes)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return avg_reward
            
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return -1000.0  # Return very bad score on failure
    
    def optimize(self) -> Dict[str, Any]:
        """Run optimization."""
        logger.info(f"Starting hyperparameter optimization for {self.algorithm_name}")
        logger.info(f"Number of trials: {self.n_trials}")
        
        self.study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        
        # Get best trial
        best_trial = self.study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Optimization Complete!")
        logger.info(f"{'='*80}")
        logger.info(f"Best Average Reward: {best_value:.2f}")
        logger.info(f"Best Parameters:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
        
        return {
            "best_value": best_value,
            "best_params": best_params,
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
        
        # Save study
        study_path = output_path.parent / f"{self.study_name}_study.json"
        optuna_study = {
            "study_name": self.study_name,
            "best_value": self.study.best_trial.value,
            "best_params": best_params,
            "n_trials": len(self.study.trials),
        }
        with open(study_path, 'w') as f:
            json.dump(optuna_study, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"Study saved to: {study_path}")


def main():
    parser = argparse.ArgumentParser(description="Optimize hyperparameters")
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["Transformer", "Hierarchical RL", "Model-Based RL", "Imitation Learning (BC)"],
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
        default=100,
        help="Number of optimization trials"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./runs/hyperparameter_optimization",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    optimizer = HyperparameterOptimizer(
        algorithm_name=args.algorithm,
        config_path=args.config,
        n_trials=args.trials,
    )
    
    results = optimizer.optimize()
    
    # Save results
    output_path = output_dir / f"{args.algorithm.lower().replace(' ', '_')}_best_params.json"
    optimizer.save_results(output_path)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

