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
        elif self.algorithm_name == "PPO":
            return self._get_ppo_factory(trial)
        elif self.algorithm_name == "SAC":
            return self._get_sac_factory(trial)
        elif self.algorithm_name == "Rainbow DQN":
            return self._get_rainbow_factory(trial)
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
    
    def _get_ppo_factory(self, trial: optuna.Trial):
        """Get PPO factory with optimized hyperparameters."""
        from src.research.novel_algorithms.phase6_advanced_rl import PPOAgent, PPOConfig
        
        # Hyperparameters from OPTIMIZATION_ROADMAP.md Phase 6.1
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
        gamma = trial.suggest_float("gamma", 0.90, 0.99)
        gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.99)
        clip_epsilon = trial.suggest_float("clip_epsilon", 0.1, 0.3)  # Conservative range
        value_coef = trial.suggest_float("value_coef", 0.1, 1.0)
        entropy_coef = trial.suggest_loguniform("entropy_coef", 1e-4, 1e-1)
        train_epochs = trial.suggest_int("train_epochs", 4, 10)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        buffer_size = trial.suggest_categorical("buffer_size", [1024, 2048, 4096])
        
        config = PPOConfig(
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            train_epochs=train_epochs,
            batch_size=batch_size,
            buffer_size=buffer_size,
        )
        
        return lambda: PPOAgent(self.state_dim, self.action_dim, config)
    
    def _get_sac_factory(self, trial: optuna.Trial):
        """Get SAC factory with optimized hyperparameters."""
        from src.research.novel_algorithms.phase6_advanced_rl import SACAgent, SACConfig
        
        # Hyperparameters from OPTIMIZATION_ROADMAP.md Phase 6.2
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
        gamma = trial.suggest_float("gamma", 0.90, 0.99)
        tau = trial.suggest_float("tau", 0.001, 0.01)  # Soft update coefficient
        alpha = trial.suggest_loguniform("alpha", 0.01, 1.0)  # Temperature parameter
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        buffer_size = trial.suggest_categorical("buffer_size", [50000, 100000, 200000])
        
        config = SACConfig(
            lr=lr,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            batch_size=batch_size,
            buffer_size=buffer_size,
        )
        
        return lambda: SACAgent(self.state_dim, self.action_dim, config)
    
    def _get_rainbow_factory(self, trial: optuna.Trial):
        """Get Rainbow DQN factory with optimized hyperparameters."""
        from src.research.novel_algorithms.phase6_advanced_rl import RainbowDQNAgent, RainbowDQNConfig
        
        # Hyperparameters from OPTIMIZATION_ROADMAP.md Phase 6.3
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
        gamma = trial.suggest_float("gamma", 0.90, 0.99)
        n_steps = trial.suggest_int("n_steps", 1, 5)  # Multi-step learning
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        buffer_size = trial.suggest_categorical("buffer_size", [50000, 100000, 200000])
        target_update_frequency = trial.suggest_categorical("target_update_frequency", [1000, 2000, 4000, 8000])
        eps_start = trial.suggest_float("eps_start", 0.9, 1.0)
        eps_end = trial.suggest_float("eps_end", 0.01, 0.1)
        eps_decay = trial.suggest_int("eps_decay", 10000, 50000)
        alpha = trial.suggest_float("alpha", 0.4, 0.8)  # PER priority exponent
        beta = trial.suggest_float("beta", 0.2, 0.6)  # PER importance sampling
        n_atoms = trial.suggest_categorical("n_atoms", [51, 101, 201])  # Distributional RL
        
        config = RainbowDQNConfig(
            lr=lr,
            gamma=gamma,
            n_steps=n_steps,
            batch_size=batch_size,
            buffer_size=buffer_size,
            target_update_frequency=target_update_frequency,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay=eps_decay,
            alpha=alpha,
            beta=beta,
            n_atoms=n_atoms,
        )
        
        return lambda: RainbowDQNAgent(self.state_dim, self.action_dim, config)
    
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
        choices=["Transformer", "Hierarchical RL", "Model-Based RL", "Imitation Learning (BC)", 
                 "PPO", "SAC", "Rainbow DQN"],
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

