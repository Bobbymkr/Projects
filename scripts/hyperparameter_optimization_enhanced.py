"""
Enhanced Hyperparameter Optimization Framework - Phase 1.

Implements:
- Multi-objective optimization (Pareto front)
- Integration with Phase 0 components
- Comprehensive hyperparameter spaces
- DQN and other algorithm support
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, Optional, List, Tuple
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler, NSGAIISampler
from optuna.visualization import plot_pareto_front, plot_optimization_history

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.traffic_env import TrafficEnv
from src.rl.training_stability import TrainingStabilityFramework
from src.rl.convergence_monitor import ConvergenceMonitor, PerformanceTracker
from scripts.train_with_optimization import train_with_optimizations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiObjectiveHyperparameterOptimizer:
    """
    Enhanced hyperparameter optimizer with multi-objective support.
    
    Implements Phase 1.1 from OPTIMIZATION_ROADMAP.md:
    - Multi-objective: Reward, stability, efficiency (Pareto front)
    - Integration with Phase 0 components
    - Comprehensive hyperparameter spaces
    """
    
    def __init__(
        self,
        algorithm_name: str,
        config_path: str,
        n_trials: int = 100,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        multi_objective: bool = True,
    ):
        self.algorithm_name = algorithm_name
        self.config_path = config_path
        self.n_trials = n_trials
        self.study_name = study_name or f"{algorithm_name}_optimization"
        self.multi_objective = multi_objective
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Create environment to get dimensions
        self.env = TrafficEnv(config=self.config)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        # Create study
        if multi_objective:
            # Multi-objective study with NSGA-II
            sampler = NSGAIISampler(population_size=20, seed=42)
            self.study = optuna.create_study(
                study_name=self.study_name,
                directions=["maximize", "maximize", "maximize"],  # reward, stability, efficiency
                sampler=sampler,
                storage=storage,
                load_if_exists=True,
            )
        else:
            # Single-objective study with TPE
            sampler = TPESampler(seed=42)
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            self.study = optuna.create_study(
                study_name=self.study_name,
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
                storage=storage,
                load_if_exists=True,
            )
    
    def suggest_dqn_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for DQN agent.
        
        Priority order from roadmap:
        1. Learning Rate (1e-5 to 1e-2, log-uniform)
        2. Discount Factor (0.90 to 0.99, uniform)
        3. Exploration Rate (ε-decay schedule)
        4. Batch Size (16, 32, 64, 128, 256)
        5. Network Architecture (hidden layers, neurons)
        6. Replay Buffer Size (1K to 100K)
        7. Target Update Frequency (1 to 1000 steps)
        """
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "gamma": trial.suggest_float("gamma", 0.90, 0.99),
            "eps_start": trial.suggest_float("eps_start", 0.9, 1.0),
            "eps_end": trial.suggest_float("eps_end", 0.01, 0.1),
            "eps_decay": trial.suggest_float("eps_decay", 0.990, 0.999),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
            "hidden_dim_1": trial.suggest_categorical("hidden_dim_1", [64, 128, 256, 512]),
            "hidden_dim_2": trial.suggest_categorical("hidden_dim_2", [64, 128, 256, 512]),
            "replay_buffer_size": trial.suggest_int("replay_buffer_size", 1000, 100000, log=True),
            "tau": trial.suggest_float("tau", 0.001, 0.01),  # Soft update coefficient
        }
    
    def suggest_stability_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for training stability framework."""
        return {
            "grad_clip_norm": trial.suggest_float("grad_clip_norm", 1.0, 20.0),
            "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["cosine", "cosine_warm_restart", "plateau"]),
            "lr_scheduler_T_max": trial.suggest_int("lr_scheduler_T_max", 500, 5000),
            "target_update_tau": trial.suggest_float("target_update_tau", 0.001, 0.01),
            "exploration_decay_type": trial.suggest_categorical("exploration_decay_type", ["linear", "exponential", "cosine"]),
        }
    
    def suggest_transformer_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for Transformer agent.
        
        Phase 1.2 from OPTIMIZATION_ROADMAP.md:
        - Attention Mechanism: Multi-head (2, 4, 8, 16)
        - Position Encoding: Learnable vs sinusoidal, relative vs absolute
        - Layer Normalization: Pre-norm vs post-norm
        - Feed-Forward: GELU vs ReLU, dimension scaling (1x, 2x, 4x)
        - Dropout: 0.0 to 0.5
        """
        return {
            "d_model": trial.suggest_categorical("d_model", [64, 128, 256, 512]),
            "nhead": trial.suggest_categorical("nhead", [2, 4, 8, 16]),
            "num_layers": trial.suggest_int("num_layers", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "ff_dim_scale": trial.suggest_categorical("ff_dim_scale", [1, 2, 4]),  # Feed-forward dimension scaling
            "position_encoding": trial.suggest_categorical("position_encoding", ["learnable", "sinusoidal"]),
            "layer_norm": trial.suggest_categorical("layer_norm", ["pre_norm", "post_norm"]),
            "activation": trial.suggest_categorical("activation", ["relu", "gelu"]),
        }
    
    def suggest_hrl_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for Hierarchical RL agent.
        
        Phase 1.2 from OPTIMIZATION_ROADMAP.md:
        - Option Discovery: Frequency (every 100-1000 episodes)
        - Termination Threshold: Adaptive based on option performance
        - Learning Rate Ratio: High-level / Low-level (0.1 to 10.0)
        - Option Count: 2 to 8 options
        """
        return {
            "option_count": trial.suggest_int("option_count", 2, 8),
            "option_discovery_frequency": trial.suggest_int("option_discovery_frequency", 100, 1000),
            "termination_threshold": trial.suggest_float("termination_threshold", 0.1, 0.9),
            "lr_ratio_high_low": trial.suggest_float("lr_ratio_high_low", 0.1, 10.0, log=True),
            "high_level_lr": trial.suggest_float("high_level_lr", 1e-5, 1e-2, log=True),
            "low_level_lr": trial.suggest_float("low_level_lr", 1e-5, 1e-2, log=True),
            "hidden_dims_high": trial.suggest_categorical("hidden_dims_high", [32, 64, 128, 256]),
            "hidden_dims_low": trial.suggest_categorical("hidden_dims_low", [32, 64, 128, 256]),
        }
    
    def suggest_mbrl_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for Model-Based RL agent.
        
        Phase 1.2 from OPTIMIZATION_ROADMAP.md:
        - World Model Architecture: Ensemble of 3-5 models
        - MPC Horizon: 5 to 30 steps
        - Planning Iterations: 10 to 100
        - Model Uncertainty: Quantile regression, ensemble disagreement
        """
        return {
            "world_model_ensemble_size": trial.suggest_int("world_model_ensemble_size", 3, 5),
            "world_model_hidden_dims": trial.suggest_categorical("world_model_hidden_dims", [64, 128, 256]),
            "world_model_lr": trial.suggest_float("world_model_lr", 1e-5, 1e-2, log=True),
            "mpc_horizon": trial.suggest_int("mpc_horizon", 5, 30),
            "mpc_planning_iterations": trial.suggest_int("mpc_planning_iterations", 10, 100),
            "mpc_num_candidates": trial.suggest_categorical("mpc_num_candidates", [50, 100, 200, 500]),
            "uncertainty_method": trial.suggest_categorical("uncertainty_method", ["ensemble", "quantile"]),
        }
    
    def create_dqn_agent(self, trial: optuna.Trial):
        """Create DQN agent with optimized hyperparameters."""
        from src.rl.pytorch_dqn import DQNAgent, DQNConfig, DQNetwork
        import torch.nn as nn
        
        params = self.suggest_dqn_hyperparameters(trial)
        
        # Create custom config
        cfg = DQNConfig()
        cfg.lr = params["learning_rate"]
        cfg.gamma = params["gamma"]
        cfg.eps_start = params["eps_start"]
        cfg.eps_end = params["eps_end"]
        cfg.eps_decay = params["eps_decay"]
        cfg.batch_size = params["batch_size"]
        cfg.tau = params["tau"]
        
        # Create agent with default network first
        agent = DQNAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            cfg=cfg
        )
        
        # Replace network with custom architecture
        class CustomDQNetwork(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dims):
                super().__init__()
                layers = []
                input_dim = state_dim
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(input_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    input_dim = hidden_dim
                layers.append(nn.Linear(input_dim, action_dim))
                self.net = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.net(x)
        
        # Create custom network with optimized architecture
        hidden_dims = [params["hidden_dim_1"], params["hidden_dim_2"]]
        custom_net = CustomDQNetwork(self.state_dim, self.action_dim, hidden_dims)
        agent.policy_net = custom_net.to(agent.cfg.device)
        agent.target_net = CustomDQNetwork(self.state_dim, self.action_dim, hidden_dims).to(agent.cfg.device)
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        
        # Update optimizer with new network
        import torch.optim as optim
        agent.optimizer = optim.Adam(agent.policy_net.parameters(), lr=cfg.lr)
        
        return agent, params
    
    def create_transformer_agent(self, trial: optuna.Trial):
        """Create Transformer agent with optimized hyperparameters."""
        from src.research.novel_algorithms.transformer_control import TransformerAgent
        
        params = self.suggest_transformer_hyperparameters(trial)
        
        agent = TransformerAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            d_model=params["d_model"],
            nhead=params["nhead"],
            num_layers=params["num_layers"],
            learning_rate=params["learning_rate"],
        )
        
        # Note: Additional hyperparameters (dropout, position encoding, layer norm, etc.)
        # would need to be passed if TransformerAgent supports them
        # For now, we optimize the core hyperparameters
        
        return agent, params
    
    def create_hrl_agent(self, trial: optuna.Trial):
        """Create Hierarchical RL agent with optimized hyperparameters."""
        from src.research.novel_algorithms.hierarchical_rl_complete import CompleteHierarchicalRLAgent
        
        params = self.suggest_hrl_hyperparameters(trial)
        
        # Create agent (options are created internally)
        agent = CompleteHierarchicalRLAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            use_domain_options=True,
        )
        
        # Store hyperparameters for potential use during training
        # Note: These would be used if the agent supports them
        agent.option_discovery_frequency = params["option_discovery_frequency"]
        agent.termination_threshold = params["termination_threshold"]
        agent.lr_ratio = params["lr_ratio_high_low"]
        agent.option_count = params["option_count"]
        
        return agent, params
    
    def create_mbrl_agent(self, trial: optuna.Trial):
        """Create Model-Based RL agent with optimized hyperparameters."""
        from src.research.novel_algorithms.model_based_rl_complete import CompleteModelBasedRLAgent
        
        params = self.suggest_mbrl_hyperparameters(trial)
        
        # Create world model config
        # Note: ensemble_size is stored but not passed to NeuralWorldModel (would need ensemble implementation)
        world_model_config = {
            "hidden_dims": [params["world_model_hidden_dims"], params["world_model_hidden_dims"]],
            "learning_rate": params["world_model_lr"],
        }
        
        # Create MPC config
        # Note: planning_iterations stored but not passed (would need MPC implementation support)
        mpc_config = {
            "horizon": params["mpc_horizon"],
            "num_candidates": params["mpc_num_candidates"],
        }
        
        agent = CompleteModelBasedRLAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            world_model_config=world_model_config,
            mpc_config=mpc_config,
        )
        
        # Store hyperparameters for potential use
        agent.uncertainty_method = params["uncertainty_method"]
        agent.ensemble_size = params["world_model_ensemble_size"]  # For future ensemble support
        agent.planning_iterations = params["mpc_planning_iterations"]  # For future MPC enhancement
        
        return agent, params
    
    def objective(self, trial: optuna.Trial):
        """
        Multi-objective objective function.
        
        Returns:
            Tuple of (reward, stability, efficiency) for multi-objective
            or single reward value for single-objective
        """
        try:
            # Create agent with trial hyperparameters
            if self.algorithm_name == "DQN":
                agent, dqn_params = self.create_dqn_agent(trial)
                algorithm_params = dqn_params
            elif self.algorithm_name == "Transformer":
                agent, transformer_params = self.create_transformer_agent(trial)
                algorithm_params = transformer_params
            elif self.algorithm_name == "Hierarchical RL":
                agent, hrl_params = self.create_hrl_agent(trial)
                algorithm_params = hrl_params
            elif self.algorithm_name == "Model-Based RL":
                agent, mbrl_params = self.create_mbrl_agent(trial)
                algorithm_params = mbrl_params
            else:
                raise ValueError(f"Algorithm {self.algorithm_name} not yet supported")
            
            # Get stability hyperparameters
            stability_params = self.suggest_stability_hyperparameters(trial)
            
            # Configure stability framework
            stability_config = {
                "grad_clip_norm": stability_params["grad_clip_norm"],
                "lr_scheduler": {
                    "enabled": True,
                    "type": stability_params["lr_scheduler_type"],
                    "params": {"T_max": stability_params["lr_scheduler_T_max"], "eta_min": 1e-6}
                },
                "target_update_tau": stability_params["target_update_tau"],
                "use_soft_update": True,
                "exploration": {
                    "enabled": True,
                    "initial_epsilon": algorithm_params.get("eps_start", 1.0),
                    "final_epsilon": algorithm_params.get("eps_end", 0.01),
                    "decay_type": stability_params["exploration_decay_type"],
                    "decay_steps": 5000  # Will be adjusted based on episodes
                }
            }
            
            # Configure convergence monitor
            convergence_config = {
                "window": 100,
                "threshold": 0.01,
                "patience": 500,
                "min_episodes": 200,
                "mode": "maximize"
            }
            
            # Train with limited episodes for optimization
            episodes = 200  # Reduced for faster optimization
            
            # Configure curriculum learning (Phase 2.1) - optional
            curriculum_config = None
            # Enable curriculum for longer training (disabled for quick optimization)
            # curriculum_config = {
            #     "enabled": True,
            #     "type": "standard",
            #     "performance_threshold": 0.7,
            #     "min_episodes_per_level": 20,
            # }
            
            # Train with Phase 0 and Phase 2 optimizations
            results = train_with_optimizations(
                agent=agent,
                env=self.env,
                episodes=episodes,
                convergence_config=convergence_config,
                stability_config=stability_config,
                curriculum_config=curriculum_config,
                output_dir=None  # Don't save during optimization
            )
            
            # Extract metrics
            avg_reward = results.get("avg_reward", -1000.0)
            std_reward = results.get("std_reward", 1000.0)
            best_reward = results.get("best_reward", -1000.0)
            
            # Compute multi-objective metrics
            # 1. Reward (primary objective)
            reward_metric = avg_reward
            
            # 2. Stability (lower std is better, so we negate it)
            stability_metric = -std_reward  # Maximize negative std = minimize std
            
            # 3. Efficiency (best reward achieved)
            efficiency_metric = best_reward
            
            # Report intermediate value for pruning (single-objective only)
            if not self.multi_objective:
                trial.report(avg_reward, step=episodes)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                return avg_reward
            else:
                # Multi-objective: return tuple
                return reward_metric, stability_metric, efficiency_metric
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial failed: {e}", exc_info=True)
            if self.multi_objective:
                return -1000.0, -1000.0, -1000.0
            else:
                return -1000.0
    
    def optimize(self) -> Dict[str, Any]:
        """Run optimization."""
        logger.info(f"Starting hyperparameter optimization for {self.algorithm_name}")
        logger.info(f"Number of trials: {self.n_trials}")
        logger.info(f"Multi-objective: {self.multi_objective}")
        
        self.study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        
        if self.multi_objective:
            # Get Pareto front
            pareto_trials = self.study.best_trials
            logger.info(f"\n{'='*80}")
            logger.info(f"Optimization Complete!")
            logger.info(f"{'='*80}")
            logger.info(f"Pareto Front Trials: {len(pareto_trials)}")
            
            # Show top 5 Pareto solutions
            for i, trial in enumerate(pareto_trials[:5]):
                logger.info(f"\nPareto Solution {i+1}:")
                logger.info(f"  Reward: {trial.values[0]:.2f}")
                logger.info(f"  Stability: {trial.values[1]:.2f}")
                logger.info(f"  Efficiency: {trial.values[2]:.2f}")
                logger.info(f"  Parameters:")
                for key, value in trial.params.items():
                    logger.info(f"    {key}: {value}")
            
            return {
                "pareto_trials": [
                    {
                        "values": trial.values,
                        "params": trial.params
                    }
                    for trial in pareto_trials
                ],
                "n_trials": len(self.study.trials),
                "study_name": self.study_name,
            }
        else:
            # Single-objective results
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
        
        if self.multi_objective:
            # Save Pareto front
            pareto_trials = self.study.best_trials
            pareto_data = [
                {
                    "values": trial.values,
                    "params": trial.params
                }
                for trial in pareto_trials
            ]
            
            with open(output_path, 'w') as f:
                json.dump(pareto_data, f, indent=2)
            
            logger.info(f"Pareto front saved to: {output_path}")
        else:
            # Save best parameters
            best_params = self.study.best_trial.params
            with open(output_path, 'w') as f:
                json.dump(best_params, f, indent=2)
            
            logger.info(f"Best parameters saved to: {output_path}")
        
        # Save full study
        study_path = output_path.parent / f"{self.study_name}_study.json"
        study_data = {
            "study_name": self.study_name,
            "n_trials": len(self.study.trials),
            "multi_objective": self.multi_objective,
        }
        
        if self.multi_objective:
            study_data["pareto_trials"] = [
                {
                    "values": trial.values,
                    "params": trial.params
                }
                for trial in self.study.best_trials
            ]
        else:
            study_data["best_value"] = self.study.best_trial.value
            study_data["best_params"] = self.study.best_trial.params
        
        with open(study_path, 'w') as f:
            json.dump(study_data, f, indent=2)
        
        logger.info(f"Study saved to: {study_path}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced hyperparameter optimization")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="DQN",
        choices=["DQN", "Transformer", "Hierarchical RL", "Model-Based RL", "Imitation Learning (BC)"],
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
        default=50,  # Reduced default for testing
        help="Number of optimization trials"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./runs/hyperparameter_optimization",
        help="Output directory"
    )
    parser.add_argument(
        "--multi-objective",
        action="store_true",
        help="Use multi-objective optimization (Pareto front)"
    )
    parser.add_argument(
        "--single-objective",
        action="store_true",
        help="Use single-objective optimization (default if not multi-objective)"
    )
    
    args = parser.parse_args()
    
    # Determine optimization mode
    multi_objective = args.multi_objective and not args.single_objective
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    optimizer = MultiObjectiveHyperparameterOptimizer(
        algorithm_name=args.algorithm,
        config_path=args.config,
        n_trials=args.trials,
        multi_objective=multi_objective,
    )
    
    results = optimizer.optimize()
    
    # Save results
    if multi_objective:
        output_path = output_dir / f"{args.algorithm.lower().replace(' ', '_')}_pareto_front.json"
    else:
        output_path = output_dir / f"{args.algorithm.lower().replace(' ', '_')}_best_params.json"
    
    optimizer.save_results(output_path)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

