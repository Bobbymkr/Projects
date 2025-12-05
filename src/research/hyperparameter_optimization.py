"""
Hyperparameter Optimization Framework.

Optuna-based hyperparameter optimization for systematic algorithm tuning
and performance improvement.
"""

import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import Optuna with graceful fallback
try:
    import optuna
    from optuna.samplers import TPESampler, NSGAIISampler
    from optuna.pruners import MedianPruner
    from optuna.multi_objective import Trial as MultiObjectiveTrial
    OPTUNA_AVAILABLE = True
    OPTUNA_MULTI_OBJECTIVE_AVAILABLE = hasattr(optuna, 'multi_objective')
except ImportError:
    OPTUNA_AVAILABLE = False
    OPTUNA_MULTI_OBJECTIVE_AVAILABLE = False
    logger.warning("Optuna not available. Hyperparameter optimization will be limited.")


@dataclass
class HyperparameterSpace:
    """Hyperparameter search space definition."""
    name: str
    param_type: str  # 'float', 'int', 'categorical'
    bounds: tuple  # (min, max) for float/int, list for categorical
    log_scale: bool = False  # For float parameters


class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna.
    
    Supports:
    - Bayesian optimization (TPE)
    - Random search
    - Grid search
    - Pruning for early stopping
    - Distributed optimization
    """
    
    def __init__(
        self,
        study_name: str = "adaptive-traffic-hyperopt",
        storage: Optional[str] = None,
        direction: str = "maximize",
        sampler: Optional[str] = "tpe",
    ):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            study_name: Name of the Optuna study
            storage: Storage backend (None for in-memory)
            direction: Optimization direction ('maximize' or 'minimize')
            sampler: Sampler type ('tpe', 'random', 'grid')
        """
        self.study_name = study_name
        self.storage = storage
        self.direction = direction
        self.enabled = OPTUNA_AVAILABLE
        
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Hyperparameter optimization disabled.")
            return
        
        # Create study
        try:
            sampler_obj = self._create_sampler(sampler)
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            
            self.study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                sampler=sampler_obj,
                pruner=pruner,
                direction=direction,
                load_if_exists=True,
            )
        except Exception as e:
            logger.warning(f"Failed to create Optuna study: {e}")
            self.enabled = False
    
    def _create_sampler(self, sampler_type: str):
        """Create Optuna sampler based on type."""
        if sampler_type == "tpe":
            return TPESampler(seed=42)
        elif sampler_type == "random":
            return optuna.samplers.RandomSampler(seed=42)
        elif sampler_type == "grid":
            return optuna.samplers.GridSampler()
        else:
            return TPESampler(seed=42)
    
    def optimize(
        self,
        objective_func: Callable,
        n_trials: int = 100,
        timeout: Optional[float] = None,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            objective_func: Function that takes a trial and returns a score
            n_trials: Number of optimization trials
            timeout: Maximum time in seconds
            show_progress: Whether to show progress bar
            
        Returns:
            Best parameters and value
        """
        if not self.enabled:
            logger.warning("Hyperparameter optimization disabled.")
            return {}
        
        try:
            self.study.optimize(
                objective_func,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=show_progress,
            )
            
            return {
                "best_params": self.study.best_params,
                "best_value": self.study.best_value,
                "n_trials": len(self.study.trials),
            }
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {e}")
            return {}
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters found so far."""
        if not self.enabled:
            return {}
        
        try:
            return self.study.best_params
        except Exception as e:
            logger.warning(f"Failed to get best parameters: {e}")
            return {}
    
    def get_trials(self) -> List[Dict[str, Any]]:
        """Get all trial results."""
        if not self.enabled:
            return []
        
        try:
            trials = []
            for trial in self.study.trials:
                trials.append({
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": trial.state.name,
                })
            return trials
        except Exception as e:
            logger.warning(f"Failed to get trials: {e}")
            return []
    
    def suggest_float(
        self,
        trial: Any,
        name: str,
        low: float,
        high: float,
        log: bool = False,
    ) -> float:
        """Suggest a float hyperparameter."""
        if not self.enabled:
            return (low + high) / 2
        
        try:
            return trial.suggest_float(name, low, high, log=log)
        except Exception as e:
            logger.warning(f"Failed to suggest float: {e}")
            return (low + high) / 2
    
    def suggest_int(
        self,
        trial: Any,
        name: str,
        low: int,
        high: int,
        log: bool = False,
    ) -> int:
        """Suggest an int hyperparameter."""
        if not self.enabled:
            return (low + high) // 2
        
        try:
            return trial.suggest_int(name, low, high, log=log)
        except Exception as e:
            logger.warning(f"Failed to suggest int: {e}")
            return (low + high) // 2
    
    def suggest_categorical(
        self,
        trial: Any,
        name: str,
        choices: List[Any],
    ) -> Any:
        """Suggest a categorical hyperparameter."""
        if not self.enabled:
            return choices[0]
        
        try:
            return trial.suggest_categorical(name, choices)
        except Exception as e:
            logger.warning(f"Failed to suggest categorical: {e}")
            return choices[0]


def create_dqn_optimization_objective(
    train_func: Callable,
    eval_func: Callable,
) -> Callable:
    """
    Create optimization objective for DQN hyperparameters.
    
    Args:
        train_func: Training function
        eval_func: Evaluation function
        
    Returns:
        Objective function for Optuna
    """
    def objective(trial: Any) -> float:
        """Optuna objective function."""
        # Suggest hyperparameters with expanded search space
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])
        gamma = trial.suggest_float("gamma", 0.90, 0.99)
        epsilon_start = trial.suggest_float("epsilon_start", 0.9, 1.0)
        epsilon_end = trial.suggest_float("epsilon_end", 0.01, 0.1)
        epsilon_decay = trial.suggest_int("epsilon_decay", 10000, 50000)
        replay_buffer_size = trial.suggest_int("replay_buffer_size", 10000, 100000, log=True)
        target_update = trial.suggest_int("target_update", 100, 2000)
        
        # Training stability hyperparameters (Phase 0.2)
        grad_clip_norm = trial.suggest_float("grad_clip_norm", 1.0, 20.0)
        lr_schedule = trial.suggest_categorical("lr_schedule", ["cosine", "linear", "constant"])
        soft_update_tau = trial.suggest_float("soft_update_tau", 0.001, 0.01)
        
        # Create config
        config = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gamma": gamma,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay": epsilon_decay,
            "replay_buffer_size": replay_buffer_size,
            "target_update": target_update,
            "grad_clip_norm": grad_clip_norm,
            "lr_schedule": lr_schedule,
            "soft_update_tau": soft_update_tau,
        }
        
        # Train and evaluate
        try:
            model = train_func(config)
            score = eval_func(model)
            return score
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return float('-inf') if trial.study.direction == "maximize" else float('inf')
    
    return objective


def create_multi_objective_dqn_objective(
    train_func: Callable,
    eval_func: Callable,
) -> Callable:
    """
    Create multi-objective optimization objective for DQN.
    
    Optimizes for:
    1. Performance (reward)
    2. Stability (variance)
    3. Efficiency (training time)
    
    Args:
        train_func: Training function that returns (model, metrics)
        eval_func: Evaluation function that returns (reward, variance, efficiency)
        
    Returns:
        Multi-objective function for Optuna
    """
    def objective(trial: Any) -> tuple:
        """Multi-objective Optuna function."""
        # Suggest hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])
        gamma = trial.suggest_float("gamma", 0.90, 0.99)
        epsilon_start = trial.suggest_float("epsilon_start", 0.9, 1.0)
        epsilon_end = trial.suggest_float("epsilon_end", 0.01, 0.1)
        replay_buffer_size = trial.suggest_int("replay_buffer_size", 10000, 100000, log=True)
        
        # Training stability
        grad_clip_norm = trial.suggest_float("grad_clip_norm", 1.0, 20.0)
        lr_schedule = trial.suggest_categorical("lr_schedule", ["cosine", "linear", "constant"])
        
        config = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gamma": gamma,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "replay_buffer_size": replay_buffer_size,
            "grad_clip_norm": grad_clip_norm,
            "lr_schedule": lr_schedule,
        }
        
        # Train and evaluate
        try:
            model, train_metrics = train_func(config)
            reward, variance, efficiency = eval_func(model)
            
            # Return tuple for multi-objective optimization
            # Maximize reward, minimize variance, maximize efficiency
            return reward, -variance, efficiency
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return float('-inf'), float('inf'), float('-inf')
    
    return objective

