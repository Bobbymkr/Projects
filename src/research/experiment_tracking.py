"""
Experiment Tracking System.

MLflow-based experiment tracking for systematic algorithm research,
hyperparameter optimization, and experiment comparison.
"""

import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Try to import MLflow with graceful fallback
try:
    import mlflow
    import mlflow.pytorch
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Experiment tracking will be limited.")


class ExperimentTracker:
    """
    Experiment tracking manager using MLflow.
    
    Provides comprehensive experiment tracking including:
    - Parameters and hyperparameters
    - Metrics and performance measures
    - Model artifacts
    - Code versions
    - Environment information
    """
    
    def __init__(
        self,
        experiment_name: str = "adaptive-traffic-research",
        tracking_uri: Optional[str] = None,
        enabled: bool = True,
    ):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI (default: local file store)
            enabled: Whether tracking is enabled
        """
        self.experiment_name = experiment_name
        self.enabled = enabled and MLFLOW_AVAILABLE
        
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available. Experiment tracking disabled.")
            return
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Default: local file store
            default_uri = str(Path(__file__).parent.parent.parent / "mlruns")
            os.makedirs(default_uri, exist_ok=True)
            mlflow.set_tracking_uri(f"file://{default_uri}")
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"Could not set up MLflow experiment: {e}")
            self.enabled = False
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Optional[Any]:
        """
        Start a new experiment run.
        
        Args:
            run_name: Optional name for this run
            tags: Optional tags for categorization
            **kwargs: Additional MLflow run parameters
            
        Returns:
            MLflow run object (or None if tracking disabled)
        """
        if not self.enabled:
            return None
        
        try:
            tags = tags or {}
            tags.update({
                "phase": "phase5",
                "component": "research",
                "timestamp": datetime.now().isoformat(),
            })
            
            return mlflow.start_run(run_name=run_name, tags=tags, **kwargs)
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            return None
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters for current run."""
        if not self.enabled:
            return
        
        try:
            mlflow.log_params(params)
        except Exception as e:
            logger.warning(f"Failed to log parameters: {e}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """
        Log metrics for current run.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for time-series metrics
        """
        if not self.enabled:
            return
        
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")
    
    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        **kwargs: Any,
    ) -> None:
        """
        Log a model artifact.
        
        Args:
            model: Model object to log
            artifact_path: Path within artifacts
            **kwargs: Additional MLflow log_model parameters
        """
        if not self.enabled:
            return
        
        try:
            # Try PyTorch model logging
            if hasattr(model, 'state_dict'):
                mlflow.pytorch.log_model(model, artifact_path, **kwargs)
            # Try sklearn model logging
            elif hasattr(model, 'predict'):
                mlflow.sklearn.log_model(model, artifact_path, **kwargs)
            else:
                logger.warning(f"Unknown model type, skipping model logging")
        except Exception as e:
            logger.warning(f"Failed to log model: {e}")
    
    def log_artifacts(
        self,
        local_dir: str,
        artifact_path: Optional[str] = None,
    ) -> None:
        """Log directory of artifacts."""
        if not self.enabled:
            return
        
        try:
            mlflow.log_artifacts(local_dir, artifact_path)
        except Exception as e:
            logger.warning(f"Failed to log artifacts: {e}")
    
    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None,
    ) -> None:
        """Log a single artifact file."""
        if not self.enabled:
            return
        
        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            logger.warning(f"Failed to log artifact: {e}")
    
    def end_run(self) -> None:
        """End the current run."""
        if not self.enabled:
            return
        
        try:
            mlflow.end_run()
        except Exception as e:
            logger.warning(f"Failed to end run: {e}")
    
    def search_runs(
        self,
        filter_string: Optional[str] = None,
        max_results: int = 100,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Search for previous runs.
        
        Args:
            filter_string: MLflow filter string
            max_results: Maximum number of results
            **kwargs: Additional search parameters
            
        Returns:
            List of run information dictionaries
        """
        if not self.enabled:
            return []
        
        try:
            runs = mlflow.search_runs(
                experiment_names=[self.experiment_name],
                filter_string=filter_string,
                max_results=max_results,
                **kwargs,
            )
            return runs.to_dict('records')
        except Exception as e:
            logger.warning(f"Failed to search runs: {e}")
            return []
    
    def compare_runs(
        self,
        run_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            
        Returns:
            Comparison dictionary with metrics, parameters, etc.
        """
        if not self.enabled:
            return {}
        
        try:
            comparison = {}
            for run_id in run_ids:
                run = mlflow.get_run(run_id)
                comparison[run_id] = {
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags,
                }
            return comparison
        except Exception as e:
            logger.warning(f"Failed to compare runs: {e}")
            return {}


class ExperimentConfig:
    """Configuration for a single experiment."""
    
    def __init__(
        self,
        algorithm: str,
        hyperparameters: Dict[str, Any],
        dataset: str,
        description: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize experiment configuration."""
        self.algorithm = algorithm
        self.hyperparameters = hyperparameters
        self.dataset = dataset
        self.description = description
        self.extra_params = kwargs
        self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "algorithm": self.algorithm,
            "hyperparameters": self.hyperparameters,
            "dataset": self.dataset,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            **self.extra_params,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        config = cls(
            algorithm=data["algorithm"],
            hyperparameters=data["hyperparameters"],
            dataset=data["dataset"],
            description=data.get("description"),
        )
        config.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        return config
    
    def save(self, path: str) -> None:
        """Save configuration to file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load configuration from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# Global experiment tracker instance
_tracker: Optional[ExperimentTracker] = None


def get_tracker() -> ExperimentTracker:
    """Get or create global experiment tracker."""
    global _tracker
    if _tracker is None:
        _tracker = ExperimentTracker()
    return _tracker


def set_tracker(tracker: ExperimentTracker) -> None:
    """Set global experiment tracker."""
    global _tracker
    _tracker = tracker

