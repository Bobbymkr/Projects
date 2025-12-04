"""
Convergence Detection and Early Stopping.

Implements Phase 0.3 from OPTIMIZATION_ROADMAP.md:
- Convergence detection based on reward stability
- Early stopping to prevent overfitting
- Performance tracking and monitoring
"""

import numpy as np
from typing import List, Optional, Dict, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ConvergenceMonitor:
    """
    Monitor training convergence and trigger early stopping.
    
    Detects convergence when performance plateaus for a specified number of episodes.
    """
    
    def __init__(
        self,
        window: int = 100,
        threshold: float = 0.01,
        patience: int = 500,
        min_episodes: int = 200,
        mode: str = "maximize"
    ):
        """
        Initialize convergence monitor.
        
        Args:
            window: Window size for computing moving average
            threshold: Minimum improvement threshold to reset patience
            patience: Number of episodes without improvement before stopping
            min_episodes: Minimum episodes before early stopping can trigger
            mode: "maximize" or "minimize" (for reward optimization)
        """
        self.window = window
        self.threshold = threshold
        self.patience = patience
        self.min_episodes = min_episodes
        self.mode = mode
        
        self.reward_history = deque(maxlen=window * 2)
        self.best_reward = -np.inf if mode == "maximize" else np.inf
        self.no_improvement_count = 0
        self.episode_count = 0
        self.converged = False
        self.best_episode = 0
    
    def update(self, reward: float, episode: int) -> Dict[str, Any]:
        """
        Update monitor with new reward value.
        
        Args:
            reward: Episode reward
            episode: Current episode number
        
        Returns:
            Dictionary with convergence status and metrics
        """
        self.reward_history.append(reward)
        self.episode_count = episode
        
        # Update best reward
        if self.mode == "maximize":
            improved = reward > self.best_reward + self.threshold
        else:
            improved = reward < self.best_reward - self.threshold
        
        if improved:
            self.best_reward = reward
            self.best_episode = episode
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        # Check convergence
        should_stop = False
        if self.episode_count >= self.min_episodes:
            if self.no_improvement_count >= self.patience:
                self.converged = True
                should_stop = True
        
        # Compute statistics
        recent_rewards = list(self.reward_history)[-self.window:]
        if len(recent_rewards) >= self.window:
            recent_avg = np.mean(recent_rewards)
            recent_std = np.std(recent_rewards)
            recent_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
        else:
            recent_avg = np.mean(recent_rewards) if recent_rewards else 0.0
            recent_std = 0.0
            recent_trend = 0.0
        
        return {
            "converged": self.converged,
            "should_stop": should_stop,
            "best_reward": self.best_reward,
            "best_episode": self.best_episode,
            "no_improvement_count": self.no_improvement_count,
            "recent_avg": recent_avg,
            "recent_std": recent_std,
            "recent_trend": recent_trend,
            "episode_count": self.episode_count
        }
    
    def check_convergence(self) -> bool:
        """
        Check if training has converged.
        
        Returns:
            True if converged, False otherwise
        """
        return self.converged
    
    def should_stop(self) -> bool:
        """
        Check if training should stop early.
        
        Returns:
            True if should stop, False otherwise
        """
        if not self.converged:
            return False
        
        if self.episode_count < self.min_episodes:
            return False
        
        return self.no_improvement_count >= self.patience
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        recent_rewards = list(self.reward_history)[-self.window:]
        return {
            "best_reward": self.best_reward,
            "best_episode": self.best_episode,
            "no_improvement_count": self.no_improvement_count,
            "episode_count": self.episode_count,
            "converged": self.converged,
            "recent_avg": np.mean(recent_rewards) if recent_rewards else 0.0,
            "recent_std": np.std(recent_rewards) if len(recent_rewards) > 1 else 0.0,
        }
    
    def reset(self):
        """Reset monitor state."""
        self.reward_history.clear()
        self.best_reward = -np.inf if self.mode == "maximize" else np.inf
        self.no_improvement_count = 0
        self.episode_count = 0
        self.converged = False
        self.best_episode = 0


class PerformanceTracker:
    """
    Track training performance metrics over time.
    """
    
    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Initialize performance tracker.
        
        Args:
            metrics: List of metric names to track
        """
        self.metrics = metrics or ["reward", "loss", "q_value"]
        self.history: Dict[str, List[float]] = {metric: [] for metric in self.metrics}
        self.episode_history: List[int] = []
    
    def log(self, episode: int, **kwargs):
        """
        Log metrics for an episode.
        
        Args:
            episode: Episode number
            **kwargs: Metric values
        """
        self.episode_history.append(episode)
        for metric in self.metrics:
            value = kwargs.get(metric, None)
            if value is not None:
                self.history[metric].append(value)
            else:
                # Pad with NaN if metric not provided
                self.history[metric].append(np.nan)
    
    def get_recent_average(self, metric: str, window: int = 100) -> float:
        """
        Get recent average of a metric.
        
        Args:
            metric: Metric name
            window: Window size
        
        Returns:
            Average value
        """
        if metric not in self.history:
            return 0.0
        
        values = self.history[metric][-window:]
        valid_values = [v for v in values if not np.isnan(v)]
        return np.mean(valid_values) if valid_values else 0.0
    
    def get_statistics(self, metric: str) -> Dict[str, float]:
        """
        Get statistics for a metric.
        
        Args:
            metric: Metric name
        
        Returns:
            Dictionary with statistics
        """
        if metric not in self.history:
            return {}
        
        values = [v for v in self.history[metric] if not np.isnan(v)]
        if not values:
            return {}
        
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "recent_avg": self.get_recent_average(metric),
        }
    
    def clear(self):
        """Clear all history."""
        for metric in self.metrics:
            self.history[metric].clear()
        self.episode_history.clear()

