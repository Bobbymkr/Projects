"""
Curriculum Learning with Adaptive Difficulty.

Implements Phase 2.1 from OPTIMIZATION_ROADMAP.md:
- Progressive curriculum with multiple difficulty levels
- Adaptive progression based on performance
- Traffic density and vehicle arrival rate scaling
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CurriculumLevel:
    """Represents a single curriculum level."""
    level_id: int
    traffic_density: float
    vehicles_per_hour: float
    arrival_rate_multiplier: float
    rush_hour: bool = False
    description: str = ""


class TrafficCurriculum:
    """
    Curriculum Learning for Traffic Control.
    
    Progressive curriculum that adaptively increases difficulty
    based on agent performance.
    """
    
    def __init__(
        self,
        base_arrival_rates: List[float],
        performance_threshold: float = 0.7,
        min_episodes_per_level: int = 50,
        performance_window: int = 100,
    ):
        """
        Initialize traffic curriculum.
        
        Args:
            base_arrival_rates: Base arrival rates per lane
            performance_threshold: Performance threshold for level progression (0-1)
            min_episodes_per_level: Minimum episodes before allowing progression
            performance_window: Window size for performance evaluation
        """
        self.base_arrival_rates = np.array(base_arrival_rates)
        self.performance_threshold = performance_threshold
        self.min_episodes_per_level = min_episodes_per_level
        self.performance_window = performance_window
        
        # Define curriculum levels
        self.levels = self._create_levels()
        self.current_level = 0
        self.episodes_at_current_level = 0
        self.performance_history = []
        
        logger.info(f"Initialized TrafficCurriculum with {len(self.levels)} levels")
    
    def _create_levels(self) -> List[CurriculumLevel]:
        """
        Create curriculum levels with progressive difficulty.
        
        Returns:
            List of curriculum levels
        """
        levels = [
            CurriculumLevel(
                level_id=0,
                traffic_density=0.1,
                vehicles_per_hour=100,
                arrival_rate_multiplier=0.1,
                description="Very Easy: Light traffic"
            ),
            CurriculumLevel(
                level_id=1,
                traffic_density=0.3,
                vehicles_per_hour=300,
                arrival_rate_multiplier=0.3,
                description="Easy: Moderate traffic"
            ),
            CurriculumLevel(
                level_id=2,
                traffic_density=0.5,
                vehicles_per_hour=500,
                arrival_rate_multiplier=0.5,
                description="Medium: Normal traffic"
            ),
            CurriculumLevel(
                level_id=3,
                traffic_density=0.7,
                vehicles_per_hour=700,
                arrival_rate_multiplier=0.7,
                description="Hard: Heavy traffic"
            ),
            CurriculumLevel(
                level_id=4,
                traffic_density=0.9,
                vehicles_per_hour=900,
                arrival_rate_multiplier=0.9,
                description="Very Hard: Very heavy traffic"
            ),
            CurriculumLevel(
                level_id=5,
                traffic_density=1.0,
                vehicles_per_hour=1200,
                arrival_rate_multiplier=1.0,
                rush_hour=True,
                description="Extreme: Rush hour traffic"
            ),
        ]
        return levels
    
    def get_current_level(self) -> CurriculumLevel:
        """Get current curriculum level."""
        return self.levels[self.current_level]
    
    def get_arrival_rates(self) -> np.ndarray:
        """
        Get arrival rates for current curriculum level.
        
        Returns:
            Array of arrival rates per lane
        """
        level = self.get_current_level()
        return self.base_arrival_rates * level.arrival_rate_multiplier
    
    def update_performance(self, episode_reward: float, episode: int):
        """
        Update performance history and check for level progression.
        
        Args:
            episode_reward: Reward from current episode
            episode: Current episode number
        """
        self.performance_history.append(episode_reward)
        self.episodes_at_current_level += 1
        
        # Keep only recent performance history
        if len(self.performance_history) > self.performance_window:
            self.performance_history.pop(0)
        
        # Check for progression
        if self._should_progress():
            self._progress_to_next_level()
    
    def _should_progress(self) -> bool:
        """
        Check if agent should progress to next level.
        
        Returns:
            True if should progress, False otherwise
        """
        # Must have minimum episodes at current level
        if self.episodes_at_current_level < self.min_episodes_per_level:
            return False
        
        # Must not be at max level
        if self.current_level >= len(self.levels) - 1:
            return False
        
        # Check performance threshold
        if len(self.performance_history) < self.performance_window:
            return False
        
        # Compute performance metric (normalized reward)
        recent_performance = np.mean(self.performance_history[-self.performance_window:])
        
        # Normalize performance (assuming rewards are negative, so we normalize)
        # Higher (less negative) rewards = better performance
        # For progression, we want performance above threshold
        # Since rewards are typically negative, we need to adjust threshold logic
        
        # Simple heuristic: if average reward is improving and above a threshold
        # For normalized rewards around -1 to -10, threshold of -5 might be reasonable
        # But we'll use a relative improvement approach instead
        
        # Check if performance is consistently good
        # Use percentile-based threshold
        if len(self.performance_history) >= self.performance_window:
            recent_avg = np.mean(self.performance_history[-self.performance_window:])
            overall_avg = np.mean(self.performance_history)
            
            # Progress if recent performance is better than overall average
            # and above a threshold
            performance_ratio = recent_avg / max(abs(overall_avg), 1e-6)
            
            # For negative rewards, we want recent_avg to be less negative (higher)
            # So we check if recent_avg > overall_avg * threshold
            return recent_avg > overall_avg * self.performance_threshold
        
        return False
    
    def _progress_to_next_level(self):
        """Progress to next curriculum level."""
        if self.current_level < len(self.levels) - 1:
            self.current_level += 1
            self.episodes_at_current_level = 0
            level = self.get_current_level()
            logger.info(
                f"Progressed to level {self.current_level}: {level.description} "
                f"(density={level.traffic_density}, vehicles/hour={level.vehicles_per_hour})"
            )
    
    def get_config_update(self) -> Dict[str, Any]:
        """
        Get environment configuration update for current level.
        
        Returns:
            Dictionary with configuration updates
        """
        level = self.get_current_level()
        arrival_rates = self.get_arrival_rates()
        
        return {
            "arrival_rates": arrival_rates.tolist(),
            "curriculum_level": self.current_level,
            "traffic_density": level.traffic_density,
            "vehicles_per_hour": level.vehicles_per_hour,
            "rush_hour": level.rush_hour,
        }
    
    def reset(self):
        """Reset curriculum to initial level."""
        self.current_level = 0
        self.episodes_at_current_level = 0
        self.performance_history = []
        logger.info("Curriculum reset to level 0")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get curriculum statistics."""
        level = self.get_current_level()
        return {
            "current_level": self.current_level,
            "level_description": level.description,
            "episodes_at_level": self.episodes_at_current_level,
            "traffic_density": level.traffic_density,
            "vehicles_per_hour": level.vehicles_per_hour,
            "total_levels": len(self.levels),
            "recent_performance": np.mean(self.performance_history[-self.performance_window:]) if len(self.performance_history) >= self.performance_window else None,
        }


class AdaptiveCurriculum:
    """
    Advanced adaptive curriculum that adjusts difficulty based on multiple metrics.
    
    More sophisticated than basic curriculum, adapts based on:
    - Reward trends
    - Convergence speed
    - Variance in performance
    """
    
    def __init__(
        self,
        base_arrival_rates: List[float],
        initial_level: int = 0,
        progression_rate: float = 0.1,
        regression_threshold: float = 0.3,
    ):
        """
        Initialize adaptive curriculum.
        
        Args:
            base_arrival_rates: Base arrival rates per lane
            initial_level: Starting curriculum level
            progression_rate: Rate of difficulty increase (0-1)
            regression_threshold: Threshold for regressing to easier level
        """
        self.base_arrival_rates = np.array(base_arrival_rates)
        self.progression_rate = progression_rate
        self.regression_threshold = regression_threshold
        
        # Create curriculum with more granular levels
        self.levels = self._create_adaptive_levels()
        self.current_level = initial_level
        self.performance_trend = []
        
        logger.info(f"Initialized AdaptiveCurriculum with {len(self.levels)} levels")
    
    def _create_adaptive_levels(self) -> List[CurriculumLevel]:
        """Create adaptive curriculum levels with fine-grained difficulty."""
        levels = []
        densities = np.linspace(0.1, 1.0, 20)  # 20 levels for smooth progression
        
        for i, density in enumerate(densities):
            levels.append(CurriculumLevel(
                level_id=i,
                traffic_density=density,
                vehicles_per_hour=100 + density * 1100,
                arrival_rate_multiplier=density,
                rush_hour=(density >= 0.9),
                description=f"Level {i}: Density {density:.2f}"
            ))
        
        return levels
    
    def update(self, episode_reward: float, episode: int):
        """
        Update curriculum based on performance.
        
        Args:
            episode_reward: Reward from current episode
            episode: Current episode number
        """
        self.performance_trend.append(episode_reward)
        
        # Keep recent history
        if len(self.performance_trend) > 100:
            self.performance_trend.pop(0)
        
        # Adaptive progression/regression
        if len(self.performance_trend) >= 50:
            self._adaptive_update()
    
    def _adaptive_update(self):
        """Adaptively update curriculum level based on performance trends."""
        recent = np.array(self.performance_trend[-50:])
        older = np.array(self.performance_trend[-100:-50]) if len(self.performance_trend) >= 100 else recent
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        # Compute improvement
        improvement = (recent_avg - older_avg) / max(abs(older_avg), 1e-6)
        
        # Progress if improving
        if improvement > self.progression_rate and self.current_level < len(self.levels) - 1:
            self.current_level = min(self.current_level + 1, len(self.levels) - 1)
            logger.debug(f"Adaptive progression to level {self.current_level}")
        
        # Regress if performance degrading
        elif improvement < -self.regression_threshold and self.current_level > 0:
            self.current_level = max(self.current_level - 1, 0)
            logger.debug(f"Adaptive regression to level {self.current_level}")
    
    def get_arrival_rates(self) -> np.ndarray:
        """Get arrival rates for current level."""
        level = self.levels[self.current_level]
        return self.base_arrival_rates * level.arrival_rate_multiplier
    
    def get_config_update(self) -> Dict[str, Any]:
        """Get environment configuration update."""
        level = self.levels[self.current_level]
        arrival_rates = self.get_arrival_rates()
        
        return {
            "arrival_rates": arrival_rates.tolist(),
            "curriculum_level": self.current_level,
            "traffic_density": level.traffic_density,
        }

