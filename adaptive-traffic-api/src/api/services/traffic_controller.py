"""
Traffic Controller Service.

Service layer that bridges API requests to traffic control algorithms.
"""

from typing import Dict, List, Optional
import asyncio
import logging
from datetime import datetime

# Import traffic control algorithms
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.control.fuzzy_control import FuzzyController
from src.rl.dqn_agent import DQNAgent, DQNConfig

logger = logging.getLogger(__name__)


class TrafficController:
    """
    Service for managing traffic control decisions.
    
    Coordinates between different algorithms and provides
    a unified interface for decision making.
    """
    
    def __init__(self):
        """Initialize traffic controller with available algorithms."""
        self.fuzzy_controller = FuzzyController()
        # TODO: Initialize other controllers (DQN, GNN, etc.)
        self.default_algorithm = "fuzzy"
    
    async def make_decision(
        self,
        intersection_id: str,
        queue_lengths: List[float],
        wait_times: List[float],
        throughput: float,
        current_phase: int,
    ) -> Dict:
        """
        Make traffic signal decision using available algorithms.
        
        Args:
            intersection_id: Unique intersection identifier
            queue_lengths: Queue lengths for each lane
            wait_times: Average wait times per lane
            throughput: Current throughput
            current_phase: Current signal phase
            
        Returns:
            Decision dictionary with phase, timing, confidence, etc.
        """
        try:
            # Use fuzzy controller for now (best performer)
            green_time = self.fuzzy_controller.compute_timing(queue_lengths)
            
            # Determine recommended phase based on queue lengths
            # Simple logic: phase with longest queue gets priority
            max_queue_idx = queue_lengths.index(max(queue_lengths))
            recommended_phase = max_queue_idx // 2  # Assuming 2 lanes per phase
            
            # Ensure phase is within valid range
            recommended_phase = max(0, min(3, recommended_phase))
            
            # Calculate confidence based on queue difference
            queue_difference = max(queue_lengths) - min(queue_lengths)
            confidence = min(1.0, 0.7 + (queue_difference / 20.0))
            
            # Estimate improvement (simple heuristic)
            current_wait = sum(wait_times) / len(wait_times)
            estimated_improvement = max(0, min(50, (current_wait - 10) / current_wait * 100))
            
            return {
                "phase": recommended_phase,
                "green_time": float(green_time),
                "confidence": round(confidence, 3),
                "algorithm": "fuzzy",
                "reasoning": f"Queue-based decision: Phase {recommended_phase} selected due to highest queue length",
                "improvement": round(estimated_improvement, 1),
            }
            
        except Exception as e:
            logger.error(f"Error making traffic decision: {e}", exc_info=True)
            # Return safe default
            return {
                "phase": current_phase,
                "green_time": 20.0,
                "confidence": 0.5,
                "algorithm": "default",
                "reasoning": "Fallback to default timing due to error",
                "improvement": 0.0,
            }
    
    async def get_all_intersections(self, status: Optional[str] = None) -> List[Dict]:
        """Get metrics for all intersections."""
        # TODO: Integrate with actual data source
        # For now, return mock data matching dashboard expectations
        return []
    
    async def get_intersection(self, intersection_id: str) -> Optional[Dict]:
        """Get detailed metrics for a specific intersection."""
        # TODO: Integrate with actual data source
        return None

