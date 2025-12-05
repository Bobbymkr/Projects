"""
Traffic Controller Service.

Service layer that bridges API requests to traffic control algorithms.
"""

from typing import Dict, List, Optional
import asyncio
import logging
from datetime import datetime, timedelta
from collections import defaultdict

# Import traffic control algorithms
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.control.fuzzy_control import FuzzyController
from src.rl.dqn_agent import DQNAgent, DQNConfig
import numpy as np

logger = logging.getLogger(__name__)


class IntersectionRegistry:
    """
    Registry for tracking active intersections and their metrics.
    
    Maintains in-memory state of intersections for real-time monitoring.
    """
    
    def __init__(self):
        """Initialize intersection registry."""
        self.intersections: Dict[str, Dict] = {}
        self.metrics_history: Dict[str, List[Dict]] = defaultdict(list)
        self.last_update: Dict[str, datetime] = {}
    
    def register_intersection(self, intersection_id: str, name: str, metadata: Optional[Dict] = None):
        """Register a new intersection."""
        if intersection_id not in self.intersections:
            self.intersections[intersection_id] = {
                "intersection_id": intersection_id,
                "name": name,
                "status": "active",
                "total_vehicles_processed": 0,
                "average_wait_time": 0.0,
                "average_queue_length": 0.0,
                "current_phase": 0,
                "uptime_hours": 0.0,
                "efficiency_score": 0.0,
                "created_at": datetime.utcnow(),
                "last_decision_at": None,
                "metadata": metadata or {}
            }
            self.last_update[intersection_id] = datetime.utcnow()
            logger.info(f"Registered intersection: {intersection_id} ({name})")
    
    def update_metrics(self, intersection_id: str, metrics: Dict):
        """Update metrics for an intersection."""
        if intersection_id not in self.intersections:
            # Auto-register if not exists
            self.register_intersection(intersection_id, intersection_id)
        
        intersection = self.intersections[intersection_id]
        intersection.update({
            k: v for k, v in metrics.items() 
            if k in intersection
        })
        intersection["last_decision_at"] = datetime.utcnow()
        self.last_update[intersection_id] = datetime.utcnow()
        
        # Store metrics history (keep last 100 entries)
        self.metrics_history[intersection_id].append({
            **metrics,
            "timestamp": datetime.utcnow()
        })
        if len(self.metrics_history[intersection_id]) > 100:
            self.metrics_history[intersection_id].pop(0)
    
    def get_intersection(self, intersection_id: str) -> Optional[Dict]:
        """Get intersection data."""
        if intersection_id not in self.intersections:
            return None
        
        intersection = self.intersections[intersection_id].copy()
        # Calculate uptime
        if intersection["created_at"]:
            uptime = (datetime.utcnow() - intersection["created_at"]).total_seconds() / 3600.0
            intersection["uptime_hours"] = uptime
        
        return intersection
    
    def get_all_intersections(self, status: Optional[str] = None) -> List[Dict]:
        """Get all intersections, optionally filtered by status."""
        intersections = []
        for intersection_id, data in self.intersections.items():
            if status is None or data["status"] == status:
                intersection = self.get_intersection(intersection_id)
                if intersection:
                    intersections.append(intersection)
        return intersections
    
    def get_active_count(self) -> int:
        """Get count of active intersections."""
        return sum(1 for i in self.intersections.values() if i["status"] == "active")


class TrafficController:
    """
    Service for managing traffic control decisions.
    
    Coordinates between different algorithms and provides
    a unified interface for decision making.
    """
    
    def __init__(self):
        """Initialize traffic controller with available algorithms."""
        self.fuzzy_controller = FuzzyController()
        
        # Initialize DQN agent (lazy loading - only if needed)
        self.dqn_agent: Optional[DQNAgent] = None
        self.dqn_agent_loaded = False
        
        # Initialize intersection registry
        self.registry = IntersectionRegistry()
        
        # Initialize with some default intersections for demo
        self._initialize_default_intersections()
        
        self.default_algorithm = "fuzzy"
    
    def _initialize_default_intersections(self):
        """Initialize default intersections for demonstration."""
        default_intersections = [
            {"id": "intersection_001", "name": "Main Street & First Avenue"},
            {"id": "intersection_002", "name": "Highway 101 & Oak Street"},
            {"id": "intersection_003", "name": "Park Avenue & Elm Street"},
            {"id": "intersection_004", "name": "Commerce Blvd & Market Street"},
        ]
        
        for inter in default_intersections:
            self.registry.register_intersection(
                inter["id"],
                inter["name"],
                {"type": "4-way", "num_lanes": 4}
            )
    
    def _get_dqn_agent(self) -> Optional[DQNAgent]:
        """Lazy load DQN agent if needed."""
        if not self.dqn_agent_loaded:
            try:
                # Initialize DQN agent with default config
                # In production, this would load from a trained model
                cfg = DQNConfig()
                # For now, we'll use a placeholder - actual agent would need state/action dims
                # self.dqn_agent = DQNAgent(state_dim=4, action_dim=12, cfg=cfg)
                self.dqn_agent_loaded = True
                logger.info("DQN agent initialized (placeholder)")
            except Exception as e:
                logger.warning(f"Failed to initialize DQN agent: {e}")
        return self.dqn_agent
    
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
            # Ensure intersection is registered
            if intersection_id not in self.registry.intersections:
                self.registry.register_intersection(intersection_id, intersection_id)
            
            # Algorithm selection based on intersection config
            intersection_config = self.registry.intersections.get(intersection_id, {}).get("metadata", {})
            algorithm = intersection_config.get("algorithm", self.default_algorithm)
            
            # Select algorithm based on config
            if algorithm == "dqn":
                green_time, recommended_phase, confidence = self._use_dqn_algorithm(
                    intersection_id, queue_lengths, wait_times, current_phase
                )
            elif algorithm == "gnn":
                # GNN not yet implemented in service layer, fallback to fuzzy
                logger.warning(f"GNN algorithm requested but not yet implemented for {intersection_id}, using fuzzy")
                green_time = self.fuzzy_controller.compute_timing(queue_lengths)
                max_queue_idx = queue_lengths.index(max(queue_lengths))
                recommended_phase = max_queue_idx // 2
                confidence = 0.7
            else:  # Default to fuzzy
                green_time = self.fuzzy_controller.compute_timing(queue_lengths)
                max_queue_idx = queue_lengths.index(max(queue_lengths))
                recommended_phase = max_queue_idx // 2
                confidence = 0.85
            
            # Ensure phase is within valid range
            recommended_phase = max(0, min(3, recommended_phase))
            
            # Calculate confidence based on queue difference (if not set by algorithm)
            if 'confidence' in locals() and confidence is not None:
                pass  # Use algorithm-provided confidence
            else:
                queue_difference = max(queue_lengths) - min(queue_lengths)
                confidence = min(1.0, 0.7 + (queue_difference / 20.0))
            
            # Estimate improvement (simple heuristic)
            current_wait = sum(wait_times) / len(wait_times) if wait_times else 0.0
            estimated_improvement = max(0, min(50, (current_wait - 10) / max(current_wait, 1) * 100)) if current_wait > 0 else 0.0
            
            # Update intersection metrics
            avg_queue = sum(queue_lengths) / len(queue_lengths) if queue_lengths else 0.0
            avg_wait = current_wait
            total_vehicles = int(throughput * 3600) if throughput > 0 else 0  # Estimate hourly
            
            # Calculate efficiency score (0-1, higher is better)
            # Based on wait time and queue length (normalized)
            efficiency_score = max(0.0, min(1.0, 1.0 - (avg_wait / 60.0) - (avg_queue / 40.0)))
            
            self.registry.update_metrics(intersection_id, {
                "total_vehicles_processed": total_vehicles,
                "average_wait_time": avg_wait,
                "average_queue_length": avg_queue,
                "current_phase": recommended_phase,
                "efficiency_score": efficiency_score,
            })
            
            return {
                "phase": recommended_phase,
                "green_time": float(green_time),
                "confidence": round(confidence, 3),
                "algorithm": algorithm,
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
        """
        Get metrics for all intersections.
        
        Args:
            status: Optional status filter ("active", "inactive", etc.)
            
        Returns:
            List of intersection metrics dictionaries
        """
        try:
            intersections = self.registry.get_all_intersections(status=status)
            
            # Convert to IntersectionMetrics format
            from src.api.schemas import IntersectionMetrics, IntersectionStatus
            
            result = []
            for inter_data in intersections:
                try:
                    result.append({
                        "intersection_id": inter_data["intersection_id"],
                        "name": inter_data["name"],
                        "status": IntersectionStatus(inter_data["status"]),
                        "total_vehicles_processed": inter_data["total_vehicles_processed"],
                        "average_wait_time": inter_data["average_wait_time"],
                        "average_queue_length": inter_data["average_queue_length"],
                        "current_phase": inter_data["current_phase"],
                        "uptime_hours": inter_data["uptime_hours"],
                        "efficiency_score": inter_data["efficiency_score"],
                    })
                except Exception as e:
                    logger.warning(f"Error formatting intersection {inter_data.get('intersection_id')}: {e}")
                    continue
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching all intersections: {e}", exc_info=True)
            return []
    
    async def get_intersection(self, intersection_id: str) -> Optional[Dict]:
        """
        Get detailed metrics for a specific intersection.
        
        Args:
            intersection_id: Unique intersection identifier
            
        Returns:
            Intersection metrics dictionary or None if not found
        """
        try:
            inter_data = self.registry.get_intersection(intersection_id)
            if not inter_data:
                return None
            
            # Convert to IntersectionMetrics format
            from src.api.schemas import IntersectionStatus
            
            return {
                "intersection_id": inter_data["intersection_id"],
                "name": inter_data["name"],
                "status": IntersectionStatus(inter_data["status"]),
                "total_vehicles_processed": inter_data["total_vehicles_processed"],
                "average_wait_time": inter_data["average_wait_time"],
                "average_queue_length": inter_data["average_queue_length"],
                "current_phase": inter_data["current_phase"],
                "uptime_hours": inter_data["uptime_hours"],
                "efficiency_score": inter_data["efficiency_score"],
            }
            
        except Exception as e:
            logger.error(f"Error fetching intersection {intersection_id}: {e}", exc_info=True)
            return None

