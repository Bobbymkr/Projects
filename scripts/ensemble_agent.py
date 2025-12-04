"""
Ensemble Agent Implementation.

Combines multiple algorithms for improved performance.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class EnsembleAgent:
    """
    Ensemble of multiple traffic control agents.
    
    Combines predictions from multiple algorithms using
    weighted voting or stacking.
    """
    
    def __init__(
        self,
        agents: List[Any],
        weights: Optional[List[float]] = None,
        method: str = "weighted_voting",
    ):
        """
        Initialize ensemble agent.
        
        Args:
            agents: List of agent instances
            weights: Weights for each agent (default: equal weights)
            method: Ensemble method ('weighted_voting', 'majority', 'stacking')
        """
        self.agents = agents
        self.method = method
        
        if weights is None:
            # Equal weights by default
            self.weights = [1.0 / len(agents)] * len(agents)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        logger.info(f"Created ensemble with {len(agents)} agents")
        logger.info(f"Method: {method}, Weights: {self.weights}")
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using ensemble method."""
        if self.method == "weighted_voting":
            return self._weighted_voting(state)
        elif self.method == "majority":
            return self._majority_voting(state)
        elif self.method == "stacking":
            return self._stacking(state)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
    
    def _weighted_voting(self, state: np.ndarray) -> int:
        """Weighted voting based on agent confidence."""
        action_scores = {}
        
        for agent, weight in zip(self.agents, self.weights):
            try:
                action = agent.select_action(state)
                
                # Get confidence if available
                confidence = 1.0
                if hasattr(agent, 'get_confidence'):
                    confidence = agent.get_confidence(state)
                elif hasattr(agent, 'get_action_probability'):
                    probs = agent.get_action_probability(state)
                    confidence = np.max(probs) if probs is not None else 1.0
                
                # Weighted score
                score = weight * confidence
                
                if action not in action_scores:
                    action_scores[action] = 0.0
                action_scores[action] += score
                
            except Exception as e:
                logger.warning(f"Agent failed: {e}")
                continue
        
        if not action_scores:
            # Fallback to first agent
            return self.agents[0].select_action(state)
        
        # Return action with highest score
        return max(action_scores.items(), key=lambda x: x[1])[0]
    
    def _majority_voting(self, state: np.ndarray) -> int:
        """Majority voting."""
        votes = []
        
        for agent in self.agents:
            try:
                action = agent.select_action(state)
                votes.append(action)
            except Exception as e:
                logger.warning(f"Agent failed: {e}")
                continue
        
        if not votes:
            return self.agents[0].select_action(state)
        
        # Return most common action
        return max(set(votes), key=votes.count)
    
    def _stacking(self, state: np.ndarray) -> int:
        """Stacking with meta-learner (simplified)."""
        # Get predictions from all agents
        predictions = []
        
        for agent in self.agents:
            try:
                action = agent.select_action(state)
                predictions.append(action)
            except Exception as e:
                logger.warning(f"Agent failed: {e}")
                continue
        
        if not predictions:
            return self.agents[0].select_action(state)
        
        # Simple meta-learner: weighted average of action indices
        # In production, use a trained meta-learner
        weighted_sum = sum(
            weight * pred
            for weight, pred in zip(self.weights, predictions)
        )
        
        # Round to nearest action
        return int(np.round(weighted_sum))
    
    def get_confidence(self, state: np.ndarray) -> float:
        """Get ensemble confidence."""
        confidences = []
        
        for agent in self.agents:
            try:
                if hasattr(agent, 'get_confidence'):
                    conf = agent.get_confidence(state)
                elif hasattr(agent, 'get_action_probability'):
                    probs = agent.get_action_probability(state)
                    conf = np.max(probs) if probs is not None else 0.5
                else:
                    conf = 0.5  # Default confidence
                
                confidences.append(conf)
            except:
                confidences.append(0.0)
        
        # Weighted average confidence
        return sum(w * c for w, c in zip(self.weights, confidences))


def create_top_ensemble(
    state_dim: int,
    action_dim: int,
    top_n: int = 3,
    weights: Optional[List[float]] = None,
) -> EnsembleAgent:
    """
    Create ensemble from top N algorithms.
    
    Args:
        state_dim: State dimension
        action_dim: Action dimension
        top_n: Number of top algorithms to include
        weights: Custom weights (default: based on performance)
    
    Returns:
        EnsembleAgent instance
    """
    agents = []
    
    # Import top algorithms
    try:
        from src.research.novel_algorithms.transformer_control import TransformerAgent
        agents.append(("Transformer", TransformerAgent(state_dim, action_dim)))
    except:
        pass
    
    try:
        from src.research.novel_algorithms.imitation_learning import BehavioralCloningAgent
        agents.append(("Imitation Learning", BehavioralCloningAgent(state_dim, action_dim)))
    except:
        pass
    
    try:
        from src.research.novel_algorithms.llm_traffic import LLMTrafficAgent
        agents.append(("LLM", LLMTrafficAgent(state_dim, action_dim)))
    except:
        pass
    
    try:
        from src.research.novel_algorithms.hierarchical_rl_complete import CompleteHierarchicalRLAgent
        agents.append(("Hierarchical RL", CompleteHierarchicalRLAgent(state_dim, action_dim)))
    except:
        pass
    
    try:
        from src.research.novel_algorithms.model_based_rl_complete import CompleteModelBasedRLAgent
        agents.append(("Model-Based RL", CompleteModelBasedRLAgent(state_dim, action_dim)))
    except:
        pass
    
    # Select top N
    if len(agents) > top_n:
        # Use performance-based weights if available
        # For now, use first top_n
        agents = agents[:top_n]
    
    if not agents:
        raise ValueError("No agents available for ensemble")
    
    # Default weights based on ranking (Transformer > IL > LLM)
    if weights is None:
        default_weights = [0.4, 0.3, 0.2, 0.05, 0.05][:len(agents)]
        weights = default_weights
    
    return EnsembleAgent(
        agents=[agent for _, agent in agents],
        weights=weights,
        method="weighted_voting",
    )

