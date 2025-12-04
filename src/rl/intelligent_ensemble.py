"""
Intelligent Ensemble Methods for Traffic Control.

Implements Phase 5 from OPTIMIZATION_ROADMAP.md:
- Weighted Ensemble: Performance-based weights
- Dynamic Ensemble: Context-aware, confidence-based, adaptive
- Meta-Learning: Stacking for optimal combination

Expected Impact: 10-15% performance, 30% variance reduction
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods."""
    method: str = "weighted_voting"  # "weighted_voting", "dynamic", "stacking", "confidence_based"
    performance_weights: Optional[Dict[str, float]] = None
    context_aware: bool = True
    confidence_based: bool = True
    adaptive: bool = True
    learning_rate: float = 0.01
    window_size: int = 100
    meta_learner_dim: int = 64


class PerformanceTracker:
    """
    Track performance of individual agents for weight calculation.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize performance tracker.
        
        Args:
            window_size: Size of performance window
        """
        self.window_size = window_size
        self.agent_performances: Dict[str, deque] = {}
        self.agent_rewards: Dict[str, deque] = {}
        self.agent_stabilities: Dict[str, deque] = {}
    
    def register_agent(self, agent_name: str):
        """
        Register an agent for tracking.
        
        Args:
            agent_name: Name of agent
        """
        self.agent_performances[agent_name] = deque(maxlen=self.window_size)
        self.agent_rewards[agent_name] = deque(maxlen=self.window_size)
        self.agent_stabilities[agent_name] = deque(maxlen=self.window_size)
    
    def update(self, agent_name: str, reward: float, performance: float, stability: float = 1.0):
        """
        Update agent performance.
        
        Args:
            agent_name: Name of agent
            reward: Episode reward
            performance: Performance metric
            stability: Stability metric (variance)
        """
        if agent_name not in self.agent_performances:
            self.register_agent(agent_name)
        
        self.agent_rewards[agent_name].append(reward)
        self.agent_performances[agent_name].append(performance)
        self.agent_stabilities[agent_name].append(stability)
    
    def get_performance_metrics(self, agent_name: str) -> Dict[str, float]:
        """
        Get performance metrics for agent.
        
        Args:
            agent_name: Name of agent
            
        Returns:
            Performance metrics
        """
        if agent_name not in self.agent_performances:
            return {"mean_reward": 0.0, "mean_performance": 0.0, "stability": 1.0}
        
        rewards = list(self.agent_rewards[agent_name])
        performances = list(self.agent_performances[agent_name])
        stabilities = list(self.agent_stabilities[agent_name])
        
        return {
            "mean_reward": np.mean(rewards) if rewards else 0.0,
            "mean_performance": np.mean(performances) if performances else 0.0,
            "stability": np.mean(stabilities) if stabilities else 1.0,
            "variance": np.var(rewards) if rewards else 1.0
        }
    
    def calculate_weights(self) -> Dict[str, float]:
        """
        Calculate performance-based weights.
        
        Returns:
            Dictionary of agent weights
        """
        weights = {}
        total_score = 0.0
        
        for agent_name in self.agent_performances.keys():
            metrics = self.get_performance_metrics(agent_name)
            # Combined score: performance + stability - variance
            score = metrics["mean_performance"] + metrics["stability"] - metrics["variance"] * 0.1
            score = max(0.0, score)  # Ensure non-negative
            weights[agent_name] = score
            total_score += score
        
        # Normalize weights
        if total_score > 0:
            weights = {k: v / total_score for k, v in weights.items()}
        else:
            # Equal weights if all scores are zero
            n = len(weights)
            weights = {k: 1.0 / n for k in weights.keys()}
        
        return weights


class ContextAwareSelector:
    """
    Context-aware agent selection based on traffic conditions.
    """
    
    def __init__(self):
        """Initialize context-aware selector."""
        self.context_rules = {
            "rush_hour": ["Transformer", "Hierarchical RL"],
            "night": ["DQN", "Model-Based RL"],
            "high_traffic": ["Transformer", "GNN"],
            "low_traffic": ["DQN", "Model-Based RL"],
            "emergency": ["Hierarchical RL", "Transformer"],
            "normal": ["Transformer", "DQN", "Model-Based RL"]
        }
    
    def select_agents(self, context: Dict[str, Any], available_agents: List[str]) -> List[str]:
        """
        Select agents based on context.
        
        Args:
            context: Environment context
            available_agents: List of available agent names
            
        Returns:
            List of selected agent names
        """
        # Determine context type
        context_type = self._determine_context(context)
        
        # Get preferred agents for context
        preferred = self.context_rules.get(context_type, self.context_rules["normal"])
        
        # Filter to available agents
        selected = [agent for agent in preferred if agent in available_agents]
        
        # If no matches, use all available
        if not selected:
            selected = available_agents
        
        return selected
    
    def _determine_context(self, context: Dict[str, Any]) -> str:
        """
        Determine context type from environment context.
        
        Args:
            context: Environment context
            
        Returns:
            Context type string
        """
        # Check for emergency
        if context.get("emergency", False):
            return "emergency"
        
        # Check traffic density
        traffic_density = context.get("traffic_density", 0.5)
        if traffic_density > 0.8:
            return "high_traffic"
        elif traffic_density < 0.3:
            return "low_traffic"
        
        # Check time of day
        hour = context.get("hour", 12)
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            return "rush_hour"
        elif 22 <= hour or hour <= 6:
            return "night"
        
        return "normal"


class ConfidenceBasedWeighting:
    """
    Confidence-based weighting for ensemble.
    """
    
    def __init__(self):
        """Initialize confidence-based weighting."""
        pass
    
    def calculate_confidence_weights(self, predictions: Dict[str, Tuple[int, float]], 
                                    base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate confidence-based weights.
        
        Args:
            predictions: Dictionary of agent_name -> (action, confidence)
            base_weights: Base performance weights
            
        Returns:
            Confidence-adjusted weights
        """
        confidence_weights = {}
        total_confidence = 0.0
        
        for agent_name, (action, confidence) in predictions.items():
            base_weight = base_weights.get(agent_name, 0.0)
            # Combine base weight with confidence
            confidence_weight = base_weight * (1.0 + confidence)
            confidence_weights[agent_name] = confidence_weight
            total_confidence += confidence_weight
        
        # Normalize
        if total_confidence > 0:
            confidence_weights = {k: v / total_confidence for k, v in confidence_weights.items()}
        else:
            # Fall back to base weights
            confidence_weights = base_weights
        
        return confidence_weights


class AdaptiveWeightLearner:
    """
    Online learning of optimal ensemble weights.
    """
    
    def __init__(self, num_agents: int, learning_rate: float = 0.01):
        """
        Initialize adaptive weight learner.
        
        Args:
            num_agents: Number of agents
            learning_rate: Learning rate for weight updates
        """
        self.num_agents = num_agents
        self.learning_rate = learning_rate
        self.weights = np.ones(num_agents) / num_agents  # Equal initial weights
        self.performance_history = deque(maxlen=100)
    
    def update_weights(self, agent_rewards: List[float], ensemble_reward: float):
        """
        Update weights based on performance.
        
        Args:
            agent_rewards: List of rewards for each agent
            ensemble_reward: Reward of ensemble action
        """
        agent_rewards = np.array(agent_rewards)
        
        # Calculate advantage: how much better each agent performed
        advantages = agent_rewards - ensemble_reward
        
        # Update weights: increase weight for agents with positive advantage
        weight_updates = self.learning_rate * advantages
        
        # Apply updates
        self.weights += weight_updates
        
        # Normalize and clip
        self.weights = np.clip(self.weights, 0.0, 1.0)
        self.weights = self.weights / (self.weights.sum() + 1e-10)
        
        # Track performance
        self.performance_history.append(ensemble_reward)
    
    def get_weights(self) -> np.ndarray:
        """
        Get current weights.
        
        Returns:
            Current weight array
        """
        return self.weights.copy()


class MetaLearner(nn.Module):
    """
    Meta-learner for stacking ensemble.
    
    Learns optimal combination of base algorithm predictions.
    """
    
    def __init__(self, num_agents: int, input_dim: int, hidden_dim: int = 64):
        """
        Initialize meta-learner.
        
        Args:
            num_agents: Number of base agents
            input_dim: Input dimension (state dimension)
            hidden_dim: Hidden dimension
        """
        super(MetaLearner, self).__init__()
        self.num_agents = num_agents
        
        # Network to learn combination weights
        # Input: state_dim + num_agents (agent confidences)
        self.network = nn.Sequential(
            nn.Linear(input_dim + num_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state: torch.Tensor, agent_predictions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor [batch, state_dim]
            agent_predictions: Agent predictions [batch, num_agents, action_dim]
            
        Returns:
            Ensemble weights [batch, num_agents]
        """
        batch_size = state.size(0)
        state_dim = state.size(1)
        
        # Get action probabilities (max over actions) or use confidence
        # Simplified: use max action probability per agent
        agent_confidences = agent_predictions.max(dim=-1)[0]  # [batch, num_agents]
        
        # Concatenate state and agent confidences
        combined = torch.cat([state, agent_confidences], dim=-1)  # [batch, state_dim + num_agents]
        
        # Get ensemble weights
        weights = self.network(combined)  # [batch, num_agents]
        
        return weights


class IntelligentEnsemble:
    """
    Intelligent ensemble with multiple combination strategies.
    """
    
    def __init__(
        self,
        agents: Dict[str, Any],
        config: Optional[EnsembleConfig] = None
    ):
        """
        Initialize intelligent ensemble.
        
        Args:
            agents: Dictionary of agent_name -> agent_instance
            config: Ensemble configuration
        """
        self.agents = agents
        self.config = config or EnsembleConfig()
        self.agent_names = list(agents.keys())
        
        # Initialize components
        self.performance_tracker = PerformanceTracker(window_size=self.config.window_size)
        for name in self.agent_names:
            self.performance_tracker.register_agent(name)
        
        self.context_selector = ContextAwareSelector() if self.config.context_aware else None
        self.confidence_weighting = ConfidenceBasedWeighting() if self.config.confidence_based else None
        self.adaptive_learner = AdaptiveWeightLearner(
            len(self.agent_names), 
            learning_rate=self.config.learning_rate
        ) if self.config.adaptive else None
        
        # Meta-learner for stacking
        if self.config.method == "stacking":
            # Assume state_dim from first agent
            first_agent = list(agents.values())[0]
            if hasattr(first_agent, 'state_dim'):
                state_dim = first_agent.state_dim
            else:
                state_dim = 8  # Default
            self.meta_learner = MetaLearner(
                len(self.agent_names),
                state_dim,
                self.config.meta_learner_dim
            )
        else:
            self.meta_learner = None
        
        # Initialize weights
        if self.config.performance_weights:
            self.weights = self.config.performance_weights
        else:
            self.weights = {name: 1.0 / len(self.agent_names) for name in self.agent_names}
        
        logger.info(f"Initialized intelligent ensemble with {len(self.agent_names)} agents")
        logger.info(f"Method: {self.config.method}")
    
    def select_action(self, state: np.ndarray, context: Optional[Dict[str, Any]] = None) -> int:
        """
        Select action using ensemble.
        
        Args:
            state: Current state
            context: Environment context (optional)
            
        Returns:
            Selected action
        """
        # Get predictions from all agents
        predictions = {}
        confidences = {}
        
        for name, agent in self.agents.items():
            try:
                if hasattr(agent, 'select_action'):
                    action = agent.select_action(state)
                    # Get confidence if available
                    confidence = 1.0
                    if hasattr(agent, 'get_confidence'):
                        confidence = agent.get_confidence(state)
                    predictions[name] = (action, confidence)
                    confidences[name] = confidence
            except Exception as e:
                logger.warning(f"Agent {name} failed: {e}")
                continue
        
        if not predictions:
            # Fallback: random action
            return np.random.randint(0, 4)
        
        # Select ensemble method
        if self.config.method == "weighted_voting":
            return self._weighted_voting(predictions)
        elif self.config.method == "dynamic":
            return self._dynamic_ensemble(state, predictions, context)
        elif self.config.method == "confidence_based":
            return self._confidence_based(predictions)
        elif self.config.method == "stacking":
            return self._stacking(state, predictions)
        else:
            return self._weighted_voting(predictions)
    
    def _weighted_voting(self, predictions: Dict[str, Tuple[int, float]]) -> int:
        """
        Weighted voting ensemble.
        
        Args:
            predictions: Dictionary of agent_name -> (action, confidence)
            
        Returns:
            Selected action
        """
        action_scores = {}
        
        for agent_name, (action, confidence) in predictions.items():
            weight = self.weights.get(agent_name, 0.0)
            if action not in action_scores:
                action_scores[action] = 0.0
            action_scores[action] += weight
        
        # Select action with highest score
        return max(action_scores.items(), key=lambda x: x[1])[0]
    
    def _dynamic_ensemble(self, state: np.ndarray, predictions: Dict[str, Tuple[int, float]],
                         context: Optional[Dict[str, Any]]) -> int:
        """
        Dynamic ensemble with context-aware selection.
        
        Args:
            state: Current state
            predictions: Agent predictions
            context: Environment context
            
        Returns:
            Selected action
        """
        # Context-aware agent selection
        if self.context_selector and context:
            selected_agents = self.context_selector.select_agents(context, list(predictions.keys()))
            # Filter predictions to selected agents
            predictions = {k: v for k, v in predictions.items() if k in selected_agents}
        
        # Confidence-based weighting
        if self.confidence_weighting:
            confidence_weights = self.confidence_weighting.calculate_confidence_weights(
                predictions, self.weights
            )
        else:
            confidence_weights = self.weights
        
        # Weighted voting with confidence weights
        action_scores = {}
        for agent_name, (action, confidence) in predictions.items():
            weight = confidence_weights.get(agent_name, 0.0)
            if action not in action_scores:
                action_scores[action] = 0.0
            action_scores[action] += weight
        
        return max(action_scores.items(), key=lambda x: x[1])[0]
    
    def _confidence_based(self, predictions: Dict[str, Tuple[int, float]]) -> int:
        """
        Confidence-based ensemble.
        
        Args:
            predictions: Agent predictions
            
        Returns:
            Selected action
        """
        if self.confidence_weighting:
            confidence_weights = self.confidence_weighting.calculate_confidence_weights(
                predictions, self.weights
            )
        else:
            confidence_weights = self.weights
        
        action_scores = {}
        for agent_name, (action, confidence) in predictions.items():
            weight = confidence_weights.get(agent_name, 0.0)
            if action not in action_scores:
                action_scores[action] = 0.0
            action_scores[action] += weight * confidence
        
        return max(action_scores.items(), key=lambda x: x[1])[0]
    
    def _stacking(self, state: np.ndarray, predictions: Dict[str, Tuple[int, float]]) -> int:
        """
        Stacking ensemble with meta-learner.
        
        Args:
            state: Current state
            predictions: Agent predictions
            
        Returns:
            Selected action
        """
        if self.meta_learner is None:
            return self._weighted_voting(predictions)
        
        # Prepare inputs for meta-learner
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # [1, state_dim]
        
        # Get action predictions from all agents as confidence matrix
        agent_confidences = torch.zeros(1, len(self.agent_names))  # [1, num_agents]
        for idx, (agent_name, (action, confidence)) in enumerate(predictions.items()):
            if idx < len(self.agent_names):
                agent_confidences[0, idx] = float(confidence)
        
        # Get ensemble weights from meta-learner
        with torch.no_grad():
            # Create agent_predictions tensor with confidences
            agent_predictions = agent_confidences.unsqueeze(-1)  # [1, num_agents, 1]
            weights = self.meta_learner(state_tensor, agent_predictions)  # [1, num_agents]
            weights = weights[0].detach().numpy()  # [num_agents]
        
        # Weighted voting with meta-learner weights
        action_scores = {}
        for idx, (agent_name, (action, confidence)) in enumerate(predictions.items()):
            if idx < len(weights):
                weight = weights[idx]
                if action not in action_scores:
                    action_scores[action] = 0.0
                action_scores[action] += weight * confidence
        
        return max(action_scores.items(), key=lambda x: x[1])[0]
    
    def update_performance(self, agent_name: str, reward: float, performance: float, stability: float = 1.0):
        """
        Update agent performance tracking.
        
        Args:
            agent_name: Name of agent
            reward: Episode reward
            performance: Performance metric
            stability: Stability metric
        """
        self.performance_tracker.update(agent_name, reward, performance, stability)
        
        # Update weights based on performance
        if self.config.adaptive:
            new_weights = self.performance_tracker.calculate_weights()
            self.weights.update(new_weights)
    
    def update_adaptive_weights(self, agent_rewards: List[float], ensemble_reward: float):
        """
        Update adaptive weights.
        
        Args:
            agent_rewards: List of rewards for each agent
            ensemble_reward: Reward of ensemble action
        """
        if self.adaptive_learner:
            self.adaptive_learner.update_weights(agent_rewards, ensemble_reward)
            # Update weights from adaptive learner
            learned_weights = self.adaptive_learner.get_weights()
            for idx, agent_name in enumerate(self.agent_names):
                if idx < len(learned_weights):
                    self.weights[agent_name] = learned_weights[idx]
            
            # Normalize
            total = sum(self.weights.values())
            if total > 0:
                self.weights = {k: v / total for k, v in self.weights.items()}
    
    def get_weights(self) -> Dict[str, float]:
        """
        Get current ensemble weights.
        
        Returns:
            Dictionary of agent weights
        """
        return self.weights.copy()

