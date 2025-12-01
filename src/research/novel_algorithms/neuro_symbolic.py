"""
Neuro-Symbolic AI for Traffic Signal Control.

Combines neural networks with symbolic reasoning for
interpretable and robust traffic control.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple
import re

logger = logging.getLogger(__name__)


class SymbolicRule:
    """
    Symbolic rule for traffic control.
    
    Represents interpretable if-then rules.
    """
    
    def __init__(self, condition: str, action: int, confidence: float = 1.0):
        """
        Initialize symbolic rule.
        
        Args:
            condition: Logical condition (e.g., "queue_north > 10 AND queue_south > 10")
            action: Action to take if condition is true
            confidence: Confidence in rule
        """
        self.condition = condition
        self.action = action
        self.confidence = confidence
    
    def evaluate(self, state: np.ndarray, state_names: List[str]) -> bool:
        """
        Evaluate rule condition.
        
        Args:
            state: Current state
            state_names: Names of state variables
            
        Returns:
            True if condition is satisfied
        """
        # Parse condition
        # Simplified: basic parsing
        try:
            # Replace variable names with values
            expr = self.condition
            for i, name in enumerate(state_names):
                expr = expr.replace(name, str(state[i]))
            
            # Evaluate expression
            result = eval(expr)
            return bool(result)
        except:
            return False


class NeuroSymbolicNetwork(nn.Module):
    """
    Neuro-Symbolic Network.
    
    Combines neural network with symbolic rules.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128],
    ):
        """
        Initialize neuro-symbolic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        self.neural_network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, symbolic_output: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass combining neural and symbolic outputs.
        
        Args:
            state: Input state
            symbolic_output: Symbolic rule output (if available)
            
        Returns:
            Combined output
        """
        neural_output = self.neural_network(state)
        
        if symbolic_output is not None:
            # Combine neural and symbolic
            combined = 0.7 * neural_output + 0.3 * symbolic_output
            return combined
        
        return neural_output


class SymbolicReasoner:
    """
    Symbolic reasoner for traffic control.
    
    Applies symbolic rules to make interpretable decisions.
    """
    
    def __init__(self, state_names: List[str]):
        """
        Initialize symbolic reasoner.
        
        Args:
            state_names: Names of state variables
        """
        self.state_names = state_names
        self.rules: List[SymbolicRule] = []
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default traffic control rules."""
        # Rule 1: If queue is very long, extend green time
        self.rules.append(SymbolicRule(
            condition="state[0] > 15 OR state[1] > 15",
            action=0,  # Extend green for north-south
            confidence=0.9,
        ))
        
        # Rule 2: If queue is short, reduce green time
        self.rules.append(SymbolicRule(
            condition="state[0] < 3 AND state[1] < 3",
            action=1,  # Switch to east-west
            confidence=0.8,
        ))
        
        # Rule 3: Emergency: clear all queues
        self.rules.append(SymbolicRule(
            condition="np.sum(state) > 30",
            action=0,  # Maximum green
            confidence=1.0,
        ))
    
    def add_rule(self, rule: SymbolicRule):
        """Add custom rule."""
        self.rules.append(rule)
    
    def reason(self, state: np.ndarray) -> Tuple[int, float]:
        """
        Apply symbolic reasoning.
        
        Args:
            state: Current state
            
        Returns:
            Action and confidence
        """
        # Evaluate rules in order
        for rule in self.rules:
            if rule.evaluate(state, self.state_names):
                return rule.action, rule.confidence
        
        # Default: no rule matches
        return 0, 0.5
    
    def get_explanation(self, state: np.ndarray) -> str:
        """
        Get explanation for decision.
        
        Args:
            state: Current state
            
        Returns:
            Explanation string
        """
        for rule in self.rules:
            if rule.evaluate(state, self.state_names):
                return f"Rule: {rule.condition} -> Action {rule.action} (confidence: {rule.confidence})"
        
        return "No matching rule found"


class NeuroSymbolicAgent:
    """
    Neuro-Symbolic Agent for Traffic Control.
    
    Combines neural learning with symbolic reasoning.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        state_names: Optional[List[str]] = None,
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ):
        """
        Initialize neuro-symbolic agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            state_names: Names of state variables
            learning_rate: Learning rate
            device: Device for computation
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # State names
        self.state_names = state_names or [f"state_{i}" for i in range(state_dim)]
        
        # Neural network
        self.neural_network = NeuroSymbolicNetwork(state_dim, action_dim).to(device)
        
        # Symbolic reasoner
        self.symbolic_reasoner = SymbolicReasoner(self.state_names)
        
        # Optimizer
        self.optimizer = optim.Adam(self.neural_network.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize is_trained flag
        self.is_trained = False
        
        # Combination weight (learnable)
        self.neural_weight = 0.7
        self.symbolic_weight = 0.3
        
        logger.info("Initialized Neuro-Symbolic Agent")
    
    def select_action(
        self,
        state: np.ndarray,
        use_symbolic: bool = True,
        return_explanation: bool = False,
    ) -> Tuple[int, Optional[str]]:
        """
        Select action using neuro-symbolic reasoning.
        
        Args:
            state: Current state
            use_symbolic: Whether to use symbolic reasoning
            return_explanation: Whether to return explanation
            
        Returns:
            Action and optional explanation
        """
        if not self.is_trained:
            # Return random action if not trained
            action = np.random.randint(0, self.action_dim)
            return int(action), None
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Neural prediction
        neural_logits = self.neural_network.neural_network(state_tensor)
        
        if use_symbolic:
            # Symbolic reasoning
            symbolic_action, symbolic_confidence = self.symbolic_reasoner.reason(state)
            
            # Create symbolic output tensor
            symbolic_output = torch.zeros(1, self.action_dim).to(self.device)
            symbolic_output[0, symbolic_action] = symbolic_confidence
            
            # Combine
            combined_logits = (
                self.neural_weight * neural_logits +
                self.symbolic_weight * symbolic_output
            )
            
            action = torch.argmax(combined_logits, dim=1).item()
            
            explanation = None
            if return_explanation:
                explanation = self.symbolic_reasoner.get_explanation(state)
            
            return int(action), explanation
        else:
            # Neural only
            action = torch.argmax(neural_logits, dim=1).item()
            return int(action), None
    
    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> Dict[str, float]:
        """
        Train neuro-symbolic agent.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            
        Returns:
            Training metrics
        """
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        
        # Forward pass
        logits = self.neural_network.neural_network(states_tensor)
        
        # Loss
        loss = self.criterion(logits, actions_tensor)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.neural_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.is_trained = True
        
        return {"loss": loss.item()}
    
    def add_symbolic_rule(self, rule: SymbolicRule):
        """Add symbolic rule."""
        self.symbolic_reasoner.add_rule(rule)
    
    def save(self, path: str):
        """Save agent."""
        torch.save({
            'neural_network_state_dict': self.neural_network.state_dict(),
            'is_trained': self.is_trained,
            'neural_weight': self.neural_weight,
            'symbolic_weight': self.symbolic_weight,
        }, path)
        logger.info(f"Saved Neuro-Symbolic Agent to {path}")
    
    def load(self, path: str):
        """Load agent."""
        checkpoint = torch.load(path, map_location=self.device)
        self.neural_network.load_state_dict(checkpoint['neural_network_state_dict'])
        self.is_trained = checkpoint.get('is_trained', False)
        self.neural_weight = checkpoint.get('neural_weight', 0.7)
        self.symbolic_weight = checkpoint.get('symbolic_weight', 0.3)
        logger.info(f"Loaded Neuro-Symbolic Agent from {path}")

