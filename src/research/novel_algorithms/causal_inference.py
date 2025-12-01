"""
Causal Inference for Traffic Signal Control.

Uses causal reasoning to understand cause-effect relationships
in traffic patterns and optimize signal timing.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
import networkx as nx

logger = logging.getLogger(__name__)


class CausalGraph:
    """
    Causal Graph for traffic relationships.
    
    Models causal relationships between traffic variables.
    """
    
    def __init__(self):
        """Initialize causal graph."""
        self.graph = nx.DiGraph()
        self.nodes = []
        self.edges = []
    
    def add_node(self, name: str, node_type: str = "variable"):
        """Add node to causal graph."""
        self.graph.add_node(name, type=node_type)
        self.nodes.append(name)
    
    def add_edge(self, source: str, target: str, weight: float = 1.0):
        """Add causal edge."""
        self.graph.add_edge(source, target, weight=weight)
        self.edges.append((source, target, weight))
    
    def get_parents(self, node: str) -> List[str]:
        """Get parent nodes (causes)."""
        return list(self.graph.predecessors(node))
    
    def get_children(self, node: str) -> List[str]:
        """Get child nodes (effects)."""
        return list(self.graph.successors(node))
    
    def get_causal_path(self, source: str, target: str) -> Optional[List[str]]:
        """Get causal path between nodes."""
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return None


class CausalModel:
    """
    Causal Model for traffic control.
    
    Learns causal relationships and uses them for decision-making.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
    ):
        """
        Initialize causal model.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            device: Device for computation
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Build causal graph
        self.causal_graph = CausalGraph()
        self._build_traffic_causal_graph()
        
        # Causal effect estimator
        self.effect_estimator = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim),
        ).to(device)
        
        self.optimizer = optim.Adam(self.effect_estimator.parameters(), lr=1e-3)
        
        self.is_trained = False
        logger.info("Initialized Causal Model")
    
    def _build_traffic_causal_graph(self):
        """Build causal graph for traffic variables."""
        # Traffic variables
        variables = [
            "queue_length_north",
            "queue_length_south",
            "queue_length_east",
            "queue_length_west",
            "green_time",
            "wait_time",
            "throughput",
        ]
        
        for var in variables:
            self.causal_graph.add_node(var)
        
        # Causal relationships
        # Green time causes changes in queue lengths
        self.causal_graph.add_edge("green_time", "queue_length_north", weight=0.8)
        self.causal_graph.add_edge("green_time", "queue_length_south", weight=0.8)
        self.causal_graph.add_edge("green_time", "queue_length_east", weight=-0.5)
        self.causal_graph.add_edge("green_time", "queue_length_west", weight=-0.5)
        
        # Queue lengths affect wait time
        for direction in ["north", "south", "east", "west"]:
            self.causal_graph.add_edge(f"queue_length_{direction}", "wait_time", weight=0.6)
        
        # Green time affects throughput
        self.causal_graph.add_edge("green_time", "throughput", weight=0.7)
    
    def estimate_causal_effect(
        self,
        state: np.ndarray,
        action: int,
    ) -> np.ndarray:
        """
        Estimate causal effect of action on state.
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            Estimated next state
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_onehot = torch.zeros(1, self.action_dim).to(self.device)
        action_onehot[0, action] = 1.0
        
        input_tensor = torch.cat([state_tensor, action_onehot], dim=1)
        
        with torch.no_grad():
            effect = self.effect_estimator(input_tensor)
        
        return effect.cpu().numpy().squeeze()
    
    def train_causal_model(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
    ) -> Dict[str, float]:
        """
        Train causal effect estimator.
        
        Args:
            states: Current states
            actions: Actions taken
            next_states: Resulting states
            
        Returns:
            Training metrics
        """
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_onehot = torch.zeros(len(actions), self.action_dim).to(self.device)
        actions_onehot.scatter_(1, torch.LongTensor(actions).unsqueeze(1), 1.0)
        
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        
        input_tensor = torch.cat([states_tensor, actions_onehot], dim=1)
        
        # Predict effect
        predicted_next = self.effect_estimator(input_tensor)
        
        # Loss
        loss = nn.functional.mse_loss(predicted_next, next_states_tensor)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.effect_estimator.parameters(), 1.0)
        self.optimizer.step()
        
        self.is_trained = True
        
        return {"loss": loss.item()}
    
    def do_intervention(
        self,
        state: np.ndarray,
        intervention: Dict[str, float],
    ) -> np.ndarray:
        """
        Perform causal intervention (do-calculus).
        
        Args:
            state: Current state
            intervention: Intervention values
            
        Returns:
            Counterfactual state
        """
        # Simplified: return modified state
        modified_state = state.copy()
        
        # Apply interventions
        for var, value in intervention.items():
            # Map variable to state index (simplified)
            if "queue" in var:
                idx = hash(var) % len(state)
                modified_state[idx] = value
        
        return modified_state


class CausalAgent:
    """
    Causal Inference Agent for Traffic Control.
    
    Uses causal reasoning to make optimal decisions.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
    ):
        """
        Initialize causal agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            device: Device for computation
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Causal model
        self.causal_model = CausalModel(state_dim, action_dim, device)
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        ).to(device)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        
        self.is_trained = False
        logger.info("Initialized Causal Agent")
    
    def select_action(self, state: np.ndarray, use_causal: bool = True) -> int:
        """
        Select action using causal reasoning.
        
        Args:
            state: Current state
            use_causal: Whether to use causal reasoning
            
        Returns:
            Selected action
        """
        if not self.is_trained:
            return np.random.randint(0, self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if use_causal:
            # Evaluate actions using causal effects
            best_action = 0
            best_value = float('-inf')
            
            for action in range(self.action_dim):
                # Estimate causal effect
                effect = self.causal_model.estimate_causal_effect(state, action)
                
                # Value based on effect (simplified: sum of improvements)
                value = np.sum(effect)
                
                if value > best_value:
                    best_value = value
                    best_action = action
            
            return best_action
        else:
            # Use policy network
            with torch.no_grad():
                logits = self.policy(state_tensor)
                action = torch.argmax(logits, dim=1).item()
            return int(action)
    
    def train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        rewards: np.ndarray,
    ) -> Dict[str, float]:
        """
        Train causal agent.
        
        Args:
            states: Current states
            actions: Actions taken
            next_states: Next states
            rewards: Rewards received
            
        Returns:
            Training metrics
        """
        # Train causal model
        causal_metrics = self.causal_model.train_causal_model(states, actions, next_states)
        
        # Train policy
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        
        logits = self.policy(states_tensor)
        probs = torch.softmax(logits, dim=1)
        action_probs = probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
        
        # Policy gradient
        policy_loss = -torch.mean(torch.log(action_probs + 1e-8) * rewards_tensor)
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()
        
        self.is_trained = True
        
        return {
            "causal_loss": causal_metrics["loss"],
            "policy_loss": policy_loss.item(),
        }
    
    def save(self, path: str):
        """Save agent."""
        torch.save({
            'causal_model': self.causal_model.effect_estimator.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'is_trained': self.is_trained,
        }, path)
        logger.info(f"Saved Causal Agent to {path}")
    
    def load(self, path: str):
        """Load agent."""
        checkpoint = torch.load(path, map_location=self.device)
        self.causal_model.effect_estimator.load_state_dict(checkpoint['causal_model'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.is_trained = checkpoint.get('is_trained', False)
        logger.info(f"Loaded Causal Agent from {path}")

