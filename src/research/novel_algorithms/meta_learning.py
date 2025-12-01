"""
Meta-Learning for Traffic Signal Control.

Uses meta-learning (learning to learn) for fast adaptation
to new traffic patterns and scenarios.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple
import copy

logger = logging.getLogger(__name__)


class MAMLNetwork(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML) Network.
    
    Learns initialization that can quickly adapt to new tasks.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128],
    ):
        """
        Initialize MAML network.
        
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
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class MAMLAgent:
    """
    Model-Agnostic Meta-Learning Agent.
    
    Learns to quickly adapt to new traffic scenarios.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        meta_lr: float = 1e-3,
        inner_lr: float = 0.01,
        inner_steps: int = 1,
        device: str = "cpu",
    ):
        """
        Initialize MAML agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            meta_lr: Meta-learning rate
            inner_lr: Inner loop learning rate
            inner_steps: Number of inner loop steps
            device: Device for computation
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = device
        
        # Meta-model
        self.meta_model = MAMLNetwork(state_dim, action_dim).to(device)
        self.meta_optimizer = optim.Adam(self.meta_model.parameters(), lr=meta_lr)
        
        self.is_trained = False
        logger.info("Initialized MAML Agent")
    
    def adapt(
        self,
        support_set: List[Tuple[np.ndarray, int]],
        num_steps: Optional[int] = None,
    ) -> nn.Module:
        """
        Adapt model to support set (few-shot learning).
        
        Args:
            support_set: List of (state, action) pairs
            num_steps: Number of adaptation steps
            
        Returns:
            Adapted model
        """
        if num_steps is None:
            num_steps = self.inner_steps
        
        # Clone model
        adapted_model = copy.deepcopy(self.meta_model)
        adapted_optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        # Prepare data
        states = torch.FloatTensor([s[0] for s in support_set]).to(self.device)
        actions = torch.LongTensor([s[1] for s in support_set]).to(self.device)
        
        # Inner loop: adapt to support set
        for step in range(num_steps):
            logits = adapted_model(states)
            loss = nn.functional.cross_entropy(logits, actions)
            
            adapted_optimizer.zero_grad()
            loss.backward()
            adapted_optimizer.step()
        
        return adapted_model
    
    def meta_train(
        self,
        tasks: List[List[Tuple[np.ndarray, int, np.ndarray, int]]],
        num_meta_iterations: int = 100,
    ) -> Dict[str, Any]:
        """
        Meta-train on multiple tasks.
        
        Args:
            tasks: List of tasks, each containing (state, action, next_state, next_action) tuples
            num_meta_iterations: Number of meta-iterations
            
        Returns:
            Training metrics
        """
        logger.info(f"Meta-training on {len(tasks)} tasks")
        
        meta_losses = []
        
        for iteration in range(num_meta_iterations):
            # Sample batch of tasks
            task_batch = np.random.choice(tasks, size=min(4, len(tasks)), replace=False)
            
            meta_loss = 0.0
            
            for task in task_batch:
                # Split into support and query sets
                split_idx = len(task) // 2
                support_set = [(s, a) for s, a, _, _ in task[:split_idx]]
                query_set = task[split_idx:]
                
                # Adapt to support set
                adapted_model = self.adapt(support_set)
                
                # Evaluate on query set
                query_states = torch.FloatTensor([s for s, _, _, _ in query_set]).to(self.device)
                query_actions = torch.LongTensor([a for _, a, _, _ in query_set]).to(self.device)
                
                query_logits = adapted_model(query_states)
                query_loss = nn.functional.cross_entropy(query_logits, query_actions)
                
                meta_loss += query_loss
            
            # Meta-update
            meta_loss = meta_loss / len(task_batch)
            
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()
            
            meta_losses.append(meta_loss.item())
            
            if (iteration + 1) % 10 == 0:
                logger.info(f"Meta-iteration {iteration + 1}/{num_meta_iterations}, Loss: {meta_loss.item():.4f}")
        
        self.is_trained = True
        
        return {
            "meta_losses": meta_losses,
            "final_loss": meta_losses[-1] if meta_losses else 0.0,
        }
    
    def select_action(
        self,
        state: np.ndarray,
        support_set: Optional[List[Tuple[np.ndarray, int]]] = None,
    ) -> int:
        """
        Select action, optionally using adapted model.
        
        Args:
            state: Current state
            support_set: Optional support set for adaptation
            
        Returns:
            Selected action
        """
        if not self.is_trained:
            return np.random.randint(0, self.action_dim)
        
        model = self.meta_model
        
        # Adapt if support set provided
        if support_set and len(support_set) > 0:
            model = self.adapt(support_set)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = model(state_tensor)
            action = torch.argmax(logits, dim=1).item()
        
        return int(action)
    
    def save(self, path: str):
        """Save agent."""
        torch.save({
            'meta_model_state_dict': self.meta_model.state_dict(),
            'is_trained': self.is_trained,
        }, path)
        logger.info(f"Saved MAML Agent to {path}")
    
    def load(self, path: str):
        """Load agent."""
        checkpoint = torch.load(path, map_location=self.device)
        self.meta_model.load_state_dict(checkpoint['meta_model_state_dict'])
        self.is_trained = checkpoint.get('is_trained', False)
        logger.info(f"Loaded MAML Agent from {path}")


class ReptileAgent:
    """
    Reptile Meta-Learning Agent.
    
    Simpler alternative to MAML for meta-learning.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        meta_lr: float = 1e-3,
        inner_lr: float = 0.01,
        inner_steps: int = 1,
        device: str = "cpu",
    ):
        """
        Initialize Reptile agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            meta_lr: Meta-learning rate
            inner_lr: Inner loop learning rate
            inner_steps: Number of inner loop steps
            device: Device for computation
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = device
        
        # Meta-model
        self.meta_model = MAMLNetwork(state_dim, action_dim).to(device)
        self.meta_optimizer = optim.Adam(self.meta_model.parameters(), lr=meta_lr)
        
        self.is_trained = False
        logger.info("Initialized Reptile Agent")
    
    def reptile_step(
        self,
        task_data: List[Tuple[np.ndarray, int]],
    ) -> Dict[str, torch.Tensor]:
        """
        Perform one Reptile step.
        
        Args:
            task_data: Task data (state, action) pairs
            
        Returns:
            Parameter update
        """
        # Clone model
        task_model = copy.deepcopy(self.meta_model)
        task_optimizer = optim.SGD(task_model.parameters(), lr=self.inner_lr)
        
        # Inner loop
        states = torch.FloatTensor([s[0] for s in task_data]).to(self.device)
        actions = torch.LongTensor([s[1] for s in task_data]).to(self.device)
        
        for step in range(self.inner_steps):
            logits = task_model(states)
            loss = nn.functional.cross_entropy(logits, actions)
            
            task_optimizer.zero_grad()
            loss.backward()
            task_optimizer.step()
        
        # Compute parameter difference
        param_diff = {}
        for name, param in task_model.named_parameters():
            meta_param = dict(self.meta_model.named_parameters())[name]
            param_diff[name] = param - meta_param
        
        return param_diff
    
    def meta_train(
        self,
        tasks: List[List[Tuple[np.ndarray, int]]],
        num_meta_iterations: int = 100,
    ) -> Dict[str, Any]:
        """
        Meta-train using Reptile.
        
        Args:
            tasks: List of tasks
            num_meta_iterations: Number of meta-iterations
            
        Returns:
            Training metrics
        """
        logger.info(f"Reptile meta-training on {len(tasks)} tasks")
        
        for iteration in range(num_meta_iterations):
            # Sample task
            task = np.random.choice(tasks)
            
            # Reptile step
            param_diff = self.reptile_step(task)
            
            # Meta-update: move towards task-adapted parameters
            for name, diff in param_diff.items():
                param = dict(self.meta_model.named_parameters())[name]
                param.data += self.inner_lr * diff
            
            if (iteration + 1) % 10 == 0:
                logger.info(f"Reptile iteration {iteration + 1}/{num_meta_iterations}")
        
        self.is_trained = True
        
        return {"status": "completed"}
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action."""
        if not self.is_trained:
            return np.random.randint(0, self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.meta_model(state_tensor)
            action = torch.argmax(logits, dim=1).item()
        
        return int(action)
    
    def save(self, path: str):
        """Save agent."""
        torch.save({
            'meta_model_state_dict': self.meta_model.state_dict(),
            'is_trained': self.is_trained,
        }, path)
        logger.info(f"Saved Reptile Agent to {path}")
    
    def load(self, path: str):
        """Load agent."""
        checkpoint = torch.load(path, map_location=self.device)
        self.meta_model.load_state_dict(checkpoint['meta_model_state_dict'])
        self.is_trained = checkpoint.get('is_trained', False)
        logger.info(f"Loaded Reptile Agent from {path}")

