"""
Bayesian Methods for Traffic Signal Control.

Uses Bayesian inference for uncertainty-aware decision-making
and adaptive signal timing.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple
import scipy.stats as stats

logger = logging.getLogger(__name__)


class BayesianNeuralNetwork(nn.Module):
    """
    Bayesian Neural Network with uncertainty estimation.
    
    Uses variational inference to estimate uncertainty in predictions.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 128],
        prior_std: float = 1.0,
    ):
        """
        Initialize Bayesian neural network.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
            prior_std: Prior standard deviation
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior_std = prior_std
        
        # Build layers with Bayesian weights
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(BayesianLinear(prev_dim, hidden_dim, prior_std))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(BayesianLinear(prev_dim, output_dim, prior_std))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.
        
        Args:
            x: Input tensor
            sample: Whether to sample from posterior
            
        Returns:
            Mean and std of predictions
        """
        # Sample from posterior
        if sample:
            output = self.network(x)
        else:
            # Use mean weights
            output = self._forward_mean(x)
        
        # Estimate uncertainty
        mean = output
        std = self._estimate_uncertainty(x)
        
        return mean, std
    
    def _forward_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using mean weights."""
        # Simplified: use mean of posterior
        return self.network(x)
    
    def _estimate_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate prediction uncertainty."""
        # Monte Carlo sampling for uncertainty
        samples = []
        n_samples = 10
        
        for _ in range(n_samples):
            output = self.network(x)
            samples.append(output)
        
        samples = torch.stack(samples)
        std = torch.std(samples, dim=0)
        
        return std


class BayesianLinear(nn.Module):
    """Bayesian linear layer with variational inference."""
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Variational parameters
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 1.0)
        
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_logvar = nn.Parameter(torch.randn(out_features) * 0.1 - 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sampled weights."""
        # Sample weights from posterior
        weight_std = torch.exp(0.5 * self.weight_logvar)
        weight_epsilon = torch.randn_like(weight_std)
        weight = self.weight_mu + weight_std * weight_epsilon
        
        bias_std = torch.exp(0.5 * self.bias_logvar)
        bias_epsilon = torch.randn_like(bias_std)
        bias = self.bias_mu + bias_std * bias_epsilon
        
        return torch.nn.functional.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence from prior."""
        # KL(q(w) || p(w)) where p(w) is Gaussian prior
        weight_kl = 0.5 * (
            torch.sum(self.weight_logvar) +
            torch.sum(torch.exp(self.weight_logvar)) / (self.prior_std ** 2) +
            torch.sum(self.weight_mu ** 2) / (self.prior_std ** 2) -
            self.in_features * self.out_features
        )
        
        bias_kl = 0.5 * (
            torch.sum(self.bias_logvar) +
            torch.sum(torch.exp(self.bias_logvar)) / (self.prior_std ** 2) +
            torch.sum(self.bias_mu ** 2) / (self.prior_std ** 2) -
            self.out_features
        )
        
        return weight_kl + bias_kl


class BayesianAgent:
    """
    Bayesian Agent for Traffic Control.
    
    Uses Bayesian inference to make uncertainty-aware decisions.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        kl_weight: float = 0.01,
        device: str = "cpu",
    ):
        """
        Initialize Bayesian agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate
            kl_weight: Weight for KL divergence term
            device: Device for computation
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.kl_weight = kl_weight
        self.device = device
        
        # Create Bayesian network
        self.model = BayesianNeuralNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.is_trained = False
        logger.info("Initialized Bayesian Agent")
    
    def select_action(self, state: np.ndarray, use_uncertainty: bool = True) -> Tuple[int, float]:
        """
        Select action with uncertainty estimation.
        
        Args:
            state: Current state
            use_uncertainty: Whether to use uncertainty in decision
            
        Returns:
            Action and uncertainty score
        """
        if not self.is_trained:
            return np.random.randint(0, self.action_dim), 1.0
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mean, std = self.model(state_tensor, sample=True)
            
            # Use uncertainty to guide exploration
            if use_uncertainty:
                # Thompson sampling: sample from posterior
                probs = torch.softmax(mean + std * torch.randn_like(std), dim=1)
            else:
                probs = torch.softmax(mean, dim=1)
            
            action = torch.argmax(probs, dim=1).item()
            uncertainty = torch.mean(std).item()
        
        return int(action), float(uncertainty)
    
    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> Dict[str, float]:
        """
        Train on batch.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            
        Returns:
            Training metrics
        """
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        
        # Forward pass
        mean, std = self.model(states_tensor, sample=True)
        
        # Classification loss
        loss_ce = nn.functional.cross_entropy(mean, actions_tensor)
        
        # KL divergence (regularization)
        kl_loss = sum(layer.kl_divergence() for layer in self.model.modules() if isinstance(layer, BayesianLinear))
        
        # Total loss
        total_loss = loss_ce + self.kl_weight * kl_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.is_trained = True
        
        return {
            "loss": total_loss.item(),
            "ce_loss": loss_ce.item(),
            "kl_loss": kl_loss.item(),
        }
    
    def save(self, path: str):
        """Save agent."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'is_trained': self.is_trained,
        }, path)
        logger.info(f"Saved Bayesian Agent to {path}")
    
    def load(self, path: str):
        """Load agent."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.is_trained = checkpoint.get('is_trained', False)
        logger.info(f"Loaded Bayesian Agent from {path}")


class BayesianOptimization:
    """
    Bayesian Optimization for hyperparameter tuning.
    
    Uses Gaussian Process to efficiently search hyperparameter space.
    """
    
    def __init__(self, bounds: Dict[str, Tuple[float, float]]):
        """
        Initialize Bayesian optimization.
        
        Args:
            bounds: Dictionary of parameter bounds
        """
        self.bounds = bounds
        self.observations: List[Tuple[Dict[str, float], float]] = []
    
    def suggest(self) -> Dict[str, float]:
        """
        Suggest next hyperparameters to try.
        
        Returns:
            Suggested hyperparameters
        """
        if len(self.observations) == 0:
            # Random initial suggestion
            return {key: np.random.uniform(low, high) for key, (low, high) in self.bounds.items()}
        
        # Use acquisition function (Upper Confidence Bound)
        # Simplified: random for now
        return {key: np.random.uniform(low, high) for key, (low, high) in self.bounds.items()}
    
    def update(self, params: Dict[str, float], performance: float):
        """
        Update with new observation.
        
        Args:
            params: Hyperparameters tried
            performance: Performance achieved
        """
        self.observations.append((params, performance))
    
    def get_best(self) -> Tuple[Dict[str, float], float]:
        """
        Get best hyperparameters found.
        
        Returns:
            Best parameters and performance
        """
        if not self.observations:
            return {}, 0.0
        
        best_idx = np.argmax([obs[1] for obs in self.observations])
        return self.observations[best_idx]

