"""
Distributional Reinforcement Learning.

Implements Phase 2.3 from OPTIMIZATION_ROADMAP.md:
- C51: 51-atom categorical distribution
- QR-DQN: Quantile regression (N=200 quantiles)
- IQN: Implicit quantile networks

Benefits:
- Better uncertainty estimation
- More stable learning
- Risk-aware decision making
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DistributionalRLConfig:
    """Configuration for distributional RL algorithms."""
    algorithm: str = "C51"  # "C51", "QR-DQN", "IQN"
    num_atoms: int = 51  # For C51
    num_quantiles: int = 200  # For QR-DQN
    v_min: float = -10.0  # Minimum value support
    v_max: float = 10.0  # Maximum value support
    risk_type: str = "neutral"  # "neutral", "risk_averse", "risk_seeking"


class C51Network(nn.Module):
    """
    C51 Network: Categorical DQN with 51 atoms.
    
    Outputs probability distribution over value atoms for each action.
    """
    
    def __init__(self, state_dim: int, action_dim: int, num_atoms: int = 51, 
                 v_min: float = -10.0, v_max: float = 10.0, hidden_dims: list = [128, 128]):
        """
        Initialize C51 network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            num_atoms: Number of atoms in distribution (51 for C51)
            v_min: Minimum value in support
            v_max: Maximum value in support
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        self.num_atoms = num_atoms
        self.action_dim = action_dim
        self.v_min = v_min
        self.v_max = v_max
        
        # Compute atom values
        self.atoms = torch.linspace(v_min, v_max, num_atoms)
        
        # Build network
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Output layer: action_dim * num_atoms (logits for each action-atom pair)
        layers.append(nn.Linear(input_dim, action_dim * num_atoms))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: State tensor [batch_size, state_dim]
            
        Returns:
            Logits tensor [batch_size, action_dim, num_atoms]
        """
        logits = self.net(x)
        # Reshape to [batch_size, action_dim, num_atoms]
        return logits.view(-1, self.action_dim, self.num_atoms)
    
    def get_distribution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability distribution over atoms.
        
        Args:
            x: State tensor
            
        Returns:
            Probability distribution [batch_size, action_dim, num_atoms]
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
    
    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values by computing expected value of distribution.
        
        Args:
            x: State tensor
            
        Returns:
            Q-values [batch_size, action_dim]
        """
        dist = self.get_distribution(x)
        # Expected value: sum(prob * atom_value)
        q_values = torch.sum(dist * self.atoms.to(x.device).unsqueeze(0).unsqueeze(0), dim=-1)
        return q_values


class QR_DQNNetwork(nn.Module):
    """
    Quantile Regression DQN Network.
    
    Outputs quantile values for each action.
    """
    
    def __init__(self, state_dim: int, action_dim: int, num_quantiles: int = 200,
                 hidden_dims: list = [128, 128]):
        """
        Initialize QR-DQN network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            num_quantiles: Number of quantiles (200 for QR-DQN)
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        self.num_quantiles = num_quantiles
        self.action_dim = action_dim
        
        # Compute quantile fractions (tau values)
        self.taus = torch.linspace(0.0, 1.0, num_quantiles + 1)[1:]  # [0.005, 0.015, ..., 0.995]
        
        # Build network
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Output layer: action_dim * num_quantiles
        layers.append(nn.Linear(input_dim, action_dim * num_quantiles))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: State tensor [batch_size, state_dim]
            
        Returns:
            Quantile values [batch_size, action_dim, num_quantiles]
        """
        quantiles = self.net(x)
        # Reshape to [batch_size, action_dim, num_quantiles]
        return quantiles.view(-1, self.action_dim, self.num_quantiles)
    
    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values by averaging quantiles.
        
        Args:
            x: State tensor
            
        Returns:
            Q-values [batch_size, action_dim]
        """
        quantiles = self.forward(x)
        # Average quantiles to get Q-value
        q_values = torch.mean(quantiles, dim=-1)
        return q_values


class IQNNetwork(nn.Module):
    """
    Implicit Quantile Network.
    
    Samples quantile fractions and outputs quantile values.
    """
    
    def __init__(self, state_dim: int, action_dim: int, num_quantiles: int = 8,
                 num_samples: int = 32, hidden_dims: list = [128, 128], 
                 embedding_dim: int = 64):
        """
        Initialize IQN network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            num_quantiles: Number of quantiles to sample
            num_samples: Number of samples for training
            hidden_dims: Hidden layer dimensions
            embedding_dim: Dimension of quantile embedding
        """
        super().__init__()
        self.num_quantiles = num_quantiles
        self.num_samples = num_samples
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        
        # Quantile embedding network
        self.quantile_embedding = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU(),
        )
        
        # State embedding
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.state_net = nn.Sequential(*layers)
        
        # Combined embedding dimension
        combined_dim = hidden_dims[-1] + embedding_dim
        
        # Output network
        output_layers = []
        for hidden_dim in hidden_dims:
            output_layers.append(nn.Linear(combined_dim, hidden_dim))
            output_layers.append(nn.ReLU())
            combined_dim = hidden_dim
        
        output_layers.append(nn.Linear(combined_dim, action_dim))
        self.output_net = nn.Sequential(*output_layers)
    
    def forward(self, x: torch.Tensor, tau: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: State tensor [batch_size, state_dim]
            tau: Quantile fractions [batch_size, num_quantiles] (optional)
            
        Returns:
            Quantile values [batch_size, action_dim, num_quantiles]
        """
        batch_size = x.size(0)
        
        # Sample quantiles if not provided
        if tau is None:
            tau = torch.rand(batch_size, self.num_quantiles, device=x.device)
        
        num_quantiles = tau.size(1)  # Use actual number of quantiles from tau
        
        # Embed quantiles
        tau_reshaped = tau.unsqueeze(-1)  # [batch_size, num_quantiles, 1]
        tau_embed = self.quantile_embedding(tau_reshaped)  # [batch_size, num_quantiles, embedding_dim]
        
        # Embed states
        state_embed = self.state_net(x)  # [batch_size, hidden_dim]
        state_embed = state_embed.unsqueeze(1).expand(-1, num_quantiles, -1)  # [batch_size, num_quantiles, hidden_dim]
        
        # Combine embeddings
        combined = torch.cat([state_embed, tau_embed], dim=-1)  # [batch_size, num_quantiles, combined_dim]
        
        # Reshape for network
        combined_flat = combined.view(-1, combined.size(-1))  # [batch_size * num_quantiles, combined_dim]
        output_flat = self.output_net(combined_flat)  # [batch_size * num_quantiles, action_dim]
        output = output_flat.view(batch_size, num_quantiles, self.action_dim)  # [batch_size, num_quantiles, action_dim]
        
        # Transpose to [batch_size, action_dim, num_quantiles]
        return output.transpose(1, 2)
    
    def get_q_values(self, x: torch.Tensor, num_samples: Optional[int] = None) -> torch.Tensor:
        """
        Get Q-values by sampling quantiles and averaging.
        
        Args:
            x: State tensor
            num_samples: Number of quantile samples (default: self.num_samples)
            
        Returns:
            Q-values [batch_size, action_dim]
        """
        if num_samples is None:
            num_samples = self.num_samples
        
        # Sample quantiles
        tau = torch.rand(x.size(0), num_samples, device=x.device)
        quantiles = self.forward(x, tau)  # [batch_size, action_dim, num_samples]
        
        # Average quantiles
        q_values = torch.mean(quantiles, dim=-1)
        return q_values


class DistributionalDQNAgent:
    """
    Distributional DQN Agent.
    
    Supports C51, QR-DQN, and IQN algorithms.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[DistributionalRLConfig] = None,
        device: str = "cpu",
    ):
        """
        Initialize distributional DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            config: Distributional RL configuration
            device: PyTorch device
        """
        self.config = config or DistributionalRLConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        # Create network based on algorithm
        if self.config.algorithm == "C51":
            self.policy_net = C51Network(
                state_dim, action_dim,
                num_atoms=self.config.num_atoms,
                v_min=self.config.v_min,
                v_max=self.config.v_max
            ).to(self.device)
            self.target_net = C51Network(
                state_dim, action_dim,
                num_atoms=self.config.num_atoms,
                v_min=self.config.v_min,
                v_max=self.config.v_max
            ).to(self.device)
        elif self.config.algorithm == "QR-DQN":
            self.policy_net = QR_DQNNetwork(
                state_dim, action_dim,
                num_quantiles=self.config.num_quantiles
            ).to(self.device)
            self.target_net = QR_DQNNetwork(
                state_dim, action_dim,
                num_quantiles=self.config.num_quantiles
            ).to(self.device)
        elif self.config.algorithm == "IQN":
            self.policy_net = IQNNetwork(state_dim, action_dim).to(self.device)
            self.target_net = IQNNetwork(state_dim, action_dim).to(self.device)
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")
        
        # Initialize target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-3)
        
        # Replay buffer (can use PER from Phase 2.2)
        from src.rl.prioritized_replay import PrioritizedReplayBuffer
        self.memory = PrioritizedReplayBuffer(capacity=100000, alpha=0.6, beta=0.4)
        
        # Training parameters
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 32
        self.steps = 0
        self.epsilon = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.995
        
        logger.info(f"Initialized {self.config.algorithm} agent")
    
    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """
        Select action using distributional Q-values.
        
        Args:
            state: Current state
            epsilon: Exploration rate (optional)
            
        Returns:
            Selected action
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net.get_q_values(state_tensor)
            return q_values.argmax().item()
    
    def push(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step with distributional loss.
        
        Returns:
            Training loss or None
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        transitions, indices, weights = self.memory.sample(self.batch_size)
        weights = torch.FloatTensor(weights).to(self.device)
        
        from collections import namedtuple
        Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        # Compute loss based on algorithm
        if self.config.algorithm == "C51":
            loss = self._train_c51(state_batch, action_batch, reward_batch, 
                                   next_state_batch, done_batch, weights)
        elif self.config.algorithm == "QR-DQN":
            loss = self._train_qr_dqn(state_batch, action_batch, reward_batch,
                                     next_state_batch, done_batch, weights)
        elif self.config.algorithm == "IQN":
            loss = self._train_iqn(state_batch, action_batch, reward_batch,
                                  next_state_batch, done_batch, weights)
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")
        
        # Update priorities
        with torch.no_grad():
            if self.config.algorithm == "C51":
                q_values = self.policy_net.get_q_values(state_batch)
            else:
                q_values = self.policy_net.get_q_values(state_batch)
            
            next_q_values = self.target_net.get_q_values(next_state_batch)
            td_errors = (q_values.gather(1, action_batch.unsqueeze(1)).squeeze() - 
                        (reward_batch + self.gamma * next_q_values.max(1)[0] * (1 - done_batch)))
            td_errors = td_errors.cpu().numpy()
            self.memory.update_priorities(indices, np.abs(td_errors))
        
        # Update target network
        with torch.no_grad():
            for target_param, policy_param in zip(self.target_net.parameters(), 
                                                  self.policy_net.parameters()):
                target_param.data.copy_(
                    self.tau * policy_param.data + (1 - self.tau) * target_param.data
                )
        
        # Decay epsilon
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
        self.steps += 1
        
        return loss.item() if loss is not None else None
    
    def _train_c51(self, states, actions, rewards, next_states, dones, weights):
        """Train C51 network."""
        # Get current distributions
        dist = self.policy_net.get_distribution(states)
        action_dist = dist[range(self.batch_size), actions]  # [batch_size, num_atoms]
        
        # Get target distributions
        with torch.no_grad():
            next_dist = self.target_net.get_distribution(next_states)
            next_q_values = self.target_net.get_q_values(next_states)
            next_actions = next_q_values.argmax(1)
            next_action_dist = next_dist[range(self.batch_size), next_actions]
            
            # Project target distribution
            target_atoms = rewards.unsqueeze(1) + self.gamma * self.policy_net.atoms.to(self.device).unsqueeze(0) * (1 - dones.unsqueeze(1))
            target_atoms = torch.clamp(target_atoms, self.config.v_min, self.config.v_max)
            
            # Project onto atom support
            atom_delta = (self.config.v_max - self.config.v_min) / (self.config.num_atoms - 1)
            target_atoms_shifted = (target_atoms - self.config.v_min) / atom_delta
            lower = target_atoms_shifted.floor().long()
            upper = target_atoms_shifted.ceil().long()
            
            # Clamp indices
            lower = torch.clamp(lower, 0, self.config.num_atoms - 1)
            upper = torch.clamp(upper, 0, self.config.num_atoms - 1)
            
            # Compute projection weights
            m = torch.zeros(self.batch_size, self.config.num_atoms, device=self.device)
            for i in range(self.batch_size):
                for j in range(self.config.num_atoms):
                    if dones[i]:
                        m[i, j] = 1.0 if j == (target_atoms[i, 0] - self.config.v_min) / atom_delta else 0.0
                    else:
                        lower_idx = lower[i, j].item()
                        upper_idx = upper[i, j].item()
                        lower_weight = upper[i, j] - target_atoms_shifted[i, j]
                        upper_weight = target_atoms_shifted[i, j] - lower[i, j]
                        m[i, lower_idx] += next_action_dist[i, j] * lower_weight
                        m[i, upper_idx] += next_action_dist[i, j] * upper_weight
        
        # Compute loss (cross-entropy between distributions)
        loss = -torch.sum(m * torch.log(action_dist + 1e-8), dim=1)
        weighted_loss = (loss * weights.squeeze()).mean()
        
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        return weighted_loss
    
    def _train_qr_dqn(self, states, actions, rewards, next_states, dones, weights):
        """Train QR-DQN network."""
        # Get current quantiles
        quantiles = self.policy_net(states)
        action_quantiles = quantiles[range(self.batch_size), actions]  # [batch_size, num_quantiles]
        
        # Get target quantiles
        with torch.no_grad():
            next_quantiles = self.target_net(next_states)
            next_q_values = self.target_net.get_q_values(next_states)
            next_actions = next_q_values.argmax(1)
            next_action_quantiles = next_quantiles[range(self.batch_size), next_actions]
            
            # Compute target quantiles
            target_quantiles = rewards.unsqueeze(1) + self.gamma * next_action_quantiles * (1 - dones.unsqueeze(1))
        
        # Compute quantile regression loss
        taus = self.policy_net.taus.to(self.device).unsqueeze(0)  # [1, num_quantiles]
        td_errors = target_quantiles.unsqueeze(1) - action_quantiles.unsqueeze(2)  # [batch_size, num_quantiles, num_quantiles]
        
        # Huber loss for quantile regression
        huber_loss = torch.where(
            torch.abs(td_errors) < 1.0,
            0.5 * td_errors ** 2,
            torch.abs(td_errors) - 0.5
        )
        
        # Quantile loss
        quantile_loss = torch.abs(taus - (td_errors < 0).float()) * huber_loss
        loss = quantile_loss.mean(dim=(1, 2))
        weighted_loss = (loss * weights.squeeze()).mean()
        
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        return weighted_loss
    
    def _train_iqn(self, states, actions, rewards, next_states, dones, weights):
        """Train IQN network."""
        # Sample quantiles for training
        tau = torch.rand(self.batch_size, self.policy_net.num_samples, device=self.device)
        
        # Get current quantiles
        quantiles = self.policy_net(states, tau)
        action_quantiles = quantiles[range(self.batch_size), actions]  # [batch_size, num_samples]
        
        # Get target quantiles
        with torch.no_grad():
            next_tau = torch.rand(self.batch_size, self.policy_net.num_samples, device=self.device)
            next_quantiles = self.target_net(next_states, next_tau)
            next_q_values = self.target_net.get_q_values(next_states)
            next_actions = next_q_values.argmax(1)
            next_action_quantiles = next_quantiles[range(self.batch_size), next_actions]
            
            # Compute target quantiles
            target_quantiles = rewards.unsqueeze(1) + self.gamma * next_action_quantiles * (1 - dones.unsqueeze(1))
        
        # Compute quantile regression loss
        td_errors = target_quantiles - action_quantiles
        
        # Huber loss
        huber_loss = torch.where(
            torch.abs(td_errors) < 1.0,
            0.5 * td_errors ** 2,
            torch.abs(td_errors) - 0.5
        )
        
        # Quantile loss
        quantile_loss = torch.abs(tau - (td_errors < 0).float()) * huber_loss
        loss = quantile_loss.mean(dim=1)
        weighted_loss = (loss * weights.squeeze()).mean()
        
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        return weighted_loss
    
    def get_uncertainty(self, state: np.ndarray) -> Dict[str, float]:
        """
        Get uncertainty estimates from distribution.
        
        Args:
            state: Current state
            
        Returns:
            Dictionary with uncertainty metrics
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if self.config.algorithm == "C51":
                dist = self.policy_net.get_distribution(state_tensor)
                q_values = self.policy_net.get_q_values(state_tensor)
                
                # Compute variance of distribution
                atoms = self.policy_net.atoms.to(self.device)
                variances = []
                for a in range(self.action_dim):
                    mean = torch.sum(dist[0, a] * atoms)
                    variance = torch.sum(dist[0, a] * (atoms - mean) ** 2)
                    variances.append(variance.item())
                
                return {
                    "variance": np.mean(variances),
                    "std": np.sqrt(np.mean(variances)),
                    "q_values": q_values[0].cpu().numpy().tolist()
                }
            
            elif self.config.algorithm == "QR-DQN":
                quantiles = self.policy_net(state_tensor)
                q_values = self.policy_net.get_q_values(state_tensor)
                
                # Compute inter-quantile range as uncertainty
                iqr = quantiles[0, :, -1] - quantiles[0, :, 0]  # 75th - 25th quantile
                
                return {
                    "iqr": iqr.mean().item(),
                    "std": quantiles[0].std(dim=-1).mean().item(),
                    "q_values": q_values[0].cpu().numpy().tolist()
                }
            
            elif self.config.algorithm == "IQN":
                # Sample multiple times for uncertainty
                q_samples = []
                for _ in range(10):
                    q = self.policy_net.get_q_values(state_tensor, num_samples=32)
                    q_samples.append(q[0].cpu().numpy())
                
                q_samples = np.array(q_samples)
                q_mean = q_samples.mean(axis=0)
                q_std = q_samples.std(axis=0)
                
                return {
                    "std": q_std.mean(),
                    "variance": (q_std ** 2).mean(),
                    "q_values": q_mean.tolist()
                }
            
            else:
                return {"uncertainty": 0.0}

