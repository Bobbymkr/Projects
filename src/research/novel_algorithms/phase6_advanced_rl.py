"""
Phase 6: Advanced RL Techniques Implementation

Implements three state-of-the-art RL algorithms:
1. PPO (Proximal Policy Optimization) - 20-25% sample efficiency improvement
2. SAC (Soft Actor-Critic) - 15-20% performance, 25% sample efficiency
3. Rainbow DQN - 30-35% performance improvement

Based on OPTIMIZATION_ROADMAP.md Phase 6 specifications.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Dict, List, Optional, Any, Union
from dataclasses import dataclass
from collections import deque
import random
import math


# ============================================================================
# PPO (Proximal Policy Optimization)
# ============================================================================

@dataclass
class PPOConfig:
    """Configuration for PPO agent."""
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2  # Conservative clipping
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    train_epochs: int = 4  # Multiple epochs per batch
    batch_size: int = 64
    buffer_size: int = 2048
    device: str = None


class PPOPolicyNetwork(nn.Module):
    """Policy network for PPO."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_policy = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and value estimate."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc_policy(x)
        return logits, None  # Value will be computed separately


class PPOValueNetwork(nn.Module):
    """Separate value network for PPO."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning value estimate."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc_value(x)


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent.
    
    Features:
    - Clipped surrogate objective (ε=0.2)
    - Generalized Advantage Estimation (GAE) with λ=0.95, γ=0.99
    - Multiple training epochs per batch (4-10)
    - Separate value function network
    - Gradient clipping for stability
    
    Expected Impact: 20-25% sample efficiency, more stable training
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[PPOConfig] = None,
    ):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            config: PPO configuration
        """
        self.config = config or PPOConfig()
        self.device = self.config.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.policy_net = PPOPolicyNetwork(state_dim, action_dim).to(self.device)
        self.value_net = PPOValueNetwork(state_dim).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.config.lr)
        
        # Experience buffer
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
        }
        
        self.step_count = 0
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Union[int, Tuple[int, float, float]]:
        """
        Select action using current policy.
        
        Args:
            state: Current state vector
            deterministic: If True, use greedy action
            
        Returns:
            Action index, or tuple of (action, log_prob, value) if called internally
            The training loop handles tuple returns automatically
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, _ = self.policy_net(state_tensor)
            probs = F.softmax(logits, dim=-1)
            
            if deterministic:
                action = torch.argmax(probs, dim=-1).item()
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
            
            log_prob = F.log_softmax(logits, dim=-1)[0, action].item()
            value = self.value_net(state_tensor).item()
        
        # Store for PPO training (on-policy)
        self._last_log_prob = log_prob
        self._last_value = value
        
        # Return tuple for compatibility with training loop
        return (action, log_prob, value)
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """Store transition in buffer."""
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['dones'].append(done)
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value of next state (for bootstrap)
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.config.gamma * next_value - values[t]
                gae = delta + self.config.gamma * self.config.gae_lambda * gae
            
            advantages.insert(0, gae)
            next_value = values[t]
        
        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + np.array(values, dtype=np.float32)
        
        return advantages, returns
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform PPO training step.
        
        Returns:
            Dictionary with training metrics
        """
        if len(self.buffer['states']) < self.config.batch_size:
            return {'loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0}
        
        # Convert buffer to tensors
        states = torch.FloatTensor(np.array(self.buffer['states'])).to(self.device)
        actions = torch.LongTensor(self.buffer['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer['log_probs']).to(self.device)
        values = torch.FloatTensor(self.buffer['values']).to(self.device)
        
        # Compute GAE
        advantages, returns = self.compute_gae(
            self.buffer['rewards'],
            self.buffer['values'],
            self.buffer['dones'],
        )
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        # Multiple epochs
        for epoch in range(self.config.train_epochs):
            # Shuffle indices
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get current policy
                logits, _ = self.policy_net(batch_states)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                
                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                current_values = self.value_net(batch_states).squeeze()
                value_loss = F.mse_loss(current_values, batch_returns)
                
                # Entropy bonus
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
                
                # Update policy and value together (they share the same graph)
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.config.max_grad_norm)
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
        
        # Clear buffer
        self._clear_buffer()
        
        return {
            'loss': total_policy_loss + total_value_loss,
            'policy_loss': total_policy_loss / self.config.train_epochs,
            'value_loss': total_value_loss / self.config.train_epochs,
        }
    
    def _clear_buffer(self):
        """Clear experience buffer."""
        for key in self.buffer:
            self.buffer[key] = []
    
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Store experience (for compatibility with training loop).
        
        Note: PPO uses on-policy data, so we need to collect full trajectories.
        This method stores transitions using the log_prob and value from select_action.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
        """
        # Use stored log_prob and value from select_action
        log_prob = getattr(self, '_last_log_prob', 0.0)
        value = getattr(self, '_last_value', 0.0)
        
        self.store_transition(state, action, reward, value, log_prob, done)
    
    def reset(self):
        """Reset agent state."""
        self._clear_buffer()


# ============================================================================
# SAC (Soft Actor-Critic)
# ============================================================================

@dataclass
class SACConfig:
    """Configuration for SAC agent."""
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005  # Soft update coefficient
    alpha: float = 0.2  # Temperature parameter (entropy regularization)
    batch_size: int = 256
    buffer_size: int = 100000
    update_frequency: int = 1
    target_update_frequency: int = 1
    device: str = None


class SACActor(nn.Module):
    """Actor network for SAC (discrete actions)."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning action logits."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def get_action_and_log_prob(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability."""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(1)).squeeze(1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob


class SACCritic(nn.Module):
    """Critic network for SAC (Q-function)."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-value."""
        # For discrete actions, use one-hot encoding
        if action.dim() == 0:
            action = action.unsqueeze(0)
        action_one_hot = F.one_hot(action.long(), num_classes=self.action_dim).float()
        x = torch.cat([state, action_one_hot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SACAgent:
    """
    Soft Actor-Critic (SAC) Agent.
    
    Features:
    - Maximum entropy RL for better exploration
    - Off-policy learning (sample efficient)
    - Soft Q-learning with temperature parameter
    - Twin Q-networks for stability
    
    Expected Impact: 15-20% performance, 25% sample efficiency
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[SACConfig] = None,
    ):
        """
        Initialize SAC agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            config: SAC configuration
        """
        self.config = config or SACConfig()
        self.device = self.config.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.actor = SACActor(state_dim, action_dim).to(self.device)
        self.critic1 = SACCritic(state_dim, action_dim).to(self.device)
        self.critic2 = SACCritic(state_dim, action_dim).to(self.device)
        
        # Target networks
        self.critic1_target = SACCritic(state_dim, action_dim).to(self.device)
        self.critic2_target = SACCritic(state_dim, action_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.config.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.config.lr)
        
        # Replay buffer
        self.buffer = deque(maxlen=self.config.buffer_size)
        
        self.step_count = 0
        
    def select_action(self, state: np.ndarray, deterministic: bool = False, evaluate: bool = False) -> int:
        """
        Select action using current policy.
        
        Args:
            state: Current state vector
            deterministic: If True, use greedy action
            evaluate: Alias for deterministic (for compatibility)
            
        Returns:
            Selected action index
        """
        # Use evaluate parameter if provided, otherwise use deterministic
        use_deterministic = deterministic or evaluate
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, _ = self.actor.get_action_and_log_prob(state_tensor, use_deterministic)
        
        return action.item()
    
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform SAC training step.
        
        Returns:
            Dictionary with training metrics
        """
        if len(self.buffer) < self.config.batch_size:
            return {'loss': 0.0, 'actor_loss': 0.0, 'critic_loss': 0.0}
        
        # Sample batch
        batch = random.sample(self.buffer, self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        
        # Update critics
        with torch.no_grad():
            # Get next action and log prob from current policy
            next_actions, next_log_probs = self.actor.get_action_and_log_prob(next_states)
            
            # Compute target Q-values
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.config.alpha * next_log_probs.unsqueeze(1)
            target_q = rewards + (1 - dones) * self.config.gamma * q_next
        
        # Current Q-values
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        
        # Critic losses
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        actions_new, log_probs_new = self.actor.get_action_and_log_prob(states)
        q1_new = self.critic1(states, actions_new)
        q2_new = self.critic2(states, actions_new)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.config.alpha * log_probs_new.unsqueeze(1) - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.critic1, self.critic1_target, self.config.tau)
        self._soft_update(self.critic2, self.critic2_target, self.config.tau)
        
        return {
            'loss': critic1_loss.item() + critic2_loss.item() + actor_loss.item(),
            'actor_loss': actor_loss.item(),
            'critic_loss': (critic1_loss.item() + critic2_loss.item()) / 2,
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module, tau: float):
        """Soft update target network."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
    
    def reset(self):
        """Reset agent state."""
        pass


# ============================================================================
# Rainbow DQN
# ============================================================================

@dataclass
class RainbowDQNConfig:
    """Configuration for Rainbow DQN."""
    lr: float = 6.25e-5
    gamma: float = 0.99
    n_steps: int = 3  # Multi-step learning
    batch_size: int = 32
    buffer_size: int = 100000
    update_frequency: int = 4
    target_update_frequency: int = 8000
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: int = 25000
    alpha: float = 0.6  # PER priority exponent
    beta: float = 0.4  # PER importance sampling
    beta_increment: float = 0.001
    v_min: float = -10.0
    v_max: float = 10.0
    n_atoms: int = 51  # Distributional RL atoms
    device: str = None


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration."""
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        """Reset noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate scaled noise."""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DuelingDQN(nn.Module):
    """Dueling DQN architecture with distributional RL and noisy networks."""
    
    def __init__(self, state_dim: int, action_dim: int, n_atoms: int = 51, v_min: float = -10.0, v_max: float = 10.0):
        super().__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Feature layer
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, n_atoms),
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, action_dim * n_atoms),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-value distribution."""
        features = self.feature(x)
        value = self.value_stream(features).view(-1, 1, self.n_atoms)
        advantage = self.advantage_stream(features).view(-1, self.action_dim, self.n_atoms)
        
        # Combine value and advantage (dueling architecture)
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Apply softmax to get probability distribution
        return F.softmax(q_dist, dim=-1)
    
    def reset_noise(self):
        """Reset noise in noisy layers."""
        self.value_stream[2].reset_noise()
        self.advantage_stream[2].reset_noise()
    
    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Get Q-values from distribution."""
        dist = self.forward(x)
        support = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(x.device)
        q_values = (dist * support.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        return q_values


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay for Rainbow DQN."""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        self.epsilon = 0.01
        
        self.memory = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        """Add experience with maximum priority."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.position] = (state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Sample batch with importance sampling weights."""
        if len(self.memory) < batch_size:
            return None, None, None
        
        priorities = self.priorities[:len(self.memory)]
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        # Importance sampling weights
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.memory)


class RainbowDQNAgent:
    """
    Rainbow DQN Agent.
    
    Combines multiple DQN improvements:
    - Double DQN (target network)
    - Prioritized Experience Replay (PER)
    - Dueling Networks
    - Distributional RL (C51)
    - Noisy Networks (for exploration)
    - Multi-step learning (n=3)
    
    Expected Impact: 30-35% performance improvement
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[RainbowDQNConfig] = None,
    ):
        """
        Initialize Rainbow DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            config: Rainbow DQN configuration
        """
        self.config = config or RainbowDQNConfig()
        self.device = self.config.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.q_net = DuelingDQN(state_dim, action_dim, self.config.n_atoms, self.config.v_min, self.config.v_max).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim, self.config.n_atoms, self.config.v_min, self.config.v_max).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.config.lr)
        
        # Replay buffer
        self.buffer = PrioritizedReplayBuffer(self.config.buffer_size, self.config.alpha, self.config.beta)
        
        # Multi-step buffer
        self.n_step_buffer = deque(maxlen=self.config.n_steps)
        
        self.step_count = 0
        self.support = torch.linspace(self.config.v_min, self.config.v_max, self.config.n_atoms).to(self.device)
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Select action using epsilon-greedy or noisy network.
        
        Args:
            state: Current state vector
            evaluate: If True, use greedy policy
            
        Returns:
            Selected action index
        """
        if not evaluate:
            eps = self._epsilon()
            if np.random.random() < eps:
                return np.random.randint(0, self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_net.get_q_values(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def _epsilon(self) -> float:
        """Calculate current epsilon value."""
        if self.step_count >= self.config.eps_decay:
            return self.config.eps_end
        return self.config.eps_start - (self.config.eps_start - self.config.eps_end) * (self.step_count / self.config.eps_decay)
    
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store experience in n-step buffer and replay buffer."""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) == self.config.n_steps or done:
            # Compute n-step return
            n_step_reward = sum([self.n_step_buffer[i][2] * (self.config.gamma ** i) for i in range(len(self.n_step_buffer))])
            n_step_state = self.n_step_buffer[0][0]
            n_step_action = self.n_step_buffer[0][1]
            n_step_next_state = self.n_step_buffer[-1][3]
            n_step_done = self.n_step_buffer[-1][4]
            
            self.buffer.push(n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done)
            
            if done:
                self.n_step_buffer.clear()
        
        self.step_count += 1
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform Rainbow DQN training step.
        
        Returns:
            Dictionary with training metrics
        """
        if len(self.buffer) < self.config.batch_size:
            return {'loss': 0.0}
        
        # Sample batch
        batch, indices, weights = self.buffer.sample(self.config.batch_size)
        if batch is None:
            return {'loss': 0.0}
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q-distribution
        q_dist = self.q_net(states)
        action_q_dist = q_dist[range(self.config.batch_size), actions]
        
        # Target Q-distribution (Double DQN)
        with torch.no_grad():
            next_q_values = self.q_net.get_q_values(next_states)
            next_actions = next_q_values.argmax(dim=1)
            
            next_q_dist = self.target_net(next_states)
            next_action_q_dist = next_q_dist[range(self.config.batch_size), next_actions]
            
            # Project onto support
            target_dist = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.config.gamma * self.support.unsqueeze(0)
            target_dist = torch.clamp(target_dist, self.config.v_min, self.config.v_max)
            
            # Project distribution
            delta_z = (self.config.v_max - self.config.v_min) / (self.config.n_atoms - 1)
            tz = target_dist
            b = (tz - self.config.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            target_dist_proj = torch.zeros_like(next_action_q_dist)
            for i in range(self.config.batch_size):
                target_dist_proj[i].index_add_(0, l[i], next_action_q_dist[i] * (u[i].float() - b[i]))
                target_dist_proj[i].index_add_(0, u[i], next_action_q_dist[i] * (b[i] - l[i].float()))
        
        # Loss (cross-entropy between distributions)
        loss = -torch.sum(target_dist_proj * torch.log(action_q_dist + 1e-8), dim=1)
        loss = (weights * loss).mean()
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Update priorities
        with torch.no_grad():
            td_errors = torch.abs(action_q_dist.sum(dim=1) - target_dist_proj.sum(dim=1)).cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)
        
        # Reset noise
        self.q_net.reset_noise()
        
        # Update target network
        if self.step_count % self.config.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        return {'loss': loss.item()}
    
    def reset(self):
        """Reset agent state."""
        self.n_step_buffer.clear()
        self.q_net.reset_noise()

