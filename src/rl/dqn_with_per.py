"""
DQN Agent with Prioritized Experience Replay (PER).

Enhanced DQN agent that uses PER for better sample efficiency.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Optional

from src.rl.pytorch_dqn import DQNAgent, DQNConfig, DQNetwork
from src.rl.prioritized_replay import PrioritizedReplayBuffer, AdaptivePER, Transition


@dataclass
class PERConfig:
    """Configuration for Prioritized Experience Replay."""
    enabled: bool = True
    alpha: float = 0.6  # Priority exponent
    beta: float = 0.4  # Importance sampling exponent
    beta_increment: float = 0.001
    epsilon: float = 1e-6
    adaptive: bool = False  # Use adaptive PER


class DQNAgentWithPER(DQNAgent):
    """
    DQN Agent with Prioritized Experience Replay.
    
    Extends DQNAgent to use PER instead of uniform replay buffer.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: Optional[DQNConfig] = None,
        per_config: Optional[PERConfig] = None,
    ):
        """
        Initialize DQN agent with PER.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            cfg: DQN configuration
            per_config: PER configuration
        """
        # Initialize base DQN agent
        super().__init__(state_dim, action_dim, cfg)
        
        # PER configuration
        self.per_config = per_config or PERConfig()
        
        # Replace uniform replay buffer with prioritized buffer
        if self.per_config.enabled:
            from src.rl.prioritized_replay import PrioritizedReplayBuffer, AdaptivePER
            
            self.memory = PrioritizedReplayBuffer(
                capacity=100000,
                alpha=self.per_config.alpha,
                beta=self.per_config.beta,
                beta_increment=self.per_config.beta_increment,
                epsilon=self.per_config.epsilon,
            )
            
            # Adaptive PER if enabled
            if self.per_config.adaptive:
                self.adaptive_per = AdaptivePER(self.memory)
            else:
                self.adaptive_per = None
        else:
            self.adaptive_per = None
    
    def train_step(self):
        """
        Perform one training step using prioritized experience replay.
        
        Returns:
            Training loss value, or None if insufficient data
        """
        if len(self.memory) < self.cfg.batch_size:
            return None
        
        # Sample with priorities and importance weights
        if self.per_config.enabled:
            transitions, indices, weights = self.memory.sample(self.cfg.batch_size)
            weights = torch.FloatTensor(weights).to(self.cfg.device).unsqueeze(1)
        else:
            transitions = self.memory.sample(self.cfg.batch_size)
            indices = None
            weights = torch.ones(self.cfg.batch_size, 1).to(self.cfg.device)
        
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.cfg.device)
        action_batch = torch.LongTensor(batch.action).to(self.cfg.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.cfg.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.cfg.device)
        done_batch = torch.FloatTensor(batch.done).to(self.cfg.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            expected_q_values = reward_batch + self.cfg.gamma * next_q_values * (1 - done_batch)
        
        # Compute TD-errors
        td_errors = (current_q_values.squeeze() - expected_q_values).detach().cpu().numpy()
        
        # Compute loss with importance sampling weights
        loss = nn.functional.smooth_l1_loss(
            current_q_values.squeeze(),
            expected_q_values,
            reduction='none'
        )
        weighted_loss = (loss * weights.squeeze()).mean()
        
        # Update network
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        # Update priorities in PER buffer
        if self.per_config.enabled and indices is not None:
            self.memory.update_priorities(indices, np.abs(td_errors))
        
        # Update target network
        with torch.no_grad():
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(
                    (1 - self.cfg.tau) * target_param.data + self.cfg.tau * policy_param.data
                )
        
        # Decay epsilon
        self.epsilon = max(self.cfg.eps_end, self.epsilon * self.cfg.eps_decay)
        self.steps += 1
        
        # Update adaptive PER if enabled
        if self.adaptive_per is not None:
            learning_progress = abs(np.mean(td_errors))
            self.adaptive_per.update_alpha(learning_progress)
        
        return weighted_loss.item()
    
    def push(self, state, action, reward, next_state, done):
        """
        Store experience in prioritized replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Episode termination flag
        """
        if self.per_config.enabled:
            self.memory.push(state, action, reward, next_state, done)
        else:
            # Fallback to base class method
            super().push(state, action, reward, next_state, done)

