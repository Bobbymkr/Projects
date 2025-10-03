import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from dataclasses import dataclass

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

@dataclass
class DQNConfig:
    batch_size: int = 32
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.995
    tau: float = 0.005
    lr: float = 1e-3
    device: str = None  # Will be set to 'cuda' if available, else 'cpu'

class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions.
    
    Stores agent experiences and provides random sampling for training.
    Uses a circular buffer with maximum capacity.
    """
    def __init__(self, capacity=100000):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.memory = deque([], maxlen=capacity)
        
    def push(self, *args):
        """Store a transition in the buffer.
        
        Args:
            *args: Transition tuple (state, action, reward, next_state, done)
        """
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        """Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            List of sampled transitions
        """
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQNetwork(nn.Module):
    """Deep Q-Network for value function approximation.
    
    Three-layer fully connected neural network that maps states to Q-values.
    Uses ReLU activations and outputs Q-values for each possible action.
    """
    def __init__(self, state_dim, action_dim):
        """Initialize the DQ-Network.
        
        Args:
            state_dim: Dimensionality of the state space
            action_dim: Number of possible actions
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Q-values for each action
        """
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, cfg=None):
        """Initialize the DQN agent.
        
        Args:
            state_dim: Dimensionality of the state space
            action_dim: Number of possible actions
            cfg: Configuration object with hyperparameters
        """
        self.cfg = cfg or DQNConfig()
        self.cfg.device = self.cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = self.cfg.eps_start
        
        # Q Networks
        self.policy_net = DQNetwork(state_dim, action_dim).to(self.cfg.device)
        self.target_net = DQNetwork(state_dim, action_dim).to(self.cfg.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg.lr)
        self.memory = ReplayBuffer()
        self.steps = 0
        
    def select_action(self, state):
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            
        Returns:
            Selected action index
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.cfg.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
        
    def push(self, state, action, reward, next_state, done):
        """Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Episode termination flag
        """
        self.memory.push(state, action, reward, next_state, done)
        
    def train_step(self):
        """Perform one training step using experience replay.
        
        Returns:
            Training loss value, or None if insufficient data
        """
        if len(self.memory) < self.cfg.batch_size:
            return None
        
        transitions = self.memory.sample(self.cfg.batch_size)
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
            
        # Compute loss and update
        loss = nn.functional.smooth_l1_loss(current_q_values.squeeze(), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        # Update target network
        with torch.no_grad():
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_((1 - self.cfg.tau) * target_param.data + self.cfg.tau * policy_param.data)
                
        # Decay epsilon
        self.epsilon = max(self.cfg.eps_end, self.epsilon * self.cfg.eps_decay)
        self.steps += 1
        
        return loss.item()
    
    def save(self, path):
        """Save agent state to file.
        
        Args:
            path: File path to save checkpoint
        """
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'cfg': self.cfg
        }
        torch.save(checkpoint, path)
        
    def load(self, path):
        """Load agent state from file.
        
        Args:
            path: File path to load checkpoint from
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.cfg = checkpoint['cfg']
