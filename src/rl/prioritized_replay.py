"""
Prioritized Experience Replay (PER).

Implements Phase 2.2 from OPTIMIZATION_ROADMAP.md:
- TD-Error Prioritization (α=0.6)
- Importance Sampling (β=0.4 to 1.0, annealing)
- Hindsight Experience Replay (HER) support
- Adaptive PER
"""

import numpy as np
import random
from typing import List, Tuple, Optional, Any
from collections import namedtuple
import logging

logger = logging.getLogger(__name__)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    
    Stores transitions with priorities based on TD-error and samples
    with importance sampling correction.
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
    ):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Amount to increment beta per sample
            epsilon: Small constant to ensure non-zero priorities
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_start = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # Storage
        self.buffer: List[Transition] = []
        self.priorities: List[float] = []
        self.max_priority = 1.0
        
        # Position tracking
        self.position = 0
        
        logger.info(f"Initialized PrioritizedReplayBuffer (capacity={capacity}, alpha={alpha}, beta={beta})")
    
    def push(self, state, action, reward, next_state, done):
        """
        Add transition to buffer with maximum priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
        """
        transition = Transition(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(self.max_priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = self.max_priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """
        Sample batch of transitions with importance sampling weights.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (transitions, indices, importance_weights)
        """
        if len(self.buffer) < batch_size:
            # Not enough samples, return all available
            indices = list(range(len(self.buffer)))
            transitions = [self.buffer[i] for i in indices]
            weights = np.ones(len(transitions))
            return transitions, np.array(indices), weights
        
        # Compute sampling probabilities
        priorities = np.array(self.priorities[:len(self.buffer)])
        probabilities = priorities ** self.alpha
        probabilities = probabilities / probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probabilities)
        transitions = [self.buffer[i] for i in indices]
        
        # Compute importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize to [0, 1]
        
        # Update beta (annealing)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return transitions, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on TD-errors.
        
        Args:
            indices: Indices of transitions to update
            td_errors: TD-errors for those transitions
        """
        # Compute priorities from TD-errors
        priorities = np.abs(td_errors) + self.epsilon
        
        # Update priorities
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority
        
        # Update max priority
        self.max_priority = max(self.max_priority, priorities.max())
    
    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def get_beta(self) -> float:
        """Get current beta value."""
        return self.beta
    
    def reset_beta(self):
        """Reset beta to initial value."""
        self.beta = self.beta_start


class HindsightExperienceReplay:
    """
    Hindsight Experience Replay (HER) for goal-conditioned RL.
    
    Relabels failed trajectories with achieved goals as desired goals.
    """
    
    def __init__(self, replay_buffer: PrioritizedReplayBuffer, strategy: str = "future"):
        """
        Initialize HER.
        
        Args:
            replay_buffer: Base replay buffer
            strategy: HER strategy ("future", "final", "episode")
        """
        self.replay_buffer = replay_buffer
        self.strategy = strategy
        self.episode_buffer: List[Transition] = []
    
    def store_transition(self, state, action, reward, next_state, done, goal=None):
        """
        Store transition, potentially with goal relabeling.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
            goal: Goal state (optional)
        """
        transition = Transition(state, action, reward, next_state, done)
        self.episode_buffer.append(transition)
        
        # Also store in base buffer
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        # If episode done, perform HER relabeling
        if done:
            self._relabel_episode()
            self.episode_buffer = []
    
    def _relabel_episode(self):
        """
        Relabel episode with hindsight goals.
        
        For traffic control, we can relabel based on achieved queue states.
        """
        if len(self.episode_buffer) == 0:
            return
        
        # Strategy: relabel with future states as goals
        if self.strategy == "future":
            for i, transition in enumerate(self.episode_buffer):
                # Use future states as goals
                future_indices = np.random.choice(
                    len(self.episode_buffer) - i,
                    size=min(4, len(self.episode_buffer) - i),
                    replace=False
                ) + i
                
                for future_idx in future_indices:
                    future_state = self.episode_buffer[future_idx].next_state
                    # Relabel reward based on goal
                    relabeled_reward = self._compute_goal_reward(
                        transition.state,
                        transition.action,
                        future_state
                    )
                    
                    # Store relabeled transition
                    self.replay_buffer.push(
                        transition.state,
                        transition.action,
                        relabeled_reward,
                        transition.next_state,
                        transition.done
                    )
    
    def _compute_goal_reward(self, state, action, goal_state) -> float:
        """
        Compute reward for goal-conditioned transition.
        
        Args:
            state: Current state
            action: Action taken
            goal_state: Goal state
            
        Returns:
            Reward value
        """
        # Simple reward: negative distance to goal
        distance = np.linalg.norm(state - goal_state)
        return -distance


class AdaptivePER:
    """
    Adaptive Prioritized Experience Replay.
    
    Adjusts priority computation based on learning progress.
    """
    
    def __init__(
        self,
        replay_buffer: PrioritizedReplayBuffer,
        adaptation_window: int = 1000,
    ):
        """
        Initialize adaptive PER.
        
        Args:
            replay_buffer: Base prioritized replay buffer
            adaptation_window: Window for adaptation
        """
        self.replay_buffer = replay_buffer
        self.adaptation_window = adaptation_window
        self.learning_progress = []
        self.alpha_history = []
    
    def update_alpha(self, learning_progress: float):
        """
        Adaptively update alpha based on learning progress.
        
        Args:
            learning_progress: Current learning progress metric
        """
        self.learning_progress.append(learning_progress)
        self.alpha_history.append(self.replay_buffer.alpha)
        
        # Keep recent history
        if len(self.learning_progress) > self.adaptation_window:
            self.learning_progress.pop(0)
            self.alpha_history.pop(0)
        
        # Adapt alpha based on progress
        if len(self.learning_progress) >= 100:
            recent_progress = np.mean(self.learning_progress[-100:])
            
            # If learning is slow, increase prioritization
            if recent_progress < 0.01:
                self.replay_buffer.alpha = min(1.0, self.replay_buffer.alpha + 0.01)
            # If learning is fast, decrease prioritization (more uniform)
            elif recent_progress > 0.1:
                self.replay_buffer.alpha = max(0.0, self.replay_buffer.alpha - 0.01)
    
    def get_alpha(self) -> float:
        """Get current alpha value."""
        return self.replay_buffer.alpha

