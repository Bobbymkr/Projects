"""
Complete Model-Based Reinforcement Learning Implementation for Traffic Control.

This is the FULL implementation completing the 60% partial implementation.
Uses actual neural networks for world model and MPC planning.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class WorldModelState:
    """State representation for world model."""
    state: np.ndarray
    action: int
    next_state: np.ndarray
    reward: float
    done: bool


class TransitionModel(nn.Module):
    """Neural network for predicting next state given current state and action."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        super().__init__()
        layers = []
        input_dim = state_dim + action_dim  # Concatenate state and action
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, state_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next state."""
        # One-hot encode action
        action_onehot = torch.zeros(state.shape[0], self.network[0].in_features - state.shape[1])
        action_onehot.scatter_(1, action.unsqueeze(1), 1)
        
        # Concatenate state and action
        input_tensor = torch.cat([state, action_onehot], dim=1)
        return self.network(input_tensor)


class RewardModel(nn.Module):
    """Neural network for predicting reward given state, action, and next state."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        layers = []
        # Input: state + action + next_state
        input_dim = state_dim * 2 + action_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """Predict reward."""
        # One-hot encode action
        action_onehot = torch.zeros(state.shape[0], self.network[0].in_features - state.shape[1] * 2)
        action_onehot.scatter_(1, action.unsqueeze(1), 1)
        
        # Concatenate state, action, next_state
        input_tensor = torch.cat([state, action_onehot, next_state], dim=1)
        return self.network(input_tensor)


class WorldModel:
    """
    Complete World Model for Model-Based RL.
    
    Learns a predictive model of the environment dynamics,
    enabling planning and model-predictive control.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128],
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ):
        """
        Initialize world model.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate for training
            device: Device for neural networks
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.hidden_dims = hidden_dims
        self.device = device
        
        # Initialize model components
        self.transition_model = TransitionModel(state_dim, action_dim, hidden_dims).to(device)
        self.reward_model = RewardModel(state_dim, action_dim, hidden_dims).to(device)
        
        # Optimizers
        self.transition_optimizer = optim.Adam(self.transition_model.parameters(), lr=learning_rate)
        self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=learning_rate)
        
        self.is_trained = False
        logger.info(f"Initialized World Model (state_dim={state_dim}, action_dim={action_dim})")
    
    def train(
        self,
        transitions: List[WorldModelState],
        epochs: int = 100,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Train world model on transition data.
        
        Args:
            transitions: List of state transitions
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training metrics
        """
        logger.info(f"Training World Model on {len(transitions)} transitions")
        
        if len(transitions) < batch_size:
            logger.warning("Not enough transitions for training")
            return {}
        
        # Prepare training data
        states = torch.FloatTensor(np.array([t.state for t in transitions])).to(self.device)
        actions = torch.LongTensor([t.action for t in transitions]).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in transitions])).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in transitions]).to(self.device)
        
        # Training loop
        transition_losses = []
        reward_losses = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            epoch_transition_loss = []
            epoch_reward_loss = []
            
            for i in range(0, len(states), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_next_states = next_states[batch_indices]
                batch_rewards = rewards[batch_indices]
                
                # Train transition model
                pred_next_states = self.transition_model(batch_states, batch_actions)
                transition_loss = nn.functional.mse_loss(pred_next_states, batch_next_states)
                
                self.transition_optimizer.zero_grad()
                transition_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.transition_model.parameters(), 1.0)
                self.transition_optimizer.step()
                
                epoch_transition_loss.append(transition_loss.item())
                
                # Train reward model
                pred_rewards = self.reward_model(batch_states, batch_actions, batch_next_states)
                reward_loss = nn.functional.mse_loss(pred_rewards.squeeze(), batch_rewards)
                
                self.reward_optimizer.zero_grad()
                reward_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), 1.0)
                self.reward_optimizer.step()
                
                epoch_reward_loss.append(reward_loss.item())
            
            avg_transition_loss = np.mean(epoch_transition_loss)
            avg_reward_loss = np.mean(epoch_reward_loss)
            transition_losses.append(avg_transition_loss)
            reward_losses.append(avg_reward_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Transition Loss: {avg_transition_loss:.4f} | "
                    f"Reward Loss: {avg_reward_loss:.4f}"
                )
        
        self.is_trained = True
        
        return {
            "final_transition_loss": transition_losses[-1] if transition_losses else 0.0,
            "final_reward_loss": reward_losses[-1] if reward_losses else 0.0,
            "transition_losses": transition_losses,
            "reward_losses": reward_losses,
            "epochs": epochs,
        }
    
    def predict_next_state(
        self,
        state: np.ndarray,
        action: int,
    ) -> np.ndarray:
        """
        Predict next state given current state and action.
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            Predicted next state
        """
        if not self.is_trained:
            logger.warning("World model not trained. Returning state unchanged.")
            return state
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_tensor = torch.LongTensor([action]).to(self.device)
            pred_next_state = self.transition_model(state_tensor, action_tensor)
            return pred_next_state.cpu().numpy().squeeze()
    
    def predict_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
    ) -> float:
        """
        Predict reward for transition.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            Predicted reward
        """
        if not self.is_trained:
            return 0.0
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_tensor = torch.LongTensor([action]).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            pred_reward = self.reward_model(state_tensor, action_tensor, next_state_tensor)
            return pred_reward.cpu().item()


class ModelPredictiveControl:
    """
    Model-Predictive Control (MPC) using World Model.
    
    Plans actions by optimizing over predicted future trajectories
    using the learned world model.
    """
    
    def __init__(
        self,
        world_model: WorldModel,
        horizon: int = 10,
        num_candidates: int = 100,
    ):
        """
        Initialize MPC controller.
        
        Args:
            world_model: Trained world model
            horizon: Planning horizon (number of steps ahead)
            num_candidates: Number of candidate action sequences to evaluate
        """
        self.world_model = world_model
        self.horizon = horizon
        self.num_candidates = num_candidates
    
    def select_action(
        self,
        state: np.ndarray,
        action_dim: int,
    ) -> int:
        """
        Select action using MPC planning.
        
        Args:
            state: Current state
            action_dim: Dimension of action space
            
        Returns:
            Selected action
        """
        if not self.world_model.is_trained:
            # Only log warning once to reduce noise
            if not hasattr(self.world_model, '_mpc_warning_logged'):
                logger.warning("World model not trained yet. Using random actions until enough data is collected.")
                self.world_model._mpc_warning_logged = True
            return np.random.randint(0, action_dim)
        
        # Generate candidate action sequences
        candidates = []
        for _ in range(self.num_candidates):
            # Generate random action sequence
            action_sequence = [
                np.random.randint(0, action_dim)
                for _ in range(self.horizon)
            ]
            
            # Simulate trajectory and compute value
            total_value = self._simulate_trajectory(state, action_sequence)
            
            candidates.append({
                "action_sequence": action_sequence,
                "value": total_value,
            })
        
        # Select best candidate
        best_candidate = max(candidates, key=lambda x: x["value"])
        
        # Return first action from best sequence
        return best_candidate["action_sequence"][0]
    
    def _simulate_trajectory(
        self,
        initial_state: np.ndarray,
        action_sequence: List[int],
    ) -> float:
        """
        Simulate trajectory using world model.
        
        Args:
            initial_state: Starting state
            action_sequence: Sequence of actions to take
            
        Returns:
            Total predicted value
        """
        current_state = initial_state.copy()
        total_reward = 0.0
        discount = 0.99
        
        for step, action in enumerate(action_sequence):
            # Predict next state
            next_state = self.world_model.predict_next_state(
                current_state, action
            )
            
            # Predict reward
            reward = self.world_model.predict_reward(
                current_state, action, next_state
            )
            
            total_reward += (discount ** step) * reward
            
            current_state = next_state
        
        return total_reward


class ModelBasedRLAgent:
    """
    Complete Model-Based RL Agent.
    
    Combines world model learning and MPC planning for efficient
    traffic control optimization.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        world_model_config: Optional[Dict[str, Any]] = None,
        mpc_config: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ):
        """
        Initialize model-based RL agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            world_model_config: Configuration for world model
            mpc_config: Configuration for MPC
            device: Device for neural networks
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Initialize world model
        world_model_params = world_model_config or {}
        self.world_model = WorldModel(state_dim, action_dim, device=device, **world_model_params)
        
        # Initialize MPC controller
        mpc_params = mpc_config or {}
        self.mpc = ModelPredictiveControl(self.world_model, **mpc_params)
        
        # Transition buffer
        self.transition_buffer: List[WorldModelState] = []
        self.min_transitions_for_training = 30  # Reduced for faster training
        self.max_transitions = 5000  # Maximum buffer size to prevent memory issues
        self.last_training_step = 0
        self.training_interval = 20  # Train world model every N transitions
        self._warning_logged = False  # Track if warning was logged
        
        # Convergence tracking
        self.training_losses: deque = deque(maxlen=10)  # Track last 10 training losses
        self.convergence_threshold = 0.001  # Loss change threshold for convergence
        self.convergence_patience = 3  # Number of consecutive stable trainings
        self.convergence_count = 0  # Counter for stable trainings
        self.is_converged = False  # Flag to stop training
        
        logger.info(f"Initialized Model-Based RL Agent (state_dim={state_dim}, action_dim={action_dim})")
    
    def add_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add transition to buffer."""
        transition = WorldModelState(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            done=done,
        )
        self.transition_buffer.append(transition)
        
        # Limit buffer size to prevent memory issues
        if len(self.transition_buffer) > self.max_transitions:
            # Keep most recent transitions
            self.transition_buffer = self.transition_buffer[-self.max_transitions:]
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Store experience for training (compatibility with training script).
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.add_transition(state, action, reward, next_state, done)
    
    def train_step(self, batch: List = None):
        """
        Train step for model-based RL.
        
        Args:
            batch: Optional batch (if None, uses transition buffer)
        """
        # Skip training if converged
        if self.is_converged:
            return
        
        # Train world model periodically
        if len(self.transition_buffer) >= self.min_transitions_for_training:
            # Train world model every N transitions
            transitions_since_last = len(self.transition_buffer) - self.last_training_step
            if transitions_since_last >= self.training_interval or not self.world_model.is_trained:
                try:
                    # Train with more epochs if we have more data
                    epochs = min(20, max(10, len(self.transition_buffer) // 30))
                    was_trained = self.world_model.is_trained
                    
                    # Train and get loss
                    training_results = self.train_world_model(epochs=epochs, batch_size=32)
                    self.last_training_step = len(self.transition_buffer)
                    
                    # Check for convergence
                    if training_results and 'final_transition_loss' in training_results:
                        final_loss = training_results['final_transition_loss']
                        self._check_convergence(final_loss)
                    
                    if not was_trained and self.world_model.is_trained:
                        logger.info(f"World model trained on {len(self.transition_buffer)} transitions - now using MPC")
                    
                    if self.is_converged:
                        logger.info(f"World model converged after {len(self.transition_buffer)} transitions. Stopping training.")
                        
                except Exception as e:
                    logger.warning(f"Error training world model: {e}")
    
    def _check_convergence(self, current_loss: float):
        """
        Check if world model has converged.
        
        Args:
            current_loss: Current training loss
        """
        if len(self.training_losses) == 0:
            self.training_losses.append(current_loss)
            return
        
        # Calculate loss change
        previous_loss = self.training_losses[-1]
        loss_change = abs(current_loss - previous_loss)
        
        # Check if loss change is below threshold
        if loss_change < self.convergence_threshold:
            self.convergence_count += 1
        else:
            self.convergence_count = 0  # Reset if loss changed significantly
        
        self.training_losses.append(current_loss)
        
        # Mark as converged if stable for multiple trainings
        if self.convergence_count >= self.convergence_patience:
            self.is_converged = True
    
    @property
    def replay_buffer(self):
        """Property to access replay buffer for compatibility."""
        # Return transition buffer as list for compatibility
        return self.transition_buffer
    
    def train_world_model(
        self,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Train world model on collected transitions."""
        if len(self.transition_buffer) < self.min_transitions_for_training:
            logger.warning(f"Not enough transitions for training ({len(self.transition_buffer)} < {self.min_transitions_for_training})")
            return {}
        
        return self.world_model.train(
            self.transition_buffer,
            epochs=epochs,
            batch_size=batch_size,
        )
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using MPC.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        # Ensure world model is trained before using MPC
        if not self.world_model.is_trained and not self.is_converged:
            # Try to train world model if we have enough data
            if len(self.transition_buffer) >= self.min_transitions_for_training:
                try:
                    was_trained = self.world_model.is_trained
                    training_results = self.train_world_model(epochs=15, batch_size=32)
                    
                    # Check convergence
                    if training_results and 'final_transition_loss' in training_results:
                        self._check_convergence(training_results['final_transition_loss'])
                    
                    if not was_trained and self.world_model.is_trained:
                        logger.info(f"World model trained on {len(self.transition_buffer)} transitions - now using MPC")
                    
                    if self.is_converged:
                        logger.info(f"World model converged. Using trained model for predictions.")
                        
                except Exception as e:
                    logger.debug(f"Could not train world model: {e}")
        
        return self.mpc.select_action(state, self.action_dim)
    
    def reset_buffer(self) -> None:
        """Clear transition buffer."""
        self.transition_buffer = []
    
    def save(self, path: str):
        """Save agent to file."""
        torch.save({
            'transition_model': self.world_model.transition_model.state_dict(),
            'reward_model': self.world_model.reward_model.state_dict(),
            'is_trained': self.world_model.is_trained,
        }, path)
        logger.info(f"Saved Model-Based RL Agent to {path}")
    
    def load(self, path: str):
        """Load agent from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.world_model.transition_model.load_state_dict(checkpoint['transition_model'])
        self.world_model.reward_model.load_state_dict(checkpoint['reward_model'])
        self.world_model.is_trained = checkpoint.get('is_trained', False)
        logger.info(f"Loaded Model-Based RL Agent from {path}")

