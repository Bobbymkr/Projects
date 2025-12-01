"""
Transformer-Based Traffic Signal Control.

Uses Transformer architecture to model temporal dependencies
in traffic patterns for optimal signal timing.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple
import math

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:x.size(0), :]


class TransformerTrafficController(nn.Module):
    """
    Transformer-based Traffic Signal Controller.
    
    Uses self-attention to model temporal dependencies in traffic patterns.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 100,
    ):
        """
        Initialize transformer controller.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Input projection
        self.input_projection = nn.Linear(state_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, action_dim),
        )
        
        logger.info(f"Initialized Transformer Controller (d_model={d_model}, layers={num_layers})")
    
    def forward(self, state_sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer.
        
        Args:
            state_sequence: Sequence of states [seq_len, batch, state_dim]
            
        Returns:
            Action logits [batch, action_dim]
        """
        # Project input
        x = self.input_projection(state_sequence)  # [seq_len, batch, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)  # [seq_len, batch, d_model]
        
        # Use last timestep for prediction
        last_hidden = encoded[-1]  # [batch, d_model]
        
        # Project to action space
        action_logits = self.output_projection(last_hidden)  # [batch, action_dim]
        
        return action_logits


class TransformerAgent:
    """
    Transformer-based Agent for Traffic Control.
    
    Uses transformer architecture to learn optimal signal timing
    by modeling temporal traffic patterns.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        learning_rate: float = 1e-4,
        device: str = "cpu",
    ):
        """
        Initialize transformer agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            learning_rate: Learning rate
            device: Device for computation
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Create transformer model
        self.model = TransformerTrafficController(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
        ).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # State history for sequence modeling
        self.state_history: List[np.ndarray] = []
        self.max_history_len = 50
        
        self.is_trained = False
        logger.info("Initialized Transformer Agent")
    
    def add_to_history(self, state: np.ndarray):
        """Add state to history."""
        self.state_history.append(state.copy())
        if len(self.state_history) > self.max_history_len:
            self.state_history.pop(0)
    
    def get_state_sequence(self) -> torch.Tensor:
        """Get state sequence tensor."""
        if len(self.state_history) == 0:
            # Return zero sequence
            return torch.zeros(1, 1, self.state_dim).to(self.device)
        
        # Pad or truncate to fixed length
        sequence = np.array(self.state_history[-self.model.max_seq_len:])
        
        # Pad if needed
        if len(sequence) < self.model.max_seq_len:
            padding = np.zeros((self.model.max_seq_len - len(sequence), self.state_dim))
            sequence = np.vstack([padding, sequence])
        
        # Convert to tensor [seq_len, 1, state_dim]
        return torch.FloatTensor(sequence).transpose(0, 1).to(self.device)
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select action using transformer.
        
        Args:
            state: Current state
            epsilon: Exploration rate
            
        Returns:
            Selected action
        """
        if not self.is_trained:
            return np.random.randint(0, self.action_dim)
        
        # Add to history
        self.add_to_history(state)
        
        # Get sequence
        state_seq = self.get_state_sequence()
        
        # Predict
        with torch.no_grad():
            logits = self.model(state_seq)
            probs = torch.softmax(logits, dim=1)
            
            if np.random.random() < epsilon:
                action = np.random.randint(0, self.action_dim)
            else:
                action = torch.argmax(probs, dim=1).item()
        
        return int(action)
    
    def train_step(
        self,
        state_sequences: List[np.ndarray],
        actions: List[int],
    ) -> float:
        """
        Train on batch of sequences.
        
        Args:
            state_sequences: List of state sequences
            actions: List of actions
            
        Returns:
            Loss value
        """
        if len(state_sequences) == 0:
            return 0.0
        
        # Prepare batch
        batch_size = len(state_sequences)
        seq_len = self.model.max_seq_len
        
        # Create batch tensor [seq_len, batch, state_dim]
        batch_tensor = torch.zeros(seq_len, batch_size, self.state_dim).to(self.device)
        
        for i, seq in enumerate(state_sequences):
            # Pad or truncate
            seq_array = np.array(seq[-seq_len:])
            if len(seq_array) < seq_len:
                padding = np.zeros((seq_len - len(seq_array), self.state_dim))
                seq_array = np.vstack([padding, seq_array])
            
            batch_tensor[:, i, :] = torch.FloatTensor(seq_array).to(self.device)
        
        actions_tensor = torch.LongTensor(actions).to(self.device)
        
        # Forward pass
        logits = self.model(batch_tensor)
        loss = self.criterion(logits, actions_tensor)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def save(self, path: str):
        """Save agent."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'is_trained': self.is_trained,
        }, path)
        logger.info(f"Saved Transformer Agent to {path}")
    
    def load(self, path: str):
        """Load agent."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.is_trained = checkpoint.get('is_trained', False)
        logger.info(f"Loaded Transformer Agent from {path}")

