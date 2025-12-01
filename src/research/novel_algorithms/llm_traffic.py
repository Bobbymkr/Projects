"""
LLM for Traffic Signal Control.

Uses Large Language Models for natural language reasoning
about traffic patterns and signal timing decisions.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple
import json

logger = logging.getLogger(__name__)


class TrafficLLMEncoder(nn.Module):
    """
    Encoder for traffic state to LLM input.
    
    Converts numerical traffic state to text representation.
    """
    
    def __init__(self, state_dim: int, embedding_dim: int = 128):
        """
        Initialize encoder.
        
        Args:
            state_dim: Dimension of state space
            embedding_dim: Embedding dimension
        """
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        
        # Embedding layer
        self.embedding = nn.Linear(state_dim, embedding_dim)
    
    def state_to_text(self, state: np.ndarray) -> str:
        """
        Convert state to text description.
        
        Args:
            state: Traffic state
            
        Returns:
            Text description
        """
        # Simplified: convert to natural language
        queue_north = state[0] if len(state) > 0 else 0
        queue_south = state[1] if len(state) > 1 else 0
        queue_east = state[2] if len(state) > 2 else 0
        queue_west = state[3] if len(state) > 3 else 0
        
        text = (
            f"Traffic state: North queue={queue_north:.1f}, "
            f"South queue={queue_south:.1f}, "
            f"East queue={queue_east:.1f}, "
            f"West queue={queue_west:.1f}. "
            f"Total vehicles waiting: {np.sum(state):.1f}."
        )
        
        return text
    
    def forward(self, state: np.ndarray) -> torch.Tensor:
        """Encode state to embedding."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        return self.embedding(state_tensor)


class SimpleLLM(nn.Module):
    """
    Simplified LLM for traffic control.
    
    Uses transformer architecture to process text descriptions
    and generate action decisions.
    """
    
    def __init__(
        self,
        vocab_size: int = 1000,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        action_dim: int = 12,
    ):
        """
        Initialize LLM.
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            action_dim: Action dimension
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, action_dim),
        )
    
    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            text_embeddings: Text embeddings [batch, seq_len, embedding_dim]
            
        Returns:
            Action logits [batch, action_dim]
        """
        # Transformer encoding
        encoded = self.transformer(text_embeddings)
        
        # Use mean pooling
        pooled = encoded.mean(dim=1)
        
        # Predict action
        action_logits = self.action_head(pooled)
        
        return action_logits


class LLMTrafficAgent:
    """
    LLM-based Agent for Traffic Control.
    
    Uses language model reasoning for traffic signal decisions.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
    ):
        """
        Initialize LLM agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            device: Device for computation
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Encoder
        self.encoder = TrafficLLMEncoder(state_dim).to(device)
        
        # LLM
        self.llm = SimpleLLM(action_dim=action_dim).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.llm.parameters()),
            lr=1e-4,
        )
        
        # Knowledge base (rules and patterns)
        self.knowledge_base = self._initialize_knowledge_base()
        
        self.is_trained = False
        logger.info("Initialized LLM Traffic Agent")
    
    def _initialize_knowledge_base(self) -> Dict[str, str]:
        """Initialize knowledge base with traffic rules."""
        return {
            "rule_1": "If queue length is high (>15), extend green time",
            "rule_2": "If queue length is low (<3), reduce green time",
            "rule_3": "Balance queues across all directions",
            "rule_4": "Prioritize direction with highest queue",
            "rule_5": "Minimize total wait time",
        }
    
    def reason_about_traffic(self, state: np.ndarray) -> str:
        """
        Generate reasoning about traffic state.
        
        Args:
            state: Current state
            
        Returns:
            Reasoning text
        """
        # Convert state to text
        state_text = self.encoder.state_to_text(state)
        
        # Apply knowledge base rules
        reasoning = f"{state_text} "
        
        max_queue = np.max(state)
        if max_queue > 15:
            reasoning += "High queue detected. " + self.knowledge_base["rule_1"] + ". "
        elif max_queue < 3:
            reasoning += "Low queue detected. " + self.knowledge_base["rule_2"] + ". "
        
        reasoning += self.knowledge_base["rule_3"] + ". "
        reasoning += "Recommend action based on current traffic patterns."
        
        return reasoning
    
    def select_action(self, state: np.ndarray) -> Tuple[int, str]:
        """
        Select action using LLM reasoning.
        
        Args:
            state: Current state
            
        Returns:
            Action and reasoning
        """
        if not self.is_trained:
            return np.random.randint(0, self.action_dim), "Model not trained"
        
        # Generate reasoning
        reasoning = self.reason_about_traffic(state)
        
        # Encode state
        state_embedding = self.encoder(state)
        
        # Convert reasoning to tokens (simplified: use state embedding)
        # In production, use actual tokenizer
        text_embedding = state_embedding.unsqueeze(0)  # [1, 1, embedding_dim]
        
        # LLM forward pass
        with torch.no_grad():
            action_logits = self.llm(text_embedding)
            action = torch.argmax(action_logits, dim=1).item()
        
        return int(action), reasoning
    
    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> Dict[str, float]:
        """
        Train LLM agent.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            
        Returns:
            Training metrics
        """
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        
        # Encode states
        state_embeddings = self.encoder(states_tensor)
        
        # Forward pass
        action_logits = self.llm(state_embeddings)
        
        # Loss
        loss = nn.functional.cross_entropy(action_logits, actions_tensor)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.llm.parameters()),
            1.0,
        )
        self.optimizer.step()
        
        self.is_trained = True
        
        return {"loss": loss.item()}
    
    def update_knowledge_base(self, new_rule: str):
        """Add new rule to knowledge base."""
        rule_id = f"rule_{len(self.knowledge_base) + 1}"
        self.knowledge_base[rule_id] = new_rule
        logger.info(f"Added rule to knowledge base: {new_rule}")
    
    def save(self, path: str):
        """Save agent."""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'llm_state_dict': self.llm.state_dict(),
            'is_trained': self.is_trained,
            'knowledge_base': self.knowledge_base,
        }, path)
        logger.info(f"Saved LLM Agent to {path}")
    
    def load(self, path: str):
        """Load agent."""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.llm.load_state_dict(checkpoint['llm_state_dict'])
        self.is_trained = checkpoint.get('is_trained', False)
        self.knowledge_base = checkpoint.get('knowledge_base', self._initialize_knowledge_base())
        logger.info(f"Loaded LLM Agent from {path}")

