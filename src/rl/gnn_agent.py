"""
Graph Neural Network (GNN) Agent for Multi-Intersection Traffic Control.

Implements Phase 3.1 from OPTIMIZATION_ROADMAP.md:
- Graph structure: Intersections = nodes, Roads = edges
- GCN layers for spatial reasoning
- GAT layers for attention-based aggregation
- Temporal encoder for sequence modeling
- Multi-scale coordination

Expected Impact: 25-30% improvement for multi-intersection scenarios
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GNNConfig:
    """Configuration for GNN agent."""
    hidden_dim: int = 128
    num_gcn_layers: int = 2
    num_gat_layers: int = 1
    gat_heads: int = 4
    dropout: float = 0.1
    use_temporal: bool = True
    temporal_dim: int = 64
    fusion_type: str = "attention"  # "attention", "concat", "mean"
    learning_rate: float = 1e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 100000
    batch_size: int = 64
    target_update: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GraphConvolution(nn.Module):
    """
    Graph Convolution Layer.
    
    Implements message passing between connected intersections.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initialize graph convolution layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to use bias
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset parameters."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            Output features [num_nodes, out_features]
        """
        support = torch.mm(x, self.weight)  # [num_nodes, out_features]
        output = torch.mm(adj, support)  # [num_nodes, out_features]
        if self.bias is not None:
            output += self.bias
        return output


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer.
    
    Implements attention-based message aggregation.
    """
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize graph attention layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.FloatTensor(2 * self.head_dim, 1))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset parameters."""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention.
        
        Args:
            x: Node features [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            Output features [num_nodes, out_features]
        """
        num_nodes = x.size(0)
        h = self.W(x)  # [num_nodes, out_features]
        h = h.view(num_nodes, self.num_heads, self.head_dim)  # [num_nodes, num_heads, head_dim]
        
        # Compute attention scores
        a_input = torch.cat([
            h.repeat(1, num_nodes, 1).view(num_nodes * num_nodes, self.num_heads, self.head_dim),
            h.repeat(num_nodes, 1, 1)
        ], dim=-1)  # [num_nodes * num_nodes, num_heads, 2 * head_dim]
        
        e = self.leaky_relu(torch.matmul(a_input, self.a).squeeze(-1))  # [num_nodes * num_nodes, num_heads]
        e = e.view(num_nodes, num_nodes, self.num_heads)  # [num_nodes, num_nodes, num_heads]
        
        # Mask attention with adjacency
        attention = torch.where(adj.unsqueeze(-1) > 0, e, torch.tensor(-9e15, device=x.device))
        attention = F.softmax(attention, dim=1)  # [num_nodes, num_nodes, num_heads]
        attention = self.dropout(attention)
        
        # Aggregate with attention
        h_prime = torch.matmul(attention, h)  # [num_nodes, num_heads, head_dim]
        h_prime = h_prime.view(num_nodes, self.out_features)  # [num_nodes, out_features]
        
        return h_prime


class TemporalEncoder(nn.Module):
    """
    Temporal encoder for sequence modeling.
    
    Uses Transformer encoder for temporal patterns.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize temporal encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(TemporalEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Sequence features [batch_size, seq_len, input_dim]
            
        Returns:
            Encoded features [batch_size, seq_len, hidden_dim]
        """
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_dim]
        x = self.transformer(x)  # [batch_size, seq_len, hidden_dim]
        x = self.dropout(x)
        return x


class AttentionFusion(nn.Module):
    """
    Attention-based fusion of spatial and temporal features.
    """
    
    def __init__(self, spatial_dim: int, temporal_dim: int, output_dim: int):
        """
        Initialize attention fusion.
        
        Args:
            spatial_dim: Spatial feature dimension
            temporal_dim: Temporal feature dimension
            output_dim: Output dimension
        """
        super(AttentionFusion, self).__init__()
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        
        self.spatial_proj = nn.Linear(spatial_dim, output_dim)
        self.temporal_proj = nn.Linear(temporal_dim, output_dim)
        self.attention = nn.MultiheadAttention(output_dim, num_heads=4, batch_first=True)
        self.fusion = nn.Linear(output_dim * 2, output_dim)
    
    def forward(self, spatial: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            spatial: Spatial features [batch_size, num_nodes, spatial_dim]
            temporal: Temporal features [batch_size, seq_len, temporal_dim]
            
        Returns:
            Fused features [batch_size, num_nodes, output_dim]
        """
        # Project to same dimension
        spatial_proj = self.spatial_proj(spatial)  # [batch_size, num_nodes, output_dim]
        temporal_proj = self.temporal_proj(temporal)  # [batch_size, seq_len, output_dim]
        
        # Use temporal as query, spatial as key/value
        fused, _ = self.attention(temporal_proj, spatial_proj, spatial_proj)  # [batch_size, seq_len, output_dim]
        
        # Take last temporal step
        temporal_last = fused[:, -1, :].unsqueeze(1)  # [batch_size, 1, output_dim]
        
        # Concatenate and fuse
        combined = torch.cat([spatial_proj, temporal_last.expand_as(spatial_proj)], dim=-1)  # [batch_size, num_nodes, output_dim * 2]
        output = self.fusion(combined)  # [batch_size, num_nodes, output_dim]
        
        return output


class TrafficGNN(nn.Module):
    """
    Graph Neural Network for multi-intersection traffic control.
    
    Architecture:
    - GCN layers for spatial reasoning
    - GAT layers for attention-based aggregation
    - Temporal encoder for sequence modeling
    - Attention fusion for multi-scale coordination
    """
    
    def __init__(self, state_dim: int, action_dim: int, num_intersections: int, config: GNNConfig):
        """
        Initialize TrafficGNN.
        
        Args:
            state_dim: State dimension per intersection
            action_dim: Action dimension per intersection
            num_intersections: Number of intersections (nodes)
            config: GNN configuration
        """
        super(TrafficGNN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_intersections = num_intersections
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(state_dim, config.hidden_dim)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        in_dim = config.hidden_dim
        for i in range(config.num_gcn_layers):
            self.gcn_layers.append(GraphConvolution(in_dim, config.hidden_dim))
            in_dim = config.hidden_dim
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(config.num_gat_layers):
            self.gat_layers.append(GraphAttentionLayer(
                config.hidden_dim, config.hidden_dim, 
                num_heads=config.gat_heads, 
                dropout=config.dropout
            ))
        
        # Temporal encoder (optional)
        if config.use_temporal:
            self.temporal_encoder = TemporalEncoder(
                input_dim=config.hidden_dim,
                hidden_dim=config.temporal_dim,
                num_layers=2,
                num_heads=4,
                dropout=config.dropout
            )
            self.fusion = AttentionFusion(
                spatial_dim=config.hidden_dim,
                temporal_dim=config.temporal_dim,
                output_dim=config.hidden_dim
            )
        else:
            self.temporal_encoder = None
            self.fusion = None
        
        # Output layers
        self.dropout = nn.Dropout(config.dropout)
        self.output_layers = nn.ModuleList()
        hidden = config.hidden_dim
        for _ in range(2):
            self.output_layers.append(nn.Linear(hidden, hidden))
        self.q_head = nn.Linear(hidden, action_dim)
    
    def forward(self, states: torch.Tensor, adj: torch.Tensor, 
                temporal_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            states: Node states [batch_size, num_intersections, state_dim]
            adj: Adjacency matrix [num_intersections, num_intersections]
            temporal_seq: Temporal sequence [batch_size, seq_len, num_intersections, state_dim] (optional)
            
        Returns:
            Q-values [batch_size, num_intersections, action_dim]
        """
        batch_size = states.size(0)
        num_nodes = states.size(1)
        
        # Project input
        x = self.input_proj(states)  # [batch_size, num_nodes, hidden_dim]
        
        # Extract adjacency matrix (should be same for all batches)
        if adj.dim() == 3:
            adj_2d = adj[0]  # [num_nodes, num_nodes]
        else:
            adj_2d = adj  # [num_nodes, num_nodes]
        
        # GCN layers
        for gcn in self.gcn_layers:
            # Process each node independently
            x_list = []
            for b in range(batch_size):
                x_b = gcn(x[b], adj_2d)  # [num_nodes, hidden_dim]
                x_b = F.relu(x_b)
                x_list.append(x_b)
            x = torch.stack(x_list, dim=0)  # [batch_size, num_nodes, hidden_dim]
            x = self.dropout(x)
        
        # GAT layers
        for gat in self.gat_layers:
            x_list = []
            for b in range(batch_size):
                x_b = gat(x[b], adj_2d)  # [num_nodes, hidden_dim]
                x_b = F.relu(x_b)
                x_list.append(x_b)
            x = torch.stack(x_list, dim=0)  # [batch_size, num_nodes, hidden_dim]
            x = self.dropout(x)
        
        # Temporal encoding and fusion
        if self.temporal_encoder is not None and temporal_seq is not None:
            # Reshape temporal sequence: [batch_size, seq_len, num_nodes, state_dim] -> [batch_size * num_nodes, seq_len, state_dim]
            seq_len = temporal_seq.size(1)
            temporal_flat = temporal_seq.view(batch_size * num_nodes, seq_len, self.state_dim)
            
            # Project to hidden dim
            temporal_proj = self.input_proj(temporal_flat)  # [batch_size * num_nodes, seq_len, hidden_dim]
            
            # Encode temporally
            temporal_encoded = self.temporal_encoder(temporal_proj)  # [batch_size * num_nodes, seq_len, temporal_dim]
            
            # Reshape back and take mean over sequence
            temporal_encoded = temporal_encoded.view(batch_size, num_nodes, seq_len, self.config.temporal_dim)
            temporal_encoded = temporal_encoded.mean(dim=2)  # [batch_size, num_nodes, temporal_dim]
            
            # Fuse spatial and temporal
            # Reshape temporal to [batch_size, 1, temporal_dim] for attention
            temporal_for_attention = temporal_encoded.mean(dim=1, keepdim=True)  # [batch_size, 1, temporal_dim]
            x = self.fusion(x, temporal_for_attention)  # [batch_size, num_nodes, hidden_dim]
        
        # Output layers
        for layer in self.output_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        
        # Q-values
        q_values = self.q_head(x)  # [batch_size, num_intersections, action_dim]
        
        return q_values


class GNNAgent:
    """
    GNN-based RL agent for multi-intersection traffic control.
    """
    
    def __init__(self, state_dim: int, action_dim: int, num_intersections: int, 
                 adj_matrix: np.ndarray, config: Optional[GNNConfig] = None):
        """
        Initialize GNN agent.
        
        Args:
            state_dim: State dimension per intersection
            action_dim: Action dimension per intersection
            num_intersections: Number of intersections
            adj_matrix: Adjacency matrix [num_intersections, num_intersections]
            config: GNN configuration
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_intersections = num_intersections
        self.adj_matrix = torch.FloatTensor(adj_matrix).to(config.device if config else "cpu")
        self.config = config or GNNConfig()
        self.device = torch.device(self.config.device)
        
        # Networks
        self.policy_net = TrafficGNN(state_dim, action_dim, num_intersections, self.config).to(self.device)
        self.target_net = TrafficGNN(state_dim, action_dim, num_intersections, self.config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        
        # Replay buffer
        self.memory = []
        self.buffer_size = self.config.buffer_size
        
        # Training state
        self.epsilon = self.config.epsilon_start
        self.steps = 0
        
        logger.info(f"Initialized GNN agent for {num_intersections} intersections")
    
    def select_action(self, states: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select actions for all intersections.
        
        Args:
            states: States for all intersections [num_intersections, state_dim]
            training: Whether in training mode
            
        Returns:
            Actions for all intersections [num_intersections]
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim, size=self.num_intersections)
        
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).unsqueeze(0).to(self.device)  # [1, num_intersections, state_dim]
            adj = self.adj_matrix.unsqueeze(0)  # [1, num_intersections, num_intersections]
            q_values = self.policy_net(states_tensor, adj)  # [1, num_intersections, action_dim]
            actions = q_values.argmax(dim=-1).cpu().numpy()[0]  # [num_intersections]
        
        return actions
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, 
             next_state: np.ndarray, done: bool):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current states [num_intersections, state_dim]
            action: Actions taken [num_intersections]
            reward: Rewards received [num_intersections]
            next_state: Next states [num_intersections, state_dim]
            done: Whether episode is done
        """
        experience = (state, action, reward, next_state, done)
        if len(self.memory) >= self.buffer_size:
            self.memory.pop(0)
        self.memory.append(experience)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value or None
        """
        if len(self.memory) < self.config.batch_size:
            return None
        
        # Sample batch
        batch = np.random.choice(len(self.memory), self.config.batch_size, replace=False)
        states = torch.FloatTensor([self.memory[i][0] for i in batch]).to(self.device)
        actions = torch.LongTensor([self.memory[i][1] for i in batch]).to(self.device)
        rewards = torch.FloatTensor([self.memory[i][2] for i in batch]).to(self.device)
        next_states = torch.FloatTensor([self.memory[i][3] for i in batch]).to(self.device)
        dones = torch.BoolTensor([self.memory[i][4] for i in batch]).to(self.device)
        
        # Expand adjacency matrix
        adj = self.adj_matrix.unsqueeze(0).expand(self.config.batch_size, -1, -1).to(self.device)
        
        # Current Q-values
        q_values = self.policy_net(states, adj)  # [batch_size, num_intersections, action_dim]
        actions_expanded = actions.unsqueeze(-1).unsqueeze(-1)  # [batch_size, num_intersections, 1, 1]
        q_values = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # [batch_size, num_intersections]
        
        # Next Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states, adj)  # [batch_size, num_intersections, action_dim]
            next_q_values = next_q_values.max(dim=-1)[0]  # [batch_size, num_intersections]
            # Expand dones to match shape
            dones_expanded = dones.unsqueeze(-1).float()  # [batch_size, 1]
            target_q = rewards + (self.config.gamma * next_q_values * (1 - dones_expanded))
        
        # Compute loss
        loss = F.mse_loss(q_values, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.config.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.config.epsilon_end, 
                         self.epsilon * self.config.epsilon_decay)
        
        return loss.item()


def build_intersection_graph(num_intersections: int, topology: str = "grid") -> np.ndarray:
    """
    Build adjacency matrix for intersection graph.
    
    Args:
        num_intersections: Number of intersections
        topology: Graph topology ("grid", "line", "ring", "fully_connected")
        
    Returns:
        Adjacency matrix [num_intersections, num_intersections]
    """
    adj = np.zeros((num_intersections, num_intersections))
    
    if topology == "grid":
        # Grid topology (e.g., 2x2, 3x3)
        side = int(np.sqrt(num_intersections))
        for i in range(num_intersections):
            row, col = i // side, i % side
            # Connect to neighbors
            if row > 0:
                adj[i, i - side] = 1  # Up
            if row < side - 1:
                adj[i, i + side] = 1  # Down
            if col > 0:
                adj[i, i - 1] = 1  # Left
            if col < side - 1:
                adj[i, i + 1] = 1  # Right
    
    elif topology == "line":
        # Linear topology
        for i in range(num_intersections - 1):
            adj[i, i + 1] = 1
            adj[i + 1, i] = 1
    
    elif topology == "ring":
        # Ring topology
        for i in range(num_intersections):
            adj[i, (i + 1) % num_intersections] = 1
            adj[(i + 1) % num_intersections, i] = 1
    
    elif topology == "fully_connected":
        # Fully connected
        adj = np.ones((num_intersections, num_intersections))
        np.fill_diagonal(adj, 0)
    
    # Add self-loops
    np.fill_diagonal(adj, 1)
    
    return adj

