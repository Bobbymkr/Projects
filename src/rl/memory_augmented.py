"""
Memory-Augmented Networks for Traffic Control.

Implements Phase 3.3 from OPTIMIZATION_ROADMAP.md:
- Neural Turing Machine: Store long-term patterns
- Differentiable Neural Computer: Complex memory operations
- Episodic Memory: Remember rare events (accidents, emergencies)

Expected Impact: 20-25% improvement on rare events
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory-augmented networks."""
    memory_type: str = "ntm"  # "ntm", "dnc", "episodic"
    memory_size: int = 128
    memory_dim: int = 64
    num_read_heads: int = 4
    num_write_heads: int = 1
    controller_dim: int = 128
    # NTM specific
    shift_range: int = 1
    # DNC specific
    link_matrix_size: int = 16
    # Episodic specific
    episode_capacity: int = 1000
    similarity_threshold: float = 0.7


class NeuralTuringMachine(nn.Module):
    """
    Neural Turing Machine (NTM).
    
    Stores and retrieves long-term patterns in external memory.
    """
    
    def __init__(self, input_dim: int, output_dim: int, memory_size: int = 128,
                 memory_dim: int = 64, num_read_heads: int = 4, num_write_heads: int = 1,
                 controller_dim: int = 128, shift_range: int = 1):
        """
        Initialize NTM.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            memory_size: Memory size (number of slots)
            memory_dim: Memory dimension (features per slot)
            num_read_heads: Number of read heads
            num_write_heads: Number of write heads
            controller_dim: Controller dimension
            shift_range: Shift range for location-based addressing
        """
        super(NeuralTuringMachine, self).__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.controller_dim = controller_dim
        self.shift_range = shift_range
        
        # Memory matrix
        self.register_buffer('memory', torch.zeros(memory_size, memory_dim))
        
        # Controller (LSTM)
        self.controller = nn.LSTM(input_dim, controller_dim, batch_first=True)
        
        # Read heads
        self.read_heads = nn.ModuleList([
            nn.Linear(controller_dim, memory_size) for _ in range(num_read_heads)
        ])
        
        # Write heads
        self.write_heads = nn.ModuleList([
            nn.ModuleDict({
                'erase': nn.Linear(controller_dim, memory_dim),
                'add': nn.Linear(controller_dim, memory_dim),
                'location': nn.Linear(controller_dim, memory_size)
            }) for _ in range(num_write_heads)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(controller_dim + num_read_heads * memory_dim, output_dim)
    
    def content_addressing(self, key: torch.Tensor, memory: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Content-based addressing.
        
        Args:
            key: Key vector [batch, memory_dim]
            memory: Memory matrix [memory_size, memory_dim]
            beta: Strength parameter [batch, 1]
            
        Returns:
            Attention weights [batch, memory_size]
        """
        # Cosine similarity
        key_norm = F.normalize(key, p=2, dim=-1)
        memory_norm = F.normalize(memory, p=2, dim=-1)
        similarity = torch.matmul(key_norm, memory_norm.t())  # [batch, memory_size]
        
        # Apply beta
        weights = F.softmax(beta * similarity, dim=-1)
        return weights
    
    def location_addressing(self, weights: torch.Tensor, g: torch.Tensor,
                           shift_weights: torch.Tensor) -> torch.Tensor:
        """
        Location-based addressing with shifting.
        
        Args:
            weights: Current weights [batch, memory_size]
            g: Interpolation gate [batch, 1]
            shift_weights: Shift weights [batch, 2 * shift_range + 1]
            
        Returns:
            New weights [batch, memory_size]
        """
        # Interpolation
        weights = g * weights + (1 - g) * weights
        
        # Convolutional shift
        batch_size = weights.size(0)
        shifted = torch.zeros_like(weights)
        
        for i in range(batch_size):
            w = weights[i]
            for j in range(-self.shift_range, self.shift_range + 1):
                idx = j + self.shift_range
                if shift_weights[i, idx] > 0:
                    shifted_w = torch.roll(w, j, dims=0)
                    shifted[i] += shift_weights[i, idx] * shifted_w
        
        # Sharpening
        gamma = torch.ones(batch_size, 1, device=weights.device)
        shifted = shifted ** gamma
        shifted = shifted / (shifted.sum(dim=-1, keepdim=True) + 1e-10)
        
        return shifted
    
    def read(self, memory: torch.Tensor, read_weights: torch.Tensor) -> torch.Tensor:
        """
        Read from memory.
        
        Args:
            memory: Memory matrix [memory_size, memory_dim]
            read_weights: Read weights [batch, num_read_heads, memory_size]
            
        Returns:
            Read vectors [batch, num_read_heads, memory_dim]
        """
        batch_size = read_weights.size(0)
        read_vectors = torch.zeros(batch_size, self.num_read_heads, self.memory_dim, device=memory.device)
        
        for h in range(self.num_read_heads):
            weights = read_weights[:, h, :]  # [batch, memory_size]
            read_vectors[:, h, :] = torch.matmul(weights, memory)  # [batch, memory_dim]
        
        return read_vectors
    
    def write(self, memory: torch.Tensor, write_weights: torch.Tensor,
              erase_vectors: torch.Tensor, add_vectors: torch.Tensor) -> torch.Tensor:
        """
        Write to memory.
        
        Args:
            memory: Memory matrix [memory_size, memory_dim]
            write_weights: Write weights [batch, num_write_heads, memory_size]
            erase_vectors: Erase vectors [batch, num_write_heads, memory_dim]
            add_vectors: Add vectors [batch, num_write_heads, memory_dim]
            
        Returns:
            Updated memory [memory_size, memory_dim]
        """
        new_memory = memory.clone()
        
        for h in range(self.num_write_heads):
            weights = write_weights[:, h, :].unsqueeze(-1)  # [batch, memory_size, 1]
            erase = erase_vectors[:, h, :].unsqueeze(1)  # [batch, 1, memory_dim]
            add = add_vectors[:, h, :].unsqueeze(1)  # [batch, 1, memory_dim]
            
            # Erase
            erase_weights = weights * erase  # [batch, memory_size, memory_dim]
            erase_weights = erase_weights.mean(dim=0)  # [memory_size, memory_dim]
            new_memory = new_memory * (1 - erase_weights)
            
            # Add
            add_weights = weights * add  # [batch, memory_size, memory_dim]
            add_weights = add_weights.mean(dim=0)  # [memory_size, memory_dim]
            new_memory = new_memory + add_weights
        
        return new_memory
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            
        Returns:
            Output tensor [batch, seq_len, output_dim]
            Memory state [memory_size, memory_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # Controller forward
        controller_out, (h_n, c_n) = self.controller(x)  # [batch, seq_len, controller_dim]
        
        outputs = []
        memory = self.memory.clone()
        
        for t in range(seq_len):
            controller_state = controller_out[:, t, :]  # [batch, controller_dim]
            
            # Read from memory
            read_weights_list = []
            for read_head in self.read_heads:
                # Simplified: use controller state to generate read weights
                weights = F.softmax(read_head(controller_state), dim=-1)  # [batch, memory_size]
                read_weights_list.append(weights)
            
            read_weights = torch.stack(read_weights_list, dim=1)  # [batch, num_read_heads, memory_size]
            read_vectors = self.read(memory, read_weights)  # [batch, num_read_heads, memory_dim]
            
            # Write to memory
            write_weights_list = []
            erase_vectors = []
            add_vectors = []
            
            for write_head in self.write_heads:
                weights = F.softmax(write_head['location'](controller_state), dim=-1)  # [batch, memory_size]
                write_weights_list.append(weights)
                erase_vectors.append(torch.sigmoid(write_head['erase'](controller_state)))  # [batch, memory_dim]
                add_vectors.append(torch.tanh(write_head['add'](controller_state)))  # [batch, memory_dim]
            
            write_weights = torch.stack(write_weights_list, dim=1)  # [batch, num_write_heads, memory_size]
            erase_vectors = torch.stack(erase_vectors, dim=1)  # [batch, num_write_heads, memory_dim]
            add_vectors = torch.stack(add_vectors, dim=1)  # [batch, num_write_heads, memory_dim]
            
            memory = self.write(memory, write_weights, erase_vectors, add_vectors)
            
            # Combine controller output with read vectors
            read_flat = read_vectors.view(batch_size, -1)  # [batch, num_read_heads * memory_dim]
            combined = torch.cat([controller_state, read_flat], dim=-1)  # [batch, controller_dim + num_read_heads * memory_dim]
            output = self.output_proj(combined)  # [batch, output_dim]
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)  # [batch, seq_len, output_dim]
        return outputs, memory


class DifferentiableNeuralComputer(nn.Module):
    """
    Differentiable Neural Computer (DNC).
    
    Complex memory operations with temporal link matrix.
    """
    
    def __init__(self, input_dim: int, output_dim: int, memory_size: int = 128,
                 memory_dim: int = 64, num_read_heads: int = 4, num_write_heads: int = 1,
                 controller_dim: int = 128, link_matrix_size: int = 16):
        """
        Initialize DNC.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            memory_size: Memory size
            memory_dim: Memory dimension
            num_read_heads: Number of read heads
            num_write_heads: Number of write heads
            controller_dim: Controller dimension
            link_matrix_size: Link matrix size
        """
        super(DifferentiableNeuralComputer, self).__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.controller_dim = controller_dim
        self.link_matrix_size = link_matrix_size
        
        # Memory
        self.register_buffer('memory', torch.zeros(memory_size, memory_dim))
        self.register_buffer('usage', torch.zeros(memory_size))
        self.register_buffer('link_matrix', torch.zeros(memory_size, memory_size))
        
        # Controller
        self.controller = nn.LSTM(input_dim, controller_dim, batch_first=True)
        
        # Read heads
        self.read_heads = nn.ModuleList([
            nn.Linear(controller_dim, memory_size) for _ in range(num_read_heads)
        ])
        
        # Write heads
        self.write_heads = nn.ModuleList([
            nn.ModuleDict({
                'key': nn.Linear(controller_dim, memory_dim),
                'erase': nn.Linear(controller_dim, memory_dim),
                'add': nn.Linear(controller_dim, memory_dim),
                'allocation': nn.Linear(controller_dim, memory_size)
            }) for _ in range(num_write_heads)
        ])
        
        # Output
        self.output_proj = nn.Linear(controller_dim + num_read_heads * memory_dim, output_dim)
    
    def allocation_addressing(self, usage: torch.Tensor) -> torch.Tensor:
        """
        Allocation-based addressing.
        
        Args:
            usage: Usage vector [memory_size]
            
        Returns:
            Allocation weights [memory_size]
        """
        # Free locations
        free = 1 - usage
        free_sorted, indices = torch.sort(free, descending=True)
        
        # Cumulative allocation
        allocation = torch.zeros_like(free)
        cumprod = 1.0
        for i in range(self.memory_size):
            idx = indices[i]
            allocation[idx] = free_sorted[i] * cumprod
            cumprod *= (1 - free_sorted[i])
        
        return allocation
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            
        Returns:
            Output tensor [batch, seq_len, output_dim]
            Memory state [memory_size, memory_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # Controller
        controller_out, _ = self.controller(x)
        
        outputs = []
        memory = self.memory.clone()
        usage = self.usage.clone()
        
        for t in range(seq_len):
            controller_state = controller_out[:, t, :]
            
            # Read
            read_weights_list = []
            for read_head in self.read_heads:
                weights = F.softmax(read_head(controller_state), dim=-1)
                read_weights_list.append(weights)
            
            read_weights = torch.stack(read_weights_list, dim=1)
            read_vectors = torch.matmul(read_weights, memory).view(batch_size, self.num_read_heads, self.memory_dim)
            
            # Write
            for write_head in self.write_heads:
                key = write_head['key'](controller_state)
                erase = torch.sigmoid(write_head['erase'](controller_state))
                add = torch.tanh(write_head['add'](controller_state))
                
                # Content-based addressing
                key_norm = F.normalize(key, p=2, dim=-1)
                memory_norm = F.normalize(memory, p=2, dim=-1)
                content_weights = F.softmax(torch.matmul(key_norm, memory_norm.t()), dim=-1)
                
                # Allocation-based addressing
                alloc_weights = self.allocation_addressing(usage)
                alloc_weights = alloc_weights.unsqueeze(0).expand(batch_size, -1)
                
                # Combine
                write_weights = 0.5 * content_weights + 0.5 * alloc_weights
                write_weights = write_weights / (write_weights.sum(dim=-1, keepdim=True) + 1e-10)
                
                # Update memory
                erase_weights = write_weights.unsqueeze(-1) * erase.unsqueeze(1)  # [batch, memory_size, memory_dim]
                memory = memory * (1 - erase_weights.mean(dim=0))
                add_weights = write_weights.unsqueeze(-1) * add.unsqueeze(1)  # [batch, memory_size, memory_dim]
                memory = memory + add_weights.mean(dim=0)
                
                # Update usage
                usage = usage + write_weights.mean(dim=0) * (1 - usage)
            
            # Output
            read_flat = read_vectors.view(batch_size, -1)
            combined = torch.cat([controller_state, read_flat], dim=-1)
            output = self.output_proj(combined)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs, memory


class EpisodicMemory:
    """
    Episodic Memory for rare events.
    
    Stores and retrieves experiences of rare events (accidents, emergencies).
    """
    
    def __init__(self, capacity: int = 1000, similarity_threshold: float = 0.7):
        """
        Initialize episodic memory.
        
        Args:
            capacity: Maximum number of episodes
            similarity_threshold: Similarity threshold for retrieval
        """
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.memory = deque(maxlen=capacity)
        self.rare_event_memory = deque(maxlen=capacity // 10)  # Separate rare events
    
    def store(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, is_rare: bool = False):
        """
        Store experience.
        
        Args:
            state: State
            action: Action
            reward: Reward
            next_state: Next state
            is_rare: Whether this is a rare event
        """
        episode = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'is_rare': is_rare
        }
        self.memory.append(episode)
        if is_rare:
            self.rare_event_memory.append(episode)
    
    def retrieve(self, query_state: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Retrieve similar experiences.
        
        Args:
            query_state: Query state
            k: Number of experiences to retrieve
            
        Returns:
            List of similar experiences
        """
        if len(self.memory) == 0:
            return []
        
        # Compute similarities
        similarities = []
        for episode in self.memory:
            state = episode['state']
            similarity = np.dot(query_state, state) / (
                np.linalg.norm(query_state) * np.linalg.norm(state) + 1e-10
            )
            if similarity >= self.similarity_threshold:
                similarities.append((similarity, episode))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k
        return [ep for _, ep in similarities[:k]]
    
    def retrieve_rare_events(self, query_state: np.ndarray, k: int = 3) -> List[Dict]:
        """
        Retrieve rare event experiences.
        
        Args:
            query_state: Query state
            k: Number of experiences to retrieve
            
        Returns:
            List of rare event experiences
        """
        if len(self.rare_event_memory) == 0:
            return []
        
        # Compute similarities
        similarities = []
        for episode in self.rare_event_memory:
            state = episode['state']
            similarity = np.dot(query_state, state) / (
                np.linalg.norm(query_state) * np.linalg.norm(state) + 1e-10
            )
            similarities.append((similarity, episode))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k
        return [ep for _, ep in similarities[:k]]


class MemoryAugmentedAgent(nn.Module):
    """
    Memory-Augmented Agent.
    
    Combines base agent with memory mechanisms.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: MemoryConfig):
        """
        Initialize memory-augmented agent.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            config: Memory configuration
        """
        super(MemoryAugmentedAgent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Base network
        self.base_net = nn.Sequential(
            nn.Linear(state_dim, config.controller_dim),
            nn.ReLU(),
            nn.Linear(config.controller_dim, config.controller_dim),
            nn.ReLU()
        )
        
        # Memory mechanism
        if config.memory_type == "ntm":
            self.memory_net = NeuralTuringMachine(
                input_dim=state_dim,
                output_dim=action_dim,
                memory_size=config.memory_size,
                memory_dim=config.memory_dim,
                num_read_heads=config.num_read_heads,
                num_write_heads=config.num_write_heads,
                controller_dim=config.controller_dim
            )
        elif config.memory_type == "dnc":
            self.memory_net = DifferentiableNeuralComputer(
                input_dim=state_dim,
                output_dim=action_dim,
                memory_size=config.memory_size,
                memory_dim=config.memory_dim,
                num_read_heads=config.num_read_heads,
                num_write_heads=config.num_write_heads,
                controller_dim=config.controller_dim
            )
        else:
            self.memory_net = None
        
        # Episodic memory
        if config.memory_type == "episodic":
            self.episodic_memory = EpisodicMemory(
                capacity=config.episode_capacity,
                similarity_threshold=config.similarity_threshold
            )
        else:
            self.episodic_memory = None
        
        # Output projection
        if self.memory_net is None:
            self.output_proj = nn.Linear(config.controller_dim, action_dim)
        else:
            self.output_proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, state_dim] or [batch, state_dim]
            
        Returns:
            Q-values [batch, action_dim] or [batch, seq_len, action_dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, state_dim]
        
        if self.memory_net is not None:
            output, _ = self.memory_net(x)
            return output
        else:
            # Base network
            base_out = self.base_net(x)  # [batch, seq_len, controller_dim]
            output = self.output_proj(base_out)  # [batch, seq_len, action_dim]
            return output

