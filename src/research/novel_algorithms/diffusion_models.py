"""
Diffusion Models for Traffic Signal Control.

Uses diffusion models to generate realistic traffic scenarios
and learn optimal signal timing policies.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple
import math

logger = logging.getLogger(__name__)


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings for timesteps.
        
        Args:
            time: Timestep tensor [batch]
            
        Returns:
            Embeddings [batch, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class DiffusionUNet(nn.Module):
    """
    U-Net for diffusion model.
    
    Predicts noise to be removed at each diffusion step.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        time_embed_dim: int = 128,
        hidden_dims: List[int] = [128, 256, 512],
    ):
        """
        Initialize U-Net.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            time_embed_dim: Time embedding dimension
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        input_dim = state_dim + action_dim  # Concatenate state and action
        
        # Time embedding
        self.time_embed = SinusoidalPositionalEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Encoder
        self.encoder = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.encoder.append(nn.Sequential(
                nn.Linear(prev_dim + time_embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
            prev_dim = hidden_dim
        
        # Decoder
        self.decoder = nn.ModuleList()
        for i, hidden_dim in enumerate(reversed(hidden_dims[:-1])):
            self.decoder.append(nn.Sequential(
                nn.Linear(hidden_dims[-1] + time_embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
        
        # Output
        self.output = nn.Linear(hidden_dims[0] + time_embed_dim, input_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Noisy input [batch, state_dim + action_dim]
            t: Timestep [batch]
            
        Returns:
            Predicted noise [batch, state_dim + action_dim]
        """
        # Time embedding
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)
        
        # Encoder
        h = x
        for layer in self.encoder:
            h = torch.cat([h, t_emb], dim=1)
            h = layer(h)
        
        # Decoder
        for layer in self.decoder:
            h = torch.cat([h, t_emb], dim=1)
            h = layer(h)
        
        # Output
        h = torch.cat([h, t_emb], dim=1)
        noise_pred = self.output(h)
        
        return noise_pred


class DiffusionTrafficAgent:
    """
    Diffusion Model Agent for Traffic Control.
    
    Uses diffusion process to learn optimal action distributions.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_diffusion_steps: int = 1000,
        device: str = "cpu",
    ):
        """
        Initialize diffusion agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            num_diffusion_steps: Number of diffusion steps
            device: Device for computation
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.device = device
        
        # Diffusion model
        self.model = DiffusionUNet(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        
        # Diffusion schedule
        self.beta = self._linear_beta_schedule(num_diffusion_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        
        self.is_trained = False
        logger.info("Initialized Diffusion Agent")
    
    def _linear_beta_schedule(self, num_steps: int) -> torch.Tensor:
        """Linear beta schedule for diffusion."""
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, num_steps).to(self.device)
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample from q(x_t | x_0).
        
        Args:
            x_start: Starting state [batch, dim]
            t: Timestep [batch]
            noise: Optional noise tensor
            
        Returns:
            Noisy sample [batch, dim]
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t])
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod[t])
        
        return (
            sqrt_alpha_cumprod.unsqueeze(-1) * x_start +
            sqrt_one_minus_alpha_cumprod.unsqueeze(-1) * noise
        )
    
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample from p(x_{t-1} | x_t).
        
        Args:
            x_t: Current noisy sample [batch, dim]
            t: Timestep [batch]
            state: Current state [batch, state_dim]
            
        Returns:
            Denoised sample [batch, dim]
        """
        # Predict noise
        noise_pred = self.model(x_t, t)
        
        # Extract action prediction
        action_pred = noise_pred[:, self.state_dim:]
        
        # Denoise
        alpha_t = self.alpha[t].unsqueeze(-1)
        alpha_cumprod_t = self.alpha_cumprod[t].unsqueeze(-1)
        beta_t = self.beta[t].unsqueeze(-1)
        
        # Predict x_0
        pred_x0 = (x_t - torch.sqrt(1.0 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
        
        # Sample x_{t-1}
        pred_dir = torch.sqrt(alpha_t) * beta_t / (1.0 - alpha_cumprod_t) * noise_pred
        prev_mean = torch.sqrt(alpha_t) * pred_x0 + pred_dir
        
        if t[0] == 0:
            return prev_mean
        else:
            noise = torch.randn_like(x_t)
            prev_var = beta_t * (1.0 - self.alpha_cumprod[t - 1]) / (1.0 - alpha_cumprod_t)
            return prev_mean + torch.sqrt(prev_var) * noise
    
    def p_sample_loop(
        self,
        state: np.ndarray,
        shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Full diffusion sampling loop.
        
        Args:
            state: Current state
            shape: Shape of action to generate
            
        Returns:
            Generated action [batch, action_dim]
        """
        batch_size = 1
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Start from noise
        action = torch.randn(batch_size, self.action_dim).to(self.device)
        x_t = torch.cat([state_tensor, action], dim=1)
        
        # Reverse diffusion
        for i in reversed(range(self.num_diffusion_steps)):
            t = torch.full((batch_size,), i, dtype=torch.long).to(self.device)
            x_t = self.p_sample(x_t, t, state_tensor)
        
        # Extract action
        generated_action = x_t[:, self.state_dim:]
        
        return generated_action
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using diffusion model.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        if not self.is_trained:
            return np.random.randint(0, self.action_dim)
        
        # Generate action using diffusion
        generated_action = self.p_sample_loop(state, (self.action_dim,))
        
        # Convert to discrete action
        action_probs = torch.softmax(generated_action, dim=1)
        action = torch.argmax(action_probs, dim=1).item()
        
        return int(action)
    
    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> Dict[str, float]:
        """
        Train diffusion model.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            
        Returns:
            Training metrics
        """
        batch_size = len(states)
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        
        # Concatenate state and action
        x_0 = torch.cat([states_tensor, actions_tensor], dim=1)
        
        # Sample timesteps
        t = torch.randint(0, self.num_diffusion_steps, (batch_size,)).to(self.device)
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Forward diffusion
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict noise
        noise_pred = self.model(x_t, t)
        
        # Loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.is_trained = True
        
        return {"loss": loss.item()}
    
    def save(self, path: str):
        """Save agent."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'is_trained': self.is_trained,
            'beta': self.beta.cpu(),
            'alpha_cumprod': self.alpha_cumprod.cpu(),
        }, path)
        logger.info(f"Saved Diffusion Agent to {path}")
    
    def load(self, path: str):
        """Load agent."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.is_trained = checkpoint.get('is_trained', False)
        self.beta = checkpoint.get('beta', self.beta).to(self.device)
        self.alpha_cumprod = checkpoint.get('alpha_cumprod', self.alpha_cumprod).to(self.device)
        logger.info(f"Loaded Diffusion Agent from {path}")

