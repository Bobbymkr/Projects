"""
Training Stability Framework for Reinforcement Learning.

Implements Phase 0.2 from OPTIMIZATION_ROADMAP.md:
- Gradient clipping
- Learning rate scheduling
- Target network updates
- Experience replay strategies
- Exploration schedules
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Callable, Dict, Any
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau


class GradientClipper:
    """Gradient clipping utilities for training stability."""
    
    @staticmethod
    def clip_grad_norm(parameters, max_norm: float = 10.0, norm_type: float = 2.0):
        """
        Clip gradients by norm.
        
        Args:
            parameters: Model parameters
            max_norm: Maximum gradient norm
            norm_type: Type of norm (2.0 for L2, float('inf') for L-inf)
        """
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)
    
    @staticmethod
    def clip_grad_value(parameters, clip_value: float = 10.0):
        """
        Clip gradients by value.
        
        Args:
            parameters: Model parameters
            clip_value: Maximum absolute gradient value
        """
        return torch.nn.utils.clip_grad_value_(parameters, clip_value)


class LearningRateScheduler:
    """Learning rate scheduling for stable training."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = "cosine",
        **kwargs
    ):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: Type of scheduler ("cosine", "cosine_warm_restart", "plateau", "step")
            **kwargs: Scheduler-specific parameters
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        
        if scheduler_type == "cosine":
            T_max = kwargs.get("T_max", 1000)
            eta_min = kwargs.get("eta_min", 1e-6)
            self.scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        
        elif scheduler_type == "cosine_warm_restart":
            T_0 = kwargs.get("T_0", 100)
            T_mult = kwargs.get("T_mult", 2)
            eta_min = kwargs.get("eta_min", 1e-6)
            self.scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
            )
        
        elif scheduler_type == "plateau":
            mode = kwargs.get("mode", "max")
            factor = kwargs.get("factor", 0.5)
            patience = kwargs.get("patience", 10)
            self.scheduler = ReduceLROnPlateau(
                optimizer, mode=mode, factor=factor, patience=patience
            )
        
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def step(self, metric: Optional[float] = None):
        """
        Step the scheduler.
        
        Args:
            metric: Metric value for plateau scheduler (optional)
        """
        if self.scheduler_type == "plateau" and metric is not None:
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


class TargetNetworkUpdater:
    """Soft and hard target network update strategies."""
    
    @staticmethod
    def soft_update(
        target_net: nn.Module,
        policy_net: nn.Module,
        tau: float = 0.005
    ):
        """
        Soft update target network: θ_target = τ * θ_policy + (1 - τ) * θ_target
        
        Args:
            target_net: Target network
            policy_net: Policy network
            tau: Soft update coefficient (0.005 recommended)
        """
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(
                tau * policy_param.data + (1.0 - tau) * target_param.data
            )
    
    @staticmethod
    def hard_update(
        target_net: nn.Module,
        policy_net: nn.Module
    ):
        """
        Hard update target network: θ_target = θ_policy
        
        Args:
            target_net: Target network
            policy_net: Policy network
        """
        target_net.load_state_dict(policy_net.state_dict())


class ExplorationSchedule:
    """Exploration rate scheduling for ε-greedy and UCB strategies."""
    
    def __init__(
        self,
        initial_epsilon: float = 1.0,
        final_epsilon: float = 0.01,
        decay_type: str = "linear",
        decay_steps: int = 10000,
        **kwargs
    ):
        """
        Initialize exploration schedule.
        
        Args:
            initial_epsilon: Initial exploration rate
            final_epsilon: Final exploration rate
            decay_type: Type of decay ("linear", "exponential", "cosine")
            decay_steps: Number of steps for decay
            **kwargs: Additional parameters
        """
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.decay_type = decay_type
        self.decay_steps = decay_steps
        self.current_step = 0
        
        if decay_type == "exponential":
            self.decay_rate = kwargs.get("decay_rate", 0.995)
    
    def get_epsilon(self) -> float:
        """Get current exploration rate."""
        if self.decay_type == "linear":
            if self.current_step >= self.decay_steps:
                return self.final_epsilon
            progress = self.current_step / self.decay_steps
            return self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) * progress
        
        elif self.decay_type == "exponential":
            epsilon = max(
                self.final_epsilon,
                self.initial_epsilon * (self.decay_rate ** self.current_step)
            )
            return epsilon
        
        elif self.decay_type == "cosine":
            if self.current_step >= self.decay_steps:
                return self.final_epsilon
            progress = self.current_step / self.decay_steps
            epsilon = self.final_epsilon + (self.initial_epsilon - self.final_epsilon) * \
                     (1 + np.cos(np.pi * progress)) / 2
            return epsilon
        
        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")
    
    def step(self):
        """Advance exploration schedule by one step."""
        self.current_step += 1
    
    def reset(self):
        """Reset exploration schedule."""
        self.current_step = 0


class TrainingStabilityFramework:
    """
    Comprehensive training stability framework.
    
    Combines all stability techniques for robust RL training.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        policy_net: nn.Module,
        target_net: Optional[nn.Module] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize training stability framework.
        
        Args:
            optimizer: PyTorch optimizer
            policy_net: Policy network
            target_net: Target network (optional)
            config: Configuration dictionary
        """
        self.optimizer = optimizer
        self.policy_net = policy_net
        self.target_net = target_net
        
        config = config or {}
        
        # Gradient clipping
        self.grad_clip_norm = config.get("grad_clip_norm", 10.0)
        self.grad_clip_value = config.get("grad_clip_value", None)
        
        # Learning rate scheduling
        lr_scheduler_config = config.get("lr_scheduler", {})
        self.lr_scheduler = None
        if lr_scheduler_config.get("enabled", True):
            self.lr_scheduler = LearningRateScheduler(
                optimizer,
                scheduler_type=lr_scheduler_config.get("type", "cosine"),
                **lr_scheduler_config.get("params", {})
            )
        
        # Target network updates
        self.target_update_frequency = config.get("target_update_frequency", 1)
        self.target_update_tau = config.get("target_update_tau", 0.005)
        self.use_soft_update = config.get("use_soft_update", True)
        self.update_counter = 0
        
        # Exploration schedule
        exploration_config = config.get("exploration", {})
        self.exploration_schedule = None
        if exploration_config.get("enabled", True):
            self.exploration_schedule = ExplorationSchedule(**exploration_config)
    
    def clip_gradients(self):
        """Clip gradients using configured method."""
        if self.grad_clip_norm is not None:
            GradientClipper.clip_grad_norm(
                self.policy_net.parameters(),
                max_norm=self.grad_clip_norm
            )
        if self.grad_clip_value is not None:
            GradientClipper.clip_grad_value(
                self.policy_net.parameters(),
                clip_value=self.grad_clip_value
            )
    
    def update_target_network(self):
        """Update target network using configured strategy."""
        if self.target_net is None:
            return
        
        self.update_counter += 1
        
        if self.use_soft_update:
            # Soft update every step
            TargetNetworkUpdater.soft_update(
                self.target_net,
                self.policy_net,
                tau=self.target_update_tau
            )
        else:
            # Hard update at specified frequency
            if self.update_counter % self.target_update_frequency == 0:
                TargetNetworkUpdater.hard_update(self.target_net, self.policy_net)
    
    def step_scheduler(self, metric: Optional[float] = None):
        """Step learning rate scheduler."""
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(metric)
    
    def get_exploration_rate(self) -> Optional[float]:
        """Get current exploration rate."""
        if self.exploration_schedule is not None:
            return self.exploration_schedule.get_epsilon()
        return None
    
    def step_exploration(self):
        """Advance exploration schedule."""
        if self.exploration_schedule is not None:
            self.exploration_schedule.step()
    
    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

