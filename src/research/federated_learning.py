"""
Federated Learning Framework for Traffic Control.

Privacy-preserving distributed learning across multiple intersections
without sharing raw data.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import copy

logger = logging.getLogger(__name__)


@dataclass
class FederatedModelUpdate:
    """Model update from a federated client."""
    client_id: str
    model_weights: Dict[str, np.ndarray]
    num_samples: int
    round_number: int


class FederatedLearningCoordinator:
    """
    Federated Learning Coordinator.
    
    Coordinates model aggregation across multiple clients (intersections)
    while preserving privacy.
    """
    
    def __init__(
        self,
        initial_model_weights: Dict[str, np.ndarray],
        aggregation_method: str = "fedavg",
        differential_privacy: bool = True,
        noise_scale: float = 0.1,
    ):
        """
        Initialize federated learning coordinator.
        
        Args:
            initial_model_weights: Initial global model weights
            aggregation_method: Aggregation method ("fedavg", "fedprox", etc.)
            differential_privacy: Whether to use differential privacy
            noise_scale: Noise scale for differential privacy
        """
        self.global_model_weights = copy.deepcopy(initial_model_weights)
        self.aggregation_method = aggregation_method
        self.differential_privacy = differential_privacy
        self.noise_scale = noise_scale
        self.round_number = 0
        self.client_updates: List[FederatedModelUpdate] = []
    
    def aggregate_updates(
        self,
        updates: List[FederatedModelUpdate],
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate model updates from clients.
        
        Args:
            updates: List of model updates from clients
            
        Returns:
            Aggregated global model weights
        """
        if len(updates) == 0:
            return self.global_model_weights
        
        logger.info(f"Aggregating {len(updates)} client updates")
        
        if self.aggregation_method == "fedavg":
            return self._federated_averaging(updates)
        elif self.aggregation_method == "fedprox":
            return self._federated_proximal(updates)
        else:
            return self._federated_averaging(updates)
    
    def _federated_averaging(
        self,
        updates: List[FederatedModelUpdate],
    ) -> Dict[str, np.ndarray]:
        """Federated Averaging (FedAvg) aggregation."""
        total_samples = sum(update.num_samples for update in updates)
        
        # Initialize aggregated weights
        aggregated_weights = {}
        for key in self.global_model_weights.keys():
            aggregated_weights[key] = np.zeros_like(self.global_model_weights[key])
        
        # Weighted average
        for update in updates:
            weight = update.num_samples / total_samples
            for key in update.model_weights.keys():
                aggregated_weights[key] += weight * update.model_weights[key]
        
        # Apply differential privacy if enabled
        if self.differential_privacy:
            aggregated_weights = self._add_differential_privacy_noise(
                aggregated_weights
            )
        
        return aggregated_weights
    
    def _federated_proximal(
        self,
        updates: List[FederatedModelUpdate],
    ) -> Dict[str, np.ndarray]:
        """Federated Proximal (FedProx) aggregation with regularization."""
        # Similar to FedAvg but with proximal term
        aggregated_weights = self._federated_averaging(updates)
        
        # Add proximal regularization (keep close to previous global model)
        mu = 0.01  # Proximal parameter
        for key in aggregated_weights.keys():
            aggregated_weights[key] = (
                aggregated_weights[key] + mu * self.global_model_weights[key]
            ) / (1 + mu)
        
        return aggregated_weights
    
    def _add_differential_privacy_noise(
        self,
        weights: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Add differential privacy noise to weights."""
        noisy_weights = {}
        for key, weight in weights.items():
            # Add Gaussian noise
            noise = np.random.normal(0, self.noise_scale, weight.shape)
            noisy_weights[key] = weight + noise
        
        return noisy_weights
    
    def update_global_model(
        self,
        updates: List[FederatedModelUpdate],
    ) -> Dict[str, Any]:
        """
        Update global model with aggregated client updates.
        
        Args:
            updates: List of client updates
            
        Returns:
            Training statistics
        """
        self.round_number += 1
        self.client_updates = updates
        
        # Aggregate updates
        aggregated_weights = self.aggregate_updates(updates)
        
        # Update global model
        self.global_model_weights = aggregated_weights
        
        return {
            "round": self.round_number,
            "num_clients": len(updates),
            "total_samples": sum(u.num_samples for u in updates),
        }


class FederatedClient:
    """
    Federated Learning Client (Intersection).
    
    Trains local model and sends updates to coordinator.
    """
    
    def __init__(
        self,
        client_id: str,
        local_model_weights: Dict[str, np.ndarray],
    ):
        """Initialize federated client."""
        self.client_id = client_id
        self.local_model_weights = copy.deepcopy(local_model_weights)
        self.local_data_size = 0
    
    def train_local_model(
        self,
        local_data: List[Tuple[np.ndarray, int, float, np.ndarray]],
        epochs: int = 5,
    ) -> Dict[str, Any]:
        """
        Train local model on local data.
        
        Args:
            local_data: Local training data
            epochs: Number of training epochs
            
        Returns:
            Training statistics
        """
        self.local_data_size = len(local_data)
        
        # Simplified local training (in production use actual training)
        # Update weights based on local gradients
        for epoch in range(epochs):
            for state, action, reward, next_state in local_data:
                # Simulated gradient update
                for key in self.local_model_weights.keys():
                    gradient = np.random.randn(*self.local_model_weights[key].shape) * 0.01
                    self.local_model_weights[key] += 0.001 * gradient
        
        return {
            "epochs": epochs,
            "data_size": self.local_data_size,
            "loss": 0.5,  # Simulated loss
        }
    
    def get_model_update(
        self,
        round_number: int,
    ) -> FederatedModelUpdate:
        """
        Get model update to send to coordinator.
        
        Args:
            round_number: Current federated learning round
            
        Returns:
            Model update
        """
        return FederatedModelUpdate(
            client_id=self.client_id,
            model_weights=copy.deepcopy(self.local_model_weights),
            num_samples=self.local_data_size,
            round_number=round_number,
        )
    
    def update_local_model(
        self,
        global_weights: Dict[str, np.ndarray],
    ) -> None:
        """Update local model with global weights."""
        self.local_model_weights = copy.deepcopy(global_weights)


class FederatedLearningSystem:
    """
    Complete Federated Learning System.
    
    Coordinates federated learning across multiple intersections.
    """
    
    def __init__(
        self,
        initial_model_weights: Dict[str, np.ndarray],
        num_clients: int = 4,
        aggregation_method: str = "fedavg",
        differential_privacy: bool = True,
    ):
        """Initialize federated learning system."""
        self.coordinator = FederatedLearningCoordinator(
            initial_model_weights,
            aggregation_method=aggregation_method,
            differential_privacy=differential_privacy,
        )
        
        # Create clients
        self.clients = [
            FederatedClient(f"client_{i}", initial_model_weights)
            for i in range(num_clients)
        ]
    
    def federated_round(
        self,
        client_data: List[List[Tuple[np.ndarray, int, float, np.ndarray]]],
    ) -> Dict[str, Any]:
        """
        Execute one federated learning round.
        
        Args:
            client_data: List of local data for each client
            
        Returns:
            Round statistics
        """
        # Train local models
        updates = []
        for client, data in zip(self.clients, client_data):
            client.train_local_model(data, epochs=5)
            update = client.get_model_update(self.coordinator.round_number + 1)
            updates.append(update)
        
        # Aggregate and update global model
        stats = self.coordinator.update_global_model(updates)
        
        # Distribute global model to clients
        global_weights = self.coordinator.global_model_weights
        for client in self.clients:
            client.update_local_model(global_weights)
        
        return stats
    
    def train(
        self,
        num_rounds: int = 10,
        client_data: Optional[List[List[Tuple]]] = None,
    ) -> Dict[str, Any]:
        """
        Train federated model for multiple rounds.
        
        Args:
            num_rounds: Number of federated learning rounds
            client_data: Training data for each client
            
        Returns:
            Training statistics
        """
        logger.info(f"Starting federated learning for {num_rounds} rounds")
        
        if client_data is None:
            # Generate dummy data for testing
            client_data = [
                [(np.random.rand(12), 0, 0.5, np.random.rand(12)) for _ in range(100)]
                for _ in range(len(self.clients))
            ]
        
        training_stats = {
            "rounds": [],
            "num_clients": len(self.clients),
        }
        
        for round_num in range(num_rounds):
            round_stats = self.federated_round(client_data)
            training_stats["rounds"].append(round_stats)
            logger.info(f"Round {round_num + 1}/{num_rounds} completed")
        
        return training_stats

