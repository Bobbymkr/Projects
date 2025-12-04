"""
Advanced Data Augmentation for Traffic Control.

Implements Phase 4.3 from OPTIMIZATION_ROADMAP.md:
- Traffic Flow Variations: Poisson, Weibull, log-normal
- Vehicle Type Diversity: Cars, trucks, buses, motorcycles
- Pedestrian Patterns: Crosswalk usage, jaywalking
- Emergency Vehicles: Priority scenarios
- Sensor Noise: Camera occlusion, detection failures

Expected Impact: 15-20% robustness improvement
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import random
import logging

logger = logging.getLogger(__name__)


class VehicleType(Enum):
    """Vehicle types."""
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    EMERGENCY = "emergency"


class DistributionType(Enum):
    """Distribution types for traffic flow."""
    POISSON = "poisson"
    WEIBULL = "weibull"
    LOG_NORMAL = "log_normal"
    EXPONENTIAL = "exponential"
    UNIFORM = "uniform"


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    # Traffic flow
    flow_distribution: DistributionType = DistributionType.POISSON
    flow_variance: float = 0.1  # Variance in arrival rates
    
    # Vehicle types
    vehicle_type_diversity: bool = True
    vehicle_type_proportions: Dict[str, float] = None  # Default: all equal
    
    # Pedestrians
    pedestrian_enabled: bool = False
    pedestrian_rate: float = 0.1  # Pedestrians per second
    jaywalking_probability: float = 0.05
    
    # Emergency vehicles
    emergency_enabled: bool = False
    emergency_rate: float = 0.01  # Emergency vehicles per hour
    emergency_priority: bool = True
    
    # Sensor noise
    sensor_noise_enabled: bool = False
    sensor_noise_level: float = 0.05  # Standard deviation
    occlusion_probability: float = 0.02  # Camera occlusion
    detection_failure_rate: float = 0.01  # Detection failures
    
    def __post_init__(self):
        """Initialize default values."""
        if self.vehicle_type_proportions is None:
            self.vehicle_type_proportions = {
                VehicleType.CAR.value: 0.7,
                VehicleType.TRUCK.value: 0.15,
                VehicleType.BUS.value: 0.1,
                VehicleType.MOTORCYCLE.value: 0.05
            }


class TrafficFlowAugmentation:
    """
    Augment traffic flow with different distributions.
    """
    
    def __init__(self, config: AugmentationConfig):
        """
        Initialize traffic flow augmentation.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config
    
    def sample_arrival_rate(self, base_rate: float) -> float:
        """
        Sample arrival rate from distribution.
        
        Args:
            base_rate: Base arrival rate
            
        Returns:
            Sampled arrival rate
        """
        if self.config.flow_distribution == DistributionType.POISSON:
            # Poisson: rate is the mean
            return np.random.poisson(base_rate * 100) / 100.0
        
        elif self.config.flow_distribution == DistributionType.WEIBULL:
            # Weibull distribution
            shape = 2.0
            scale = base_rate
            sample = np.random.weibull(shape) * scale
            return max(0.0, sample)
        
        elif self.config.flow_distribution == DistributionType.LOG_NORMAL:
            # Log-normal distribution
            mean = np.log(base_rate)
            std = self.config.flow_variance
            sample = np.random.lognormal(mean, std)
            return max(0.0, sample)
        
        elif self.config.flow_distribution == DistributionType.EXPONENTIAL:
            # Exponential distribution
            sample = np.random.exponential(base_rate)
            return max(0.0, sample)
        
        else:  # UNIFORM
            # Uniform distribution
            low = base_rate * (1 - self.config.flow_variance)
            high = base_rate * (1 + self.config.flow_variance)
            return np.random.uniform(low, high)
    
    def augment_arrival_rates(self, base_rates: List[float]) -> List[float]:
        """
        Augment arrival rates for all lanes.
        
        Args:
            base_rates: Base arrival rates
            
        Returns:
            Augmented arrival rates
        """
        return [self.sample_arrival_rate(rate) for rate in base_rates]


class VehicleTypeAugmentation:
    """
    Augment vehicle types for diversity.
    """
    
    def __init__(self, config: AugmentationConfig):
        """
        Initialize vehicle type augmentation.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config
        self.vehicle_types = list(self.config.vehicle_type_proportions.keys())
        self.proportions = list(self.config.vehicle_type_proportions.values())
    
    def sample_vehicle_type(self) -> str:
        """
        Sample vehicle type based on proportions.
        
        Returns:
            Vehicle type
        """
        return np.random.choice(self.vehicle_types, p=self.proportions)
    
    def get_vehicle_properties(self, vehicle_type: str) -> Dict[str, Any]:
        """
        Get properties for vehicle type.
        
        Args:
            vehicle_type: Vehicle type
            
        Returns:
            Vehicle properties
        """
        properties = {
            VehicleType.CAR.value: {
                "length": 4.5,
                "width": 1.8,
                "acceleration": 2.0,
                "max_speed": 50.0,
                "occupancy": 1.5
            },
            VehicleType.TRUCK.value: {
                "length": 12.0,
                "width": 2.5,
                "acceleration": 1.0,
                "max_speed": 40.0,
                "occupancy": 1.0
            },
            VehicleType.BUS.value: {
                "length": 12.0,
                "width": 2.5,
                "acceleration": 1.2,
                "max_speed": 45.0,
                "occupancy": 40.0
            },
            VehicleType.MOTORCYCLE.value: {
                "length": 2.0,
                "width": 0.8,
                "acceleration": 3.0,
                "max_speed": 60.0,
                "occupancy": 1.0
            },
            VehicleType.EMERGENCY.value: {
                "length": 5.0,
                "width": 2.0,
                "acceleration": 3.5,
                "max_speed": 80.0,
                "occupancy": 2.0,
                "priority": True
            }
        }
        return properties.get(vehicle_type, properties[VehicleType.CAR.value])


class PedestrianAugmentation:
    """
    Augment pedestrian patterns.
    """
    
    def __init__(self, config: AugmentationConfig):
        """
        Initialize pedestrian augmentation.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config
    
    def should_pedestrian_cross(self, time_step: int) -> bool:
        """
        Determine if pedestrian should cross.
        
        Args:
            time_step: Current time step
            
        Returns:
            True if pedestrian crosses
        """
        if not self.config.pedestrian_enabled:
            return False
        
        # Poisson process for pedestrian arrivals
        probability = self.config.pedestrian_rate * 0.1  # Per 0.1 second
        return np.random.random() < probability
    
    def is_jaywalking(self) -> bool:
        """
        Determine if pedestrian is jaywalking.
        
        Returns:
            True if jaywalking
        """
        return np.random.random() < self.config.jaywalking_probability
    
    def get_pedestrian_delay(self, is_jaywalking: bool) -> float:
        """
        Get delay caused by pedestrian.
        
        Args:
            is_jaywalking: Whether pedestrian is jaywalking
            
        Returns:
            Delay in seconds
        """
        if is_jaywalking:
            return np.random.uniform(2.0, 5.0)  # Longer delay for jaywalking
        else:
            return np.random.uniform(1.0, 3.0)  # Normal crosswalk delay


class EmergencyVehicleAugmentation:
    """
    Augment emergency vehicle scenarios.
    """
    
    def __init__(self, config: AugmentationConfig):
        """
        Initialize emergency vehicle augmentation.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config
    
    def should_emergency_arrive(self, time_step: int) -> bool:
        """
        Determine if emergency vehicle should arrive.
        
        Args:
            time_step: Current time step
            
        Returns:
            True if emergency vehicle arrives
        """
        if not self.config.emergency_enabled:
            return False
        
        # Convert rate to probability per time step
        probability = self.config.emergency_rate / 3600.0  # Per second
        return np.random.random() < probability
    
    def get_emergency_priority(self) -> Dict[str, Any]:
        """
        Get emergency vehicle priority information.
        
        Returns:
            Priority information
        """
        return {
            "priority": True,
            "lane_preference": np.random.randint(0, 4),  # Random lane
            "clearance_time": np.random.uniform(10.0, 30.0),  # Seconds to clear
            "speed_multiplier": 1.5  # Faster than normal
        }


class SensorNoiseAugmentation:
    """
    Augment sensor noise and failures.
    """
    
    def __init__(self, config: AugmentationConfig):
        """
        Initialize sensor noise augmentation.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config
        self.occluded_sensors = set()
        self.failed_sensors = set()
    
    def add_noise(self, observation: np.ndarray) -> np.ndarray:
        """
        Add noise to observation.
        
        Args:
            observation: Original observation
            
        Returns:
            Noisy observation
        """
        if not self.config.sensor_noise_enabled:
            return observation
        
        noisy = observation.copy()
        
        # Add Gaussian noise
        noise = np.random.normal(0, self.config.sensor_noise_level, size=observation.shape)
        noisy = noisy + noise
        
        # Apply occlusion
        for i in range(len(observation)):
            if np.random.random() < self.config.occlusion_probability:
                self.occluded_sensors.add(i)
                noisy[i] = 0.0  # Occluded sensor reads 0
        
        # Apply detection failures
        for i in range(len(observation)):
            if np.random.random() < self.config.detection_failure_rate:
                self.failed_sensors.add(i)
                noisy[i] = np.random.uniform(0, observation[i])  # Random incorrect reading
        
        # Remove occlusions/failures after some time
        if np.random.random() < 0.1:  # 10% chance of recovery
            if self.occluded_sensors:
                self.occluded_sensors.pop()
            if self.failed_sensors:
                self.failed_sensors.pop()
        
        return np.clip(noisy, 0.0, 1.0)
    
    def reset(self):
        """Reset sensor states."""
        self.occluded_sensors = set()
        self.failed_sensors = set()


class DataAugmentationPipeline:
    """
    Complete data augmentation pipeline.
    """
    
    def __init__(self, config: AugmentationConfig):
        """
        Initialize augmentation pipeline.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config
        self.flow_aug = TrafficFlowAugmentation(config)
        self.vehicle_aug = VehicleTypeAugmentation(config)
        self.pedestrian_aug = PedestrianAugmentation(config)
        self.emergency_aug = EmergencyVehicleAugmentation(config)
        self.sensor_aug = SensorNoiseAugmentation(config)
    
    def augment_environment_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Augment environment configuration.
        
        Args:
            base_config: Base configuration
            
        Returns:
            Augmented configuration
        """
        new_config = base_config.copy()
        
        # Augment arrival rates
        if "arrival_rates" in new_config:
            new_config["arrival_rates"] = self.flow_aug.augment_arrival_rates(
                new_config["arrival_rates"]
            )
        
        # Add vehicle type information
        if self.config.vehicle_type_diversity:
            new_config["vehicle_types"] = self.config.vehicle_type_proportions
        
        # Add pedestrian information
        if self.config.pedestrian_enabled:
            new_config["pedestrians"] = {
                "enabled": True,
                "rate": self.config.pedestrian_rate,
                "jaywalking_probability": self.config.jaywalking_probability
            }
        
        # Add emergency vehicle information
        if self.config.emergency_enabled:
            new_config["emergency_vehicles"] = {
                "enabled": True,
                "rate": self.config.emergency_rate,
                "priority": self.config.emergency_priority
            }
        
        return new_config
    
    def augment_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Augment observation with sensor noise.
        
        Args:
            observation: Original observation
            
        Returns:
            Augmented observation
        """
        return self.sensor_aug.add_noise(observation)
    
    def step_augmentation(self, time_step: int) -> Dict[str, Any]:
        """
        Perform step-level augmentation.
        
        Args:
            time_step: Current time step
            
        Returns:
            Augmentation information
        """
        info = {}
        
        # Check for pedestrians
        if self.pedestrian_aug.should_pedestrian_cross(time_step):
            is_jaywalking = self.pedestrian_aug.is_jaywalking()
            delay = self.pedestrian_aug.get_pedestrian_delay(is_jaywalking)
            info["pedestrian"] = {
                "crossing": True,
                "jaywalking": is_jaywalking,
                "delay": delay
            }
        
        # Check for emergency vehicles
        if self.emergency_aug.should_emergency_arrive(time_step):
            priority = self.emergency_aug.get_emergency_priority()
            info["emergency"] = priority
        
        return info
    
    def reset(self):
        """Reset augmentation state."""
        self.sensor_aug.reset()

