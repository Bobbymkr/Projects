"""
Self-Play & Adversarial Training.

Implements Phase 2.4 from OPTIMIZATION_ROADMAP.md:
- Adversarial Traffic: Worst-case traffic patterns
- Robustness Testing: Random failures, sensor noise
- Domain Randomization: Weather, lighting, vehicle types

Expected Impact: 30-40% robustness improvement
"""

import numpy as np
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import logging
import random

logger = logging.getLogger(__name__)


@dataclass
class AdversarialConfig:
    """Configuration for adversarial training."""
    enabled: bool = True
    adversarial_probability: float = 0.3  # Probability of adversarial scenario
    noise_level: float = 0.1  # Sensor noise level
    failure_probability: float = 0.05  # Probability of sensor failure
    domain_randomization: bool = True
    worst_case_traffic: bool = True


class AdversarialTrafficGenerator:
    """
    Generate adversarial traffic patterns.
    
    Creates worst-case scenarios to test robustness.
    """
    
    def __init__(self, base_arrival_rates: List[float]):
        """
        Initialize adversarial traffic generator.
        
        Args:
            base_arrival_rates: Base arrival rates per lane
        """
        self.base_arrival_rates = np.array(base_arrival_rates)
        self.scenarios = self._create_scenarios()
    
    def _create_scenarios(self) -> List[Dict[str, Any]]:
        """Create adversarial traffic scenarios."""
        return [
            {
                "name": "worst_case_all_lanes",
                "description": "Maximum traffic on all lanes simultaneously",
                "multiplier": 2.0,
            },
            {
                "name": "asymmetric_burst",
                "description": "Burst traffic on one direction",
                "multiplier": 3.0,
                "lane_mask": [1.0, 0.1, 0.1, 0.1],  # Heavy on lane 0
            },
            {
                "name": "oscillating_traffic",
                "description": "Oscillating traffic patterns",
                "multiplier": 1.5,
                "oscillating": True,
            },
            {
                "name": "sudden_surge",
                "description": "Sudden traffic surge",
                "multiplier": 2.5,
                "surge_duration": 300,  # seconds
            },
        ]
    
    def generate_adversarial_rates(self, scenario_name: Optional[str] = None, 
                                   time: int = 0) -> np.ndarray:
        """
        Generate adversarial arrival rates.
        
        Args:
            scenario_name: Name of scenario (random if None)
            time: Current time step
            
        Returns:
            Adversarial arrival rates
        """
        if scenario_name is None:
            scenario = random.choice(self.scenarios)
        else:
            scenario = next((s for s in self.scenarios if s["name"] == scenario_name), 
                           self.scenarios[0])
        
        rates = self.base_arrival_rates.copy()
        
        if scenario["name"] == "worst_case_all_lanes":
            rates = rates * scenario["multiplier"]
        
        elif scenario["name"] == "asymmetric_burst":
            mask = np.array(scenario.get("lane_mask", [1.0] * len(rates)))
            rates = rates * mask * scenario["multiplier"]
        
        elif scenario["name"] == "oscillating_traffic":
            oscillation = 1.0 + 0.5 * np.sin(time / 100.0)
            rates = rates * scenario["multiplier"] * oscillation
        
        elif scenario["name"] == "sudden_surge":
            if time % 600 < scenario.get("surge_duration", 300):
                rates = rates * scenario["multiplier"]
        
        return np.clip(rates, 0.0, 2.0)  # Cap at reasonable maximum


class SensorNoise:
    """
    Add sensor noise and failures for robustness testing.
    """
    
    def __init__(self, noise_level: float = 0.1, failure_probability: float = 0.05):
        """
        Initialize sensor noise.
        
        Args:
            noise_level: Standard deviation of noise (relative to signal)
            failure_probability: Probability of sensor failure per step
        """
        self.noise_level = noise_level
        self.failure_probability = failure_probability
        self.failed_sensors = set()
    
    def add_noise(self, observation: np.ndarray) -> np.ndarray:
        """
        Add noise to observation.
        
        Args:
            observation: Original observation
            
        Returns:
            Noisy observation
        """
        noisy = observation.copy()
        
        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_level, size=observation.shape)
        noisy = noisy + noise
        
        # Simulate sensor failures
        for i in range(len(observation)):
            if random.random() < self.failure_probability:
                self.failed_sensors.add(i)
                noisy[i] = 0.0  # Failed sensor reads 0
        
        # Remove failed sensors after some time (recovery)
        if random.random() < 0.1:  # 10% chance of recovery
            if self.failed_sensors:
                self.failed_sensors.pop()
        
        return np.clip(noisy, 0.0, 1.0)
    
    def reset(self):
        """Reset sensor failures."""
        self.failed_sensors = set()


class DomainRandomization:
    """
    Domain randomization for robustness.
    
    Randomizes environment parameters to improve generalization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize domain randomization.
        
        Args:
            config: Environment configuration
        """
        self.config = config
        self.randomization_params = {
            "arrival_rate_variance": 0.2,  # ±20% variance
            "queue_capacity_variance": 0.1,  # ±10% variance
            "saturation_flow_variance": 0.15,  # ±15% variance
        }
    
    def randomize_environment(self, env) -> Dict[str, Any]:
        """
        Randomize environment parameters.
        
        Args:
            env: Traffic environment
            
        Returns:
            Dictionary of randomized parameters
        """
        params = {}
        
        # Randomize arrival rates
        if "arrival_rate_variance" in self.randomization_params:
            variance = self.randomization_params["arrival_rate_variance"]
            base_rates = env.arrival_rates.copy()
            noise = np.random.uniform(-variance, variance, size=base_rates.shape)
            params["arrival_rates"] = np.clip(base_rates * (1 + noise), 0.0, 2.0)
        
        # Randomize queue capacity
        if "queue_capacity_variance" in self.randomization_params:
            variance = self.randomization_params["queue_capacity_variance"]
            base_capacity = env.queue_capacity
            noise = np.random.uniform(-variance, variance)
            params["queue_capacity"] = int(base_capacity * (1 + noise))
        
        # Randomize saturation flow (if accessible)
        if hasattr(env, 'sat_flow_per_sec'):
            variance = self.randomization_params.get("saturation_flow_variance", 0.15)
            base_flow = env.sat_flow_per_sec
            noise = np.random.uniform(-variance, variance)
            params["sat_flow_per_sec"] = base_flow * (1 + noise)
        
        return params
    
    def apply_randomization(self, env, params: Dict[str, Any]):
        """
        Apply randomized parameters to environment.
        
        Args:
            env: Traffic environment
            params: Randomized parameters
        """
        if "arrival_rates" in params:
            env.arrival_rates = params["arrival_rates"]
            env.cfg["arrival_rates"] = params["arrival_rates"].tolist()
        
        if "queue_capacity" in params:
            env.queue_capacity = params["queue_capacity"]
            env.cfg["queue_capacity"] = params["queue_capacity"]
        
        if "sat_flow_per_sec" in params and hasattr(env, 'sat_flow_per_sec'):
            env.sat_flow_per_sec = params["sat_flow_per_sec"]


class AdversarialTrainingWrapper:
    """
    Wrapper for adversarial training.
    
    Combines adversarial traffic, sensor noise, and domain randomization.
    """
    
    def __init__(
        self,
        env,
        config: Optional[AdversarialConfig] = None,
    ):
        """
        Initialize adversarial training wrapper.
        
        Args:
            env: Traffic environment
            config: Adversarial configuration
        """
        self.env = env
        self.config = config or AdversarialConfig()
        
        # Initialize components
        base_rates = env.cfg.get("arrival_rates", [0.3] * env.num_lanes)
        self.adversarial_generator = AdversarialTrafficGenerator(base_rates)
        self.sensor_noise = SensorNoise(
            noise_level=self.config.noise_level,
            failure_probability=self.config.failure_probability
        )
        self.domain_randomizer = DomainRandomization(env.cfg)
        
        self.time = 0
        self.adversarial_active = False
        
        logger.info("Adversarial training wrapper initialized")
    
    def reset(self, **kwargs):
        """
        Reset environment with potential adversarial setup.
        
        Args:
            **kwargs: Reset arguments
            
        Returns:
            Observation and info
        """
        # Reset sensor noise
        self.sensor_noise.reset()
        
        # Decide if this episode is adversarial
        self.adversarial_active = (random.random() < self.config.adversarial_probability)
        
        # Apply domain randomization
        if self.config.domain_randomization:
            params = self.domain_randomizer.randomize_environment(self.env)
            self.domain_randomizer.apply_randomization(self.env, params)
        
        # Reset environment
        obs, info = self.env.reset(**kwargs)
        
        # Apply sensor noise
        if self.config.enabled:
            obs = self.sensor_noise.add_noise(obs)
        
        self.time = 0
        return obs, info
    
    def step(self, action):
        """
        Step environment with adversarial modifications.
        
        Args:
            action: Action to take
            
        Returns:
            Observation, reward, terminated, truncated, info
        """
        # Apply adversarial traffic if active
        if self.adversarial_active and self.config.worst_case_traffic:
            adversarial_rates = self.adversarial_generator.generate_adversarial_rates(
                time=self.time
            )
            self.env.arrival_rates = adversarial_rates
            self.env.cfg["arrival_rates"] = adversarial_rates.tolist()
        
        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply sensor noise
        if self.config.enabled:
            obs = self.sensor_noise.add_noise(obs)
        
        self.time += 1
        
        # Add adversarial info
        info["adversarial"] = self.adversarial_active
        info["sensor_noise_level"] = self.sensor_noise.noise_level
        info["failed_sensors"] = len(self.sensor_noise.failed_sensors)
        
        return obs, reward, terminated, truncated, info
    
    @property
    def action_space(self):
        """Get action space."""
        return self.env.action_space
    
    @property
    def observation_space(self):
        """Get observation space."""
        return self.env.observation_space


class SelfPlayTrainer:
    """
    Self-Play Training for competitive scenarios.
    
    Trains agent against itself or adversarial policies.
    """
    
    def __init__(self, agent, env, opponent_pool_size: int = 5):
        """
        Initialize self-play trainer.
        
        Args:
            agent: Main agent to train
            env: Environment
            opponent_pool_size: Size of opponent policy pool
        """
        self.agent = agent
        self.env = env
        self.opponent_pool_size = opponent_pool_size
        self.opponent_pool: List[Any] = []  # Store opponent policies
    
    def create_opponent(self, agent_copy):
        """
        Create opponent from agent copy.
        
        Args:
            agent_copy: Copy of agent to use as opponent
            
        Returns:
            Opponent agent
        """
        return agent_copy
    
    def update_opponent_pool(self, current_agent):
        """
        Update opponent pool with current agent.
        
        Args:
            current_agent: Current agent state
        """
        # Add current agent to pool
        if len(self.opponent_pool) >= self.opponent_pool_size:
            # Remove oldest
            self.opponent_pool.pop(0)
        
        # Create copy of agent
        import copy
        opponent = copy.deepcopy(current_agent)
        self.opponent_pool.append(opponent)
    
    def select_opponent(self) -> Optional[Any]:
        """
        Select opponent from pool.
        
        Returns:
            Opponent agent or None
        """
        if len(self.opponent_pool) == 0:
            return None
        return random.choice(self.opponent_pool)
    
    def train_episode_with_opponent(self, opponent=None):
        """
        Train episode against opponent.
        
        Args:
            opponent: Opponent agent (None for self-play)
            
        Returns:
            Episode reward
        """
        obs, _ = self.env.reset()
        total_reward = 0.0
        done = False
        
        while not done:
            # Main agent selects action
            action = self.agent.select_action(obs)
            
            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Store experience
            if hasattr(self.agent, 'push'):
                # Note: In self-play, we'd need to handle opponent actions
                # For now, this is a simplified version
                pass
        
        return total_reward

