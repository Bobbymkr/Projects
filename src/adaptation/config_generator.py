"""Regional Configuration Generator.

Automatically generates optimal configuration files based on
technology recommendations and regional requirements.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .recommendation_engine import TechnologyStack, ControlAlgorithm, VisionModel, ForecastingModel, DeploymentArchitecture
from .checklist_parser import RegionalRequirements, InfrastructureLevel, TrafficDensity

logger = logging.getLogger(__name__)


class RegionalConfigGenerator:
    """Generator for regional configuration files."""
    
    def __init__(self):
        """Initialize the configuration generator."""
        self.logger = logging.getLogger(__name__)
        self._setup_default_configs()
    
    def _setup_default_configs(self):
        """Setup default configuration templates."""
        self.base_config = {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "cycle_yellow": 3,
            "cycle_all_red": 1,
            "arrival_rates": [0.3, 0.25, 0.3, 0.25],
            "queue_capacity": 40,
            "reward_weights": {
                "queue": -1.0,
                "wait_penalty": -0.1,
                "efficiency": 0.01,
                "max_queue": -0.05
            }
        }
    
    def generate(self, requirements: RegionalRequirements, 
                 tech_stack: TechnologyStack,
                 region_name: str = "new_region") -> Dict[str, Any]:
        """Generate complete regional configuration.
        
        Args:
            requirements: Regional requirements
            tech_stack: Recommended technology stack
            region_name: Name of the region
            
        Returns:
            Complete configuration dictionary
        """
        self.logger.info(f"Generating configuration for {region_name}...")
        
        config = self.base_config.copy()
        
        # Add metadata
        config["_description"] = f"Auto-generated configuration for {region_name}"
        config["_generated_at"] = datetime.now().isoformat()
        config["_technology_stack"] = tech_stack.to_dict()
        config["_region_type"] = self._determine_region_type(requirements)
        
        # Adapt traffic parameters
        config = self._adapt_traffic_parameters(config, requirements)
        
        # Adapt signal timing
        config = self._adapt_signal_timing(config, requirements)
        
        # Add vision configuration
        config["_vision_config"] = self._generate_vision_config(tech_stack.vision_model, requirements)
        
        # Add control configuration
        config["_control_config"] = self._generate_control_config(tech_stack.control_algorithm, requirements)
        
        # Add forecasting configuration
        config["_forecasting_config"] = self._generate_forecasting_config(tech_stack.forecasting_model, requirements)
        
        # Add deployment configuration
        config["_deployment_config"] = self._generate_deployment_config(tech_stack.deployment_architecture, requirements)
        
        # Add regional adaptations
        config["_regional_adaptations"] = self._generate_regional_adaptations(requirements)
        
        # Add recommendations
        config["_recommendations"] = {
            "confidence_score": tech_stack.confidence_score,
            "reasoning": tech_stack.reasoning,
            "estimated_cost": tech_stack.estimated_cost,
            "estimated_performance": tech_stack.estimated_performance
        }
        
        return config
    
    def _determine_region_type(self, req: RegionalRequirements) -> str:
        """Determine region type description."""
        infra_level = req.infrastructure.get_infrastructure_level()
        traffic_density = req.traffic.get_traffic_density()
        
        if infra_level == InfrastructureLevel.DEVELOPED:
            if traffic_density == TrafficDensity.VERY_HIGH:
                return "High-Density Urban - Developed"
            return "Urban - Developed"
        elif infra_level == InfrastructureLevel.DEVELOPING:
            if traffic_density == TrafficDensity.VERY_HIGH:
                return "High-Density Mixed Traffic - Developing"
            return "Mixed Traffic - Developing"
        else:
            return "Emerging Market - Low Infrastructure"
    
    def _adapt_traffic_parameters(self, config: Dict[str, Any], 
                                  req: RegionalRequirements) -> Dict[str, Any]:
        """Adapt traffic parameters based on requirements."""
        traffic_density = req.traffic.get_traffic_density()
        
        if traffic_density == TrafficDensity.VERY_HIGH:
            config["arrival_rates"] = [0.6, 0.55, 0.5, 0.45]
            config["queue_capacity"] = 60
        elif traffic_density == TrafficDensity.HIGH:
            config["arrival_rates"] = [0.5, 0.45, 0.4, 0.35]
            config["queue_capacity"] = 50
        elif traffic_density == TrafficDensity.MEDIUM:
            config["arrival_rates"] = [0.4, 0.35, 0.3, 0.25]
            config["queue_capacity"] = 40
        else:  # LOW
            config["arrival_rates"] = [0.2, 0.15, 0.18, 0.12]
            config["queue_capacity"] = 30
        
        # Adjust for vehicle mix
        if req.traffic.vehicle_mix_diverse:
            config["queue_capacity"] = int(config["queue_capacity"] * 1.2)
        
        return config
    
    def _adapt_signal_timing(self, config: Dict[str, Any], 
                           req: RegionalRequirements) -> Dict[str, Any]:
        """Adapt signal timing parameters."""
        traffic_density = req.traffic.get_traffic_density()
        
        if traffic_density == TrafficDensity.VERY_HIGH:
            config["min_green"] = 10
            config["max_green"] = 90
            config["cycle_yellow"] = 4
            config["cycle_all_red"] = 2
        elif traffic_density == TrafficDensity.HIGH:
            config["min_green"] = 8
            config["max_green"] = 75
            config["cycle_yellow"] = 4
            config["cycle_all_red"] = 2
        elif traffic_density == TrafficDensity.MEDIUM:
            config["min_green"] = 5
            config["max_green"] = 60
            config["cycle_yellow"] = 3
            config["cycle_all_red"] = 1
        else:  # LOW
            config["min_green"] = 5
            config["max_green"] = 45
            config["cycle_yellow"] = 3
            config["cycle_all_red"] = 1
        
        # Adjust for poor lane discipline
        if not req.traffic.lane_discipline_good:
            config["cycle_yellow"] += 1
            config["cycle_all_red"] += 1
        
        return config
    
    def _generate_vision_config(self, vision: VisionModel, 
                               req: RegionalRequirements) -> Dict[str, Any]:
        """Generate vision system configuration."""
        if vision == VisionModel.NONE:
            return {"enabled": False}
        
        config = {
            "enabled": True,
            "model": vision.value,
            "confidence_threshold": 0.4,
            "nms_threshold": 0.5
        }
        
        # Adjust for diverse vehicle mix
        if req.traffic.vehicle_mix_diverse:
            config["confidence_threshold"] = 0.35
            config["vehicle_types"] = [
                "car", "motorcycle", "auto-rickshaw", "bus", 
                "truck", "bicycle", "pedestrian"
            ]
        else:
            config["vehicle_types"] = ["car", "truck", "bus", "bicycle", "pedestrian"]
        
        # Adjust for camera quality
        if req.infrastructure.camera_quality == "720p":
            config["confidence_threshold"] = 0.3
        elif req.infrastructure.camera_quality == "4k":
            config["confidence_threshold"] = 0.5
        
        return config
    
    def _generate_control_config(self, control: ControlAlgorithm,
                                req: RegionalRequirements) -> Dict[str, Any]:
        """Generate control algorithm configuration."""
        config = {
            "algorithm": control.value,
            "enabled": True
        }
        
        if control == ControlAlgorithm.FUZZY_LOGIC:
            config["fuzzy_rules"] = {
                "queue_low": [0, 10],
                "queue_medium": [10, 30],
                "queue_high": [30, 100],
                "wait_low": [0, 15],
                "wait_medium": [15, 45],
                "wait_high": [45, 300]
            }
        elif control == ControlAlgorithm.DQN:
            config["training_episodes"] = 6000
            config["model_path"] = "models/dqn_agent.pth"
            config["requires_training"] = True
        elif control == ControlAlgorithm.MARL:
            config["num_agents"] = req.num_intersections
            config["coordination_enabled"] = True
            config["requires_training"] = True
        
        return config
    
    def _generate_forecasting_config(self, forecasting: ForecastingModel,
                                   req: RegionalRequirements) -> Dict[str, Any]:
        """Generate forecasting configuration."""
        if forecasting == ForecastingModel.NONE:
            return {"enabled": False}
        
        config = {
            "enabled": True,
            "model": forecasting.value,
            "input_timesteps": 10,
            "output_timesteps": 5
        }
        
        if forecasting == ForecastingModel.LSTM:
            config["hidden_units"] = 64
            config["num_layers"] = 2
        elif forecasting == ForecastingModel.GNN:
            config["num_nodes"] = req.num_intersections
            config["graph_structure"] = "auto"
        
        return config
    
    def _generate_deployment_config(self, deployment: DeploymentArchitecture,
                                   req: RegionalRequirements) -> Dict[str, Any]:
        """Generate deployment configuration."""
        config = {
            "architecture": deployment.value,
            "edge_computing": deployment == DeploymentArchitecture.EDGE or deployment == DeploymentArchitecture.HYBRID,
            "cloud_computing": deployment == DeploymentArchitecture.CLOUD or deployment == DeploymentArchitecture.HYBRID
        }
        
        if config["edge_computing"]:
            config["edge_device"] = {
                "cpu_cores": req.infrastructure.cpu_cores,
                "ram_gb": req.infrastructure.ram_gb,
                "gpu_available": req.infrastructure.gpu_available
            }
        
        if config["cloud_computing"]:
            config["cloud_config"] = {
                "network_bandwidth_mbps": req.infrastructure.network_bandwidth_mbps,
                "latency_requirement_ms": 100 if req.real_time_critical else 500
            }
        
        return config
    
    def _generate_regional_adaptations(self, req: RegionalRequirements) -> Dict[str, Any]:
        """Generate regional-specific adaptations."""
        return {
            "infrastructure_level": req.infrastructure.get_infrastructure_level().value,
            "traffic_density": req.traffic.get_traffic_density().value,
            "traffic_pattern": req.traffic.get_traffic_pattern().value,
            "vehicle_mix_diverse": req.traffic.vehicle_mix_diverse,
            "lane_discipline_good": req.traffic.lane_discipline_good,
            "special_considerations": []
        }
    
    def save_config(self, config: Dict[str, Any], output_path: str) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration dictionary
            output_path: Path to save configuration file
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Configuration saved to {output_path}")

