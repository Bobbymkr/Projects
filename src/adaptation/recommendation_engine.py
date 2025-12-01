"""Intelligent Technology Recommendation Engine.

Automatically selects optimal technology stack based on regional
requirements extracted from checklist.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .checklist_parser import (
    RegionalRequirements,
    InfrastructureLevel,
    TrafficPattern,
    TrafficDensity
)

logger = logging.getLogger(__name__)


class ControlAlgorithm(Enum):
    """Available control algorithms."""
    FUZZY_LOGIC = "fuzzy_logic"
    DQN = "dqn"
    MARL = "marl"
    WEBSTER = "webster"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PSO = "pso"
    HYBRID = "hybrid"


class VisionModel(Enum):
    """Available vision models."""
    YOLOV8_NANO = "yolov8n"
    YOLOV8_SMALL = "yolov8s"
    YOLOV8_MEDIUM = "yolov8m"
    YOLOV8_LARGE = "yolov8l"
    NONE = "none"


class ForecastingModel(Enum):
    """Available forecasting models."""
    LSTM = "lstm"
    GNN = "gnn"
    CNN_LSTM = "cnn_lstm"
    NONE = "none"


class DeploymentArchitecture(Enum):
    """Deployment architecture options."""
    EDGE = "edge"
    CLOUD = "cloud"
    HYBRID = "hybrid"


@dataclass
class TechnologyStack:
    """Recommended technology stack."""
    control_algorithm: ControlAlgorithm
    vision_model: VisionModel
    forecasting_model: ForecastingModel
    deployment_architecture: DeploymentArchitecture
    confidence_score: float = 0.0
    reasoning: List[str] = field(default_factory=list)
    estimated_cost: float = 0.0
    estimated_performance: Dict[str, float] = field(default_factory=dict)
    alternatives: List['TechnologyStack'] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "control_algorithm": self.control_algorithm.value,
            "vision_model": self.vision_model.value,
            "forecasting_model": self.forecasting_model.value,
            "deployment_architecture": self.deployment_architecture.value,
            "confidence_score": self.confidence_score,
            "reasoning": self.reasoning,
            "estimated_cost": self.estimated_cost,
            "estimated_performance": self.estimated_performance,
            "alternatives": [alt.to_dict() for alt in self.alternatives]
        }


class TechnologyRecommendationEngine:
    """Intelligent engine for technology recommendation."""
    
    def __init__(self):
        """Initialize the recommendation engine."""
        self.logger = logging.getLogger(__name__)
        self._setup_decision_rules()
    
    def _setup_decision_rules(self):
        """Setup decision rules and cost/performance estimates."""
        # Cost estimates per intersection (USD)
        self.cost_estimates = {
            ControlAlgorithm.FUZZY_LOGIC: 2000,
            ControlAlgorithm.DQN: 15000,
            ControlAlgorithm.MARL: 25000,
            ControlAlgorithm.WEBSTER: 1000,
            ControlAlgorithm.GENETIC_ALGORITHM: 5000,
            ControlAlgorithm.PSO: 5000,
            
            VisionModel.YOLOV8_NANO: 3000,
            VisionModel.YOLOV8_SMALL: 5000,
            VisionModel.YOLOV8_MEDIUM: 8000,
            VisionModel.YOLOV8_LARGE: 12000,
            
            ForecastingModel.LSTM: 5000,
            ForecastingModel.GNN: 10000,
            ForecastingModel.CNN_LSTM: 7000,
            
            DeploymentArchitecture.EDGE: 5000,
            DeploymentArchitecture.CLOUD: 10000,
            DeploymentArchitecture.HYBRID: 15000,
        }
        
        # Performance estimates (wait time in seconds)
        self.performance_estimates = {
            ControlAlgorithm.FUZZY_LOGIC: 8.5,
            ControlAlgorithm.DQN: 21.5,
            ControlAlgorithm.MARL: 15.0,
            ControlAlgorithm.WEBSTER: 27.4,
            ControlAlgorithm.GENETIC_ALGORITHM: 12.0,
            ControlAlgorithm.PSO: 12.0,
        }
    
    def recommend(self, requirements: RegionalRequirements) -> TechnologyStack:
        """Generate technology recommendation based on requirements.
        
        Args:
            requirements: Regional requirements from checklist
            
        Returns:
            Recommended technology stack with reasoning
        """
        self.logger.info("Generating technology recommendation...")
        
        # Determine control algorithm
        control_algo = self._recommend_control_algorithm(requirements)
        
        # Determine vision model
        vision_model = self._recommend_vision_model(requirements)
        
        # Determine forecasting model
        forecasting_model = self._recommend_forecasting_model(requirements)
        
        # Determine deployment architecture
        deployment = self._recommend_deployment(requirements)
        
        # Calculate confidence and reasoning
        confidence, reasoning = self._calculate_confidence(
            requirements, control_algo, vision_model, forecasting_model, deployment
        )
        
        # Calculate cost
        cost = self._calculate_cost(control_algo, vision_model, forecasting_model, deployment)
        
        # Estimate performance
        performance = self._estimate_performance(control_algo, vision_model, forecasting_model)
        
        # Generate alternatives
        alternatives = self._generate_alternatives(requirements, control_algo, vision_model, forecasting_model, deployment)
        
        stack = TechnologyStack(
            control_algorithm=control_algo,
            vision_model=vision_model,
            forecasting_model=forecasting_model,
            deployment_architecture=deployment,
            confidence_score=confidence,
            reasoning=reasoning,
            estimated_cost=cost,
            estimated_performance=performance,
            alternatives=alternatives
        )
        
        self.logger.info(f"Recommended: {control_algo.value} + {vision_model.value} + {forecasting_model.value}")
        self.logger.info(f"Confidence: {confidence:.2%}, Cost: ${cost:,.0f}")
        
        return stack
    
    def _recommend_control_algorithm(self, req: RegionalRequirements) -> ControlAlgorithm:
        """Recommend control algorithm based on requirements."""
        infra_level = req.infrastructure.get_infrastructure_level()
        traffic_pattern = req.traffic.get_traffic_pattern()
        traffic_density = req.traffic.get_traffic_density()
        
        # Decision tree logic
        if infra_level == InfrastructureLevel.EMERGING:
            return ControlAlgorithm.FUZZY_LOGIC
        
        if traffic_pattern == TrafficPattern.CHAOTIC or traffic_pattern == TrafficPattern.UNPREDICTABLE:
            return ControlAlgorithm.FUZZY_LOGIC
        
        if req.traffic.historical_data_months < 6:
            return ControlAlgorithm.FUZZY_LOGIC
        
        if req.num_intersections == 1:
            if req.explainability_required:
                return ControlAlgorithm.FUZZY_LOGIC
            return ControlAlgorithm.FUZZY_LOGIC  # Best performance
        
        if req.num_intersections < 5:
            return ControlAlgorithm.FUZZY_LOGIC
        
        if req.num_intersections >= 50 and req.ml_team_available:
            if req.multi_intersection_coordination:
                return ControlAlgorithm.MARL
            return ControlAlgorithm.DQN
        
        if req.num_intersections >= 20 and req.ml_team_available:
            return ControlAlgorithm.DQN
        
        # Default: Fuzzy Logic (best performance, lowest complexity)
        return ControlAlgorithm.FUZZY_LOGIC
    
    def _recommend_vision_model(self, req: RegionalRequirements) -> VisionModel:
        """Recommend vision model based on requirements."""
        infra_level = req.infrastructure.get_infrastructure_level()
        traffic_density = req.traffic.get_traffic_density()
        gpu_available = req.infrastructure.gpu_available
        
        # No vision needed if cameras not available
        if not req.infrastructure.existing_cctv and req.infrastructure.camera_quality == "none":
            return VisionModel.NONE
        
        # Edge deployment or low infrastructure
        if infra_level == InfrastructureLevel.EMERGING or req.edge_deployment_preferred:
            return VisionModel.YOLOV8_NANO
        
        # Low traffic or limited resources
        if traffic_density == TrafficDensity.LOW and not gpu_available:
            return VisionModel.YOLOV8_NANO
        
        # Medium traffic or balanced requirements
        if traffic_density in [TrafficDensity.MEDIUM, TrafficDensity.HIGH]:
            if gpu_available:
                return VisionModel.YOLOV8_SMALL
            return VisionModel.YOLOV8_NANO
        
        # Very high traffic or high accuracy needed
        if traffic_density == TrafficDensity.VERY_HIGH:
            if gpu_available:
                return VisionModel.YOLOV8_MEDIUM
            return VisionModel.YOLOV8_SMALL
        
        # Default
        return VisionModel.YOLOV8_SMALL
    
    def _recommend_forecasting_model(self, req: RegionalRequirements) -> ForecastingModel:
        """Recommend forecasting model based on requirements."""
        traffic_pattern = req.traffic.get_traffic_pattern()
        
        # No forecasting for single intersection
        if req.num_intersections == 1:
            return ForecastingModel.NONE
        
        # No historical data
        if req.traffic.historical_data_months < 6:
            return ForecastingModel.NONE
        
        # Unpredictable traffic
        if traffic_pattern in [TrafficPattern.UNPREDICTABLE, TrafficPattern.CHAOTIC]:
            return ForecastingModel.NONE
        
        # Multi-intersection coordination
        if req.multi_intersection_coordination and req.num_intersections >= 5:
            if req.num_intersections >= 20 and req.ml_team_available:
                return ForecastingModel.GNN
            return ForecastingModel.LSTM
        
        # Optional forecasting for stable patterns
        if traffic_pattern == TrafficPattern.HIGHLY_PREDICTABLE:
            return ForecastingModel.LSTM
        
        return ForecastingModel.NONE
    
    def _recommend_deployment(self, req: RegionalRequirements) -> DeploymentArchitecture:
        """Recommend deployment architecture."""
        network_reliability = req.infrastructure.network_reliability_percent
        
        # Low network reliability -> edge
        if network_reliability < 95.0:
            return DeploymentArchitecture.EDGE
        
        # Explicit preference
        if req.edge_deployment_preferred:
            return DeploymentArchitecture.EDGE
        
        # Single or few intersections
        if req.num_intersections <= 10:
            return DeploymentArchitecture.EDGE
        
        # Real-time critical
        if req.real_time_critical and network_reliability < 99.0:
            return DeploymentArchitecture.EDGE
        
        # Multi-intersection coordination
        if req.num_intersections >= 10 and network_reliability >= 99.0:
            return DeploymentArchitecture.HYBRID
        
        # High reliability cloud
        if network_reliability >= 99.5:
            return DeploymentArchitecture.CLOUD
        
        return DeploymentArchitecture.EDGE
    
    def _calculate_confidence(self, req: RegionalRequirements, control: ControlAlgorithm,
                             vision: VisionModel, forecasting: ForecastingModel,
                             deployment: DeploymentArchitecture) -> Tuple[float, List[str]]:
        """Calculate confidence score and generate reasoning."""
        confidence = 0.8  # Base confidence
        reasoning = []
        
        # Control algorithm confidence
        if control == ControlAlgorithm.FUZZY_LOGIC:
            confidence += 0.1
            reasoning.append("Fuzzy Logic is the most reliable and proven control method")
        elif control == ControlAlgorithm.DQN and req.ml_team_available:
            confidence += 0.05
            reasoning.append("DQN selected with ML team support available")
        elif control == ControlAlgorithm.DQN and not req.ml_team_available:
            confidence -= 0.1
            reasoning.append("Warning: DQN requires ML expertise for maintenance")
        
        # Vision model confidence
        if vision != VisionModel.NONE and req.infrastructure.existing_cctv:
            confidence += 0.05
            reasoning.append("Existing CCTV infrastructure supports vision deployment")
        
        # Forecasting confidence
        if forecasting != ForecastingModel.NONE and req.traffic.historical_data_months >= 6:
            confidence += 0.05
            reasoning.append(f"Sufficient historical data ({req.traffic.historical_data_months} months) for forecasting")
        elif forecasting != ForecastingModel.NONE and req.traffic.historical_data_months < 6:
            confidence -= 0.1
            reasoning.append("Warning: Limited historical data may affect forecasting accuracy")
        
        # Deployment confidence
        if deployment == DeploymentArchitecture.EDGE and req.infrastructure.network_reliability_percent < 95:
            confidence += 0.05
            reasoning.append("Edge deployment recommended due to network reliability concerns")
        
        # Budget confidence
        total_cost = self._calculate_cost(control, vision, forecasting, deployment)
        if total_cost <= req.budget_per_intersection:
            confidence += 0.05
            reasoning.append(f"Recommended stack fits within budget (${total_cost:,.0f} <= ${req.budget_per_intersection:,.0f})")
        else:
            confidence -= 0.15
            reasoning.append(f"Warning: Recommended stack exceeds budget (${total_cost:,.0f} > ${req.budget_per_intersection:,.0f})")
        
        confidence = max(0.0, min(1.0, confidence))
        return confidence, reasoning
    
    def _calculate_cost(self, control: ControlAlgorithm, vision: VisionModel,
                       forecasting: ForecastingModel, deployment: DeploymentArchitecture) -> float:
        """Calculate total estimated cost."""
        base_cost = 5000  # Base infrastructure cost
        
        cost = (base_cost +
                self.cost_estimates.get(control, 0) +
                self.cost_estimates.get(vision, 0) +
                self.cost_estimates.get(forecasting, 0) +
                self.cost_estimates.get(deployment, 0))
        
        return cost
    
    def _estimate_performance(self, control: ControlAlgorithm, vision: VisionModel,
                            forecasting: ForecastingModel) -> Dict[str, float]:
        """Estimate performance metrics."""
        base_wait_time = self.performance_estimates.get(control, 20.0)
        
        # Vision improves accuracy
        if vision != VisionModel.NONE:
            base_wait_time *= 0.9
        
        # Forecasting improves coordination
        if forecasting != ForecastingModel.NONE:
            base_wait_time *= 0.95
        
        return {
            "estimated_wait_time_seconds": base_wait_time,
            "estimated_queue_length": base_wait_time * 1.2,
            "estimated_throughput_improvement_percent": max(0, (27.4 - base_wait_time) / 27.4 * 100)
        }
    
    def _generate_alternatives(self, req: RegionalRequirements, primary_control: ControlAlgorithm,
                              primary_vision: VisionModel, primary_forecasting: ForecastingModel,
                              primary_deployment: DeploymentArchitecture) -> List[TechnologyStack]:
        """Generate alternative technology stacks."""
        alternatives = []
        
        # Alternative 1: More cost-effective
        if primary_control != ControlAlgorithm.FUZZY_LOGIC:
            alt_control = ControlAlgorithm.FUZZY_LOGIC
            alt_vision = VisionModel.YOLOV8_NANO if primary_vision != VisionModel.NONE else VisionModel.NONE
            alt_forecasting = ForecastingModel.NONE
            alt_deployment = DeploymentArchitecture.EDGE
            
            cost = self._calculate_cost(alt_control, alt_vision, alt_forecasting, alt_deployment)
            if cost < self._calculate_cost(primary_control, primary_vision, primary_forecasting, primary_deployment):
                _, reasoning = self._calculate_confidence(req, alt_control, alt_vision, alt_forecasting, alt_deployment)
                alternatives.append(TechnologyStack(
                    control_algorithm=alt_control,
                    vision_model=alt_vision,
                    forecasting_model=alt_forecasting,
                    deployment_architecture=alt_deployment,
                    confidence_score=0.7,
                    reasoning=["Cost-effective alternative"] + reasoning,
                    estimated_cost=cost,
                    estimated_performance=self._estimate_performance(alt_control, alt_vision, alt_forecasting)
                ))
        
        # Alternative 2: Higher performance (if budget allows)
        if primary_control == ControlAlgorithm.FUZZY_LOGIC and req.ml_team_available:
            alt_control = ControlAlgorithm.DQN
            alt_vision = VisionModel.YOLOV8_MEDIUM if primary_vision != VisionModel.NONE else VisionModel.YOLOV8_SMALL
            alt_forecasting = ForecastingModel.LSTM if req.num_intersections >= 5 else ForecastingModel.NONE
            alt_deployment = primary_deployment
            
            cost = self._calculate_cost(alt_control, alt_vision, alt_forecasting, alt_deployment)
            if cost <= req.budget_per_intersection * 1.5:
                _, reasoning = self._calculate_confidence(req, alt_control, alt_vision, alt_forecasting, alt_deployment)
                alternatives.append(TechnologyStack(
                    control_algorithm=alt_control,
                    vision_model=alt_vision,
                    forecasting_model=alt_forecasting,
                    deployment_architecture=alt_deployment,
                    confidence_score=0.6,
                    reasoning=["Higher performance alternative (requires ML team)"] + reasoning,
                    estimated_cost=cost,
                    estimated_performance=self._estimate_performance(alt_control, alt_vision, alt_forecasting)
                ))
        
        return alternatives

