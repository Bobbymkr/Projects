"""
Comprehensive Scenario Library for Traffic Control.

Implements Phase 4.1 from OPTIMIZATION_ROADMAP.md:
- Temporal Patterns: Rush hour, night, weekend, holidays
- Weather Conditions: Rain, fog, snow (affect visibility & behavior)
- Event Scenarios: Accidents, construction, parades, sports events
- Traffic Types: Highway, urban, residential, mixed
- Network Topologies: Single intersection, arterial, grid, irregular

Expected Impact: 30-40% generalization improvement
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
from datetime import datetime, time
import logging

logger = logging.getLogger(__name__)


class TemporalPattern(Enum):
    """Temporal patterns."""
    RUSH_HOUR_MORNING = "rush_hour_morning"
    RUSH_HOUR_EVENING = "rush_hour_evening"
    NIGHT = "night"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"
    NORMAL = "normal"


class WeatherCondition(Enum):
    """Weather conditions."""
    CLEAR = "clear"
    RAIN = "rain"
    FOG = "fog"
    SNOW = "snow"
    HEAVY_RAIN = "heavy_rain"


class EventType(Enum):
    """Event types."""
    ACCIDENT = "accident"
    CONSTRUCTION = "construction"
    PARADE = "parade"
    SPORTS_EVENT = "sports_event"
    EMERGENCY = "emergency"
    NONE = "none"


class TrafficType(Enum):
    """Traffic types."""
    HIGHWAY = "highway"
    URBAN = "urban"
    RESIDENTIAL = "residential"
    MIXED = "mixed"


class NetworkTopology(Enum):
    """Network topologies."""
    SINGLE_INTERSECTION = "single_intersection"
    ARTERIAL = "arterial"
    GRID = "grid"
    IRREGULAR = "irregular"


@dataclass
class ScenarioConfig:
    """Configuration for a traffic scenario."""
    name: str
    description: str
    temporal_pattern: TemporalPattern = TemporalPattern.NORMAL
    weather: WeatherCondition = WeatherCondition.CLEAR
    event: EventType = EventType.NONE
    traffic_type: TrafficType = TrafficType.URBAN
    network_topology: NetworkTopology = NetworkTopology.SINGLE_INTERSECTION
    
    # Traffic parameters
    base_arrival_rates: List[float] = field(default_factory=lambda: [0.3, 0.25, 0.35, 0.2])
    traffic_multiplier: float = 1.0
    visibility_reduction: float = 0.0  # 0.0 = no reduction, 1.0 = complete reduction
    
    # Event parameters
    event_location: Optional[int] = None  # Lane index affected
    event_duration: int = 0  # Duration in seconds
    event_severity: float = 0.0  # 0.0 = no impact, 1.0 = complete blockage
    
    # Network parameters
    num_intersections: int = 1
    intersection_connections: List[Tuple[int, int]] = field(default_factory=list)
    
    # Additional parameters
    metadata: Dict[str, Any] = field(default_factory=dict)


class ScenarioLibrary:
    """
    Comprehensive scenario library for diverse traffic scenarios.
    """
    
    def __init__(self):
        """Initialize scenario library."""
        self.scenarios: Dict[str, ScenarioConfig] = {}
        self._initialize_scenarios()
    
    def _initialize_scenarios(self):
        """Initialize predefined scenarios."""
        
        # Temporal Patterns
        self._add_temporal_scenarios()
        
        # Weather Scenarios
        self._add_weather_scenarios()
        
        # Event Scenarios
        self._add_event_scenarios()
        
        # Traffic Type Scenarios
        self._add_traffic_type_scenarios()
        
        # Network Topology Scenarios
        self._add_network_topology_scenarios()
        
        # Combined Scenarios
        self._add_combined_scenarios()
    
    def _add_temporal_scenarios(self):
        """Add temporal pattern scenarios."""
        
        # Morning Rush Hour
        self.scenarios["morning_rush"] = ScenarioConfig(
            name="morning_rush",
            description="Morning rush hour (7-9 AM) with high traffic",
            temporal_pattern=TemporalPattern.RUSH_HOUR_MORNING,
            traffic_multiplier=1.8,
            base_arrival_rates=[0.5, 0.4, 0.6, 0.45],
            metadata={"time_range": (time(7, 0), time(9, 0))}
        )
        
        # Evening Rush Hour
        self.scenarios["evening_rush"] = ScenarioConfig(
            name="evening_rush",
            description="Evening rush hour (5-7 PM) with high traffic",
            temporal_pattern=TemporalPattern.RUSH_HOUR_EVENING,
            traffic_multiplier=1.9,
            base_arrival_rates=[0.6, 0.5, 0.55, 0.5],
            metadata={"time_range": (time(17, 0), time(19, 0))}
        )
        
        # Night
        self.scenarios["night"] = ScenarioConfig(
            name="night",
            description="Night time (10 PM - 6 AM) with low traffic",
            temporal_pattern=TemporalPattern.NIGHT,
            traffic_multiplier=0.3,
            base_arrival_rates=[0.1, 0.08, 0.12, 0.09],
            metadata={"time_range": (time(22, 0), time(6, 0))}
        )
        
        # Weekend
        self.scenarios["weekend"] = ScenarioConfig(
            name="weekend",
            description="Weekend traffic patterns",
            temporal_pattern=TemporalPattern.WEEKEND,
            traffic_multiplier=0.7,
            base_arrival_rates=[0.25, 0.2, 0.3, 0.22],
            metadata={"day_type": "weekend"}
        )
        
        # Holiday
        self.scenarios["holiday"] = ScenarioConfig(
            name="holiday",
            description="Holiday traffic patterns",
            temporal_pattern=TemporalPattern.HOLIDAY,
            traffic_multiplier=0.5,
            base_arrival_rates=[0.15, 0.12, 0.18, 0.14],
            metadata={"day_type": "holiday"}
        )
    
    def _add_weather_scenarios(self):
        """Add weather condition scenarios."""
        
        # Rain
        self.scenarios["rain"] = ScenarioConfig(
            name="rain",
            description="Rainy weather affecting visibility and behavior",
            weather=WeatherCondition.RAIN,
            visibility_reduction=0.2,
            traffic_multiplier=0.9,  # Slightly reduced traffic
            metadata={"speed_reduction": 0.1}
        )
        
        # Heavy Rain
        self.scenarios["heavy_rain"] = ScenarioConfig(
            name="heavy_rain",
            description="Heavy rain with significant visibility reduction",
            weather=WeatherCondition.HEAVY_RAIN,
            visibility_reduction=0.5,
            traffic_multiplier=0.7,
            metadata={"speed_reduction": 0.3}
        )
        
        # Fog
        self.scenarios["fog"] = ScenarioConfig(
            name="fog",
            description="Foggy conditions with reduced visibility",
            weather=WeatherCondition.FOG,
            visibility_reduction=0.6,
            traffic_multiplier=0.8,
            metadata={"speed_reduction": 0.2}
        )
        
        # Snow
        self.scenarios["snow"] = ScenarioConfig(
            name="snow",
            description="Snowy conditions with reduced visibility and speed",
            weather=WeatherCondition.SNOW,
            visibility_reduction=0.4,
            traffic_multiplier=0.6,
            metadata={"speed_reduction": 0.4}
        )
    
    def _add_event_scenarios(self):
        """Add event scenarios."""
        
        # Accident
        self.scenarios["accident"] = ScenarioConfig(
            name="accident",
            description="Traffic accident blocking one lane",
            event=EventType.ACCIDENT,
            event_location=0,  # Lane 0 blocked
            event_duration=1800,  # 30 minutes
            event_severity=0.8,  # 80% capacity reduction
            metadata={"emergency_vehicles": True}
        )
        
        # Construction
        self.scenarios["construction"] = ScenarioConfig(
            name="construction",
            description="Road construction affecting traffic flow",
            event=EventType.CONSTRUCTION,
            event_location=1,  # Lane 1 affected
            event_duration=7200,  # 2 hours
            event_severity=0.5,  # 50% capacity reduction
            metadata={"lane_closures": 1}
        )
        
        # Parade
        self.scenarios["parade"] = ScenarioConfig(
            name="parade",
            description="Parade blocking multiple lanes",
            event=EventType.PARADE,
            event_location=None,  # Multiple lanes
            event_duration=3600,  # 1 hour
            event_severity=1.0,  # Complete blockage
            metadata={"affected_lanes": [0, 1]}
        )
        
        # Sports Event
        self.scenarios["sports_event"] = ScenarioConfig(
            name="sports_event",
            description="Sports event causing increased traffic",
            event=EventType.SPORTS_EVENT,
            traffic_multiplier=1.5,
            metadata={"pre_event": True, "post_event": True}
        )
    
    def _add_traffic_type_scenarios(self):
        """Add traffic type scenarios."""
        
        # Highway
        self.scenarios["highway"] = ScenarioConfig(
            name="highway",
            description="Highway traffic with high speeds",
            traffic_type=TrafficType.HIGHWAY,
            base_arrival_rates=[0.4, 0.4, 0.4, 0.4],
            metadata={"speed_limit": 65, "vehicle_types": ["car", "truck"]}
        )
        
        # Urban
        self.scenarios["urban"] = ScenarioConfig(
            name="urban",
            description="Urban traffic with mixed vehicle types",
            traffic_type=TrafficType.URBAN,
            base_arrival_rates=[0.3, 0.25, 0.35, 0.2],
            metadata={"speed_limit": 35, "vehicle_types": ["car", "bus", "truck", "motorcycle"]}
        )
        
        # Residential
        self.scenarios["residential"] = ScenarioConfig(
            name="residential",
            description="Residential area with low speeds and pedestrians",
            traffic_type=TrafficType.RESIDENTIAL,
            base_arrival_rates=[0.15, 0.12, 0.18, 0.14],
            metadata={"speed_limit": 25, "pedestrians": True}
        )
        
        # Mixed
        self.scenarios["mixed"] = ScenarioConfig(
            name="mixed",
            description="Mixed traffic with diverse patterns",
            traffic_type=TrafficType.MIXED,
            base_arrival_rates=[0.3, 0.3, 0.3, 0.3],
            metadata={"speed_limit": 45, "vehicle_types": "all"}
        )
    
    def _add_network_topology_scenarios(self):
        """Add network topology scenarios."""
        
        # Single Intersection
        self.scenarios["single_intersection"] = ScenarioConfig(
            name="single_intersection",
            description="Single isolated intersection",
            network_topology=NetworkTopology.SINGLE_INTERSECTION,
            num_intersections=1,
            metadata={"isolation": True}
        )
        
        # Arterial
        self.scenarios["arterial"] = ScenarioConfig(
            name="arterial",
            description="Arterial road with multiple intersections",
            network_topology=NetworkTopology.ARTERIAL,
            num_intersections=5,
            intersection_connections=[(0, 1), (1, 2), (2, 3), (3, 4)],
            metadata={"linear": True}
        )
        
        # Grid
        self.scenarios["grid"] = ScenarioConfig(
            name="grid",
            description="Grid network with multiple intersections",
            network_topology=NetworkTopology.GRID,
            num_intersections=9,  # 3x3 grid
            intersection_connections=[
                (0, 1), (0, 3), (1, 2), (1, 4),
                (2, 5), (3, 4), (3, 6), (4, 5),
                (4, 7), (5, 8), (6, 7), (7, 8)
            ],
            metadata={"grid_size": (3, 3)}
        )
    
    def _add_combined_scenarios(self):
        """Add combined scenarios."""
        
        # Rush Hour + Rain
        self.scenarios["rush_hour_rain"] = ScenarioConfig(
            name="rush_hour_rain",
            description="Rush hour during rainy weather",
            temporal_pattern=TemporalPattern.RUSH_HOUR_EVENING,
            weather=WeatherCondition.RAIN,
            traffic_multiplier=1.6,
            visibility_reduction=0.2,
            base_arrival_rates=[0.5, 0.4, 0.55, 0.45]
        )
        
        # Weekend + Sports Event
        self.scenarios["weekend_sports"] = ScenarioConfig(
            name="weekend_sports",
            description="Weekend with sports event",
            temporal_pattern=TemporalPattern.WEEKEND,
            event=EventType.SPORTS_EVENT,
            traffic_multiplier=1.2,
            base_arrival_rates=[0.3, 0.25, 0.35, 0.3]
        )
        
        # Night + Fog
        self.scenarios["night_fog"] = ScenarioConfig(
            name="night_fog",
            description="Night time with foggy conditions",
            temporal_pattern=TemporalPattern.NIGHT,
            weather=WeatherCondition.FOG,
            traffic_multiplier=0.2,
            visibility_reduction=0.7,
            base_arrival_rates=[0.08, 0.06, 0.1, 0.07]
        )
    
    def get_scenario(self, name: str) -> Optional[ScenarioConfig]:
        """
        Get scenario by name.
        
        Args:
            name: Scenario name
            
        Returns:
            Scenario configuration or None
        """
        return self.scenarios.get(name)
    
    def list_scenarios(self) -> List[str]:
        """
        List all available scenarios.
        
        Returns:
            List of scenario names
        """
        return list(self.scenarios.keys())
    
    def get_scenarios_by_type(self, scenario_type: str) -> List[ScenarioConfig]:
        """
        Get scenarios by type.
        
        Args:
            scenario_type: Type of scenario ("temporal", "weather", "event", "traffic", "network", "combined")
            
        Returns:
            List of matching scenarios
        """
        if scenario_type == "temporal":
            return [s for s in self.scenarios.values() 
                   if s.temporal_pattern != TemporalPattern.NORMAL]
        elif scenario_type == "weather":
            return [s for s in self.scenarios.values() 
                   if s.weather != WeatherCondition.CLEAR]
        elif scenario_type == "event":
            return [s for s in self.scenarios.values() 
                   if s.event != EventType.NONE]
        elif scenario_type == "traffic":
            return [s for s in self.scenarios.values() 
                   if s.traffic_type != TrafficType.URBAN]
        elif scenario_type == "network":
            return [s for s in self.scenarios.values() 
                   if s.network_topology != NetworkTopology.SINGLE_INTERSECTION]
        elif scenario_type == "combined":
            return [s for s in self.scenarios.values() 
                   if (s.temporal_pattern != TemporalPattern.NORMAL and s.weather != WeatherCondition.CLEAR) or
                      (s.temporal_pattern != TemporalPattern.NORMAL and s.event != EventType.NONE) or
                      (s.weather != WeatherCondition.CLEAR and s.event != EventType.NONE)]
        else:
            return []
    
    def apply_scenario(self, config: Dict[str, Any], scenario: ScenarioConfig) -> Dict[str, Any]:
        """
        Apply scenario to environment configuration.
        
        Args:
            config: Base environment configuration
            scenario: Scenario configuration
            
        Returns:
            Modified configuration
        """
        new_config = config.copy()
        
        # Apply traffic multiplier
        if "arrival_rates" in new_config:
            base_rates = np.array(new_config["arrival_rates"])
            new_config["arrival_rates"] = (base_rates * scenario.traffic_multiplier).tolist()
        else:
            new_config["arrival_rates"] = (np.array(scenario.base_arrival_rates) * scenario.traffic_multiplier).tolist()
        
        # Apply event effects
        if scenario.event != EventType.NONE and scenario.event_location is not None:
            if "arrival_rates" in new_config:
                rates = np.array(new_config["arrival_rates"])
                rates[scenario.event_location] *= (1 - scenario.event_severity)
                new_config["arrival_rates"] = rates.tolist()
        
        # Add scenario metadata
        new_config["scenario"] = {
            "name": scenario.name,
            "description": scenario.description,
            "temporal_pattern": scenario.temporal_pattern.value,
            "weather": scenario.weather.value,
            "event": scenario.event.value,
            "traffic_type": scenario.traffic_type.value,
            "network_topology": scenario.network_topology.value,
            "visibility_reduction": scenario.visibility_reduction,
            "metadata": scenario.metadata
        }
        
        return new_config
    
    def sample_random_scenario(self) -> ScenarioConfig:
        """
        Sample a random scenario.
        
        Returns:
            Random scenario configuration
        """
        return random.choice(list(self.scenarios.values()))
    
    def create_custom_scenario(self, name: str, description: str, **kwargs) -> ScenarioConfig:
        """
        Create a custom scenario.
        
        Args:
            name: Scenario name
            description: Scenario description
            **kwargs: Additional parameters
            
        Returns:
            Custom scenario configuration
        """
        config = ScenarioConfig(
            name=name,
            description=description,
            **kwargs
        )
        self.scenarios[name] = config
        return config

