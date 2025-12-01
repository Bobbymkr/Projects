"""Checklist Parser for Regional Adaptation.

Parses and analyzes regional adaptation checklist selections to extract
requirements and constraints for technology recommendation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class InfrastructureLevel(Enum):
    """Infrastructure quality levels."""
    DEVELOPED = "developed"
    DEVELOPING = "developing"
    EMERGING = "emerging"


class TrafficPattern(Enum):
    """Traffic pattern predictability."""
    HIGHLY_PREDICTABLE = "highly_predictable"
    MODERATELY_PREDICTABLE = "moderately_predictable"
    UNPREDICTABLE = "unpredictable"
    CHAOTIC = "chaotic"


class TrafficDensity(Enum):
    """Traffic density levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class InfrastructureAssessment:
    """Infrastructure assessment results."""
    power_availability_hours: float = 24.0
    power_outage_frequency: str = "rare"
    network_type: str = "fiber"
    network_bandwidth_mbps: float = 100.0
    network_reliability_percent: float = 99.9
    cpu_cores: int = 8
    ram_gb: int = 16
    gpu_available: bool = True
    camera_quality: str = "4k"
    existing_cctv: bool = True
    
    def get_infrastructure_level(self) -> InfrastructureLevel:
        """Determine infrastructure level based on assessment."""
        if (self.power_availability_hours >= 23.5 and 
            self.network_reliability_percent >= 99.0 and
            self.network_bandwidth_mbps >= 50 and
            self.gpu_available):
            return InfrastructureLevel.DEVELOPED
        elif (self.power_availability_hours >= 20.0 and
              self.network_reliability_percent >= 95.0):
            return InfrastructureLevel.DEVELOPING
        else:
            return InfrastructureLevel.EMERGING


@dataclass
class TrafficAssessment:
    """Traffic pattern assessment results."""
    peak_hours_defined: bool = True
    historical_data_months: int = 0
    vehicle_mix_diverse: bool = False
    lane_discipline_good: bool = True
    predictable_patterns: bool = True
    seasonal_variations: bool = False
    special_events_frequent: bool = False
    average_queue_length: float = 20.0
    max_queue_length: float = 40.0
    
    def get_traffic_density(self) -> TrafficDensity:
        """Determine traffic density level."""
        if self.average_queue_length >= 50:
            return TrafficDensity.VERY_HIGH
        elif self.average_queue_length >= 30:
            return TrafficDensity.HIGH
        elif self.average_queue_length >= 15:
            return TrafficDensity.MEDIUM
        else:
            return TrafficDensity.LOW
    
    def get_traffic_pattern(self) -> TrafficPattern:
        """Determine traffic pattern predictability."""
        if (self.historical_data_months >= 6 and
            self.predictable_patterns and
            not self.special_events_frequent):
            return TrafficPattern.HIGHLY_PREDICTABLE
        elif (self.historical_data_months >= 3 and
              self.predictable_patterns):
            return TrafficPattern.MODERATELY_PREDICTABLE
        elif self.special_events_frequent or not self.predictable_patterns:
            return TrafficPattern.UNPREDICTABLE
        else:
            return TrafficPattern.CHAOTIC


@dataclass
class RegionalRequirements:
    """Extracted regional requirements from checklist."""
    infrastructure: InfrastructureAssessment = field(default_factory=InfrastructureAssessment)
    traffic: TrafficAssessment = field(default_factory=TrafficAssessment)
    num_intersections: int = 1
    budget_per_intersection: float = 25000.0
    explainability_required: bool = False
    regulatory_compliance_strict: bool = False
    maintenance_team_available: bool = True
    ml_team_available: bool = False
    edge_deployment_preferred: bool = False
    multi_intersection_coordination: bool = False
    real_time_critical: bool = True
    custom_requirements: Dict[str, Any] = field(default_factory=dict)


class ChecklistParser:
    """Parser for regional adaptation checklist."""
    
    def __init__(self):
        """Initialize the checklist parser."""
        self.logger = logging.getLogger(__name__)
    
    def parse_from_dict(self, checklist_data: Dict[str, Any]) -> RegionalRequirements:
        """Parse checklist data from dictionary.
        
        Args:
            checklist_data: Dictionary containing checklist selections
            
        Returns:
            RegionalRequirements object with extracted requirements
        """
        requirements = RegionalRequirements()
        
        # Parse infrastructure assessment
        requirements.infrastructure = self._parse_infrastructure(checklist_data)
        
        # Parse traffic assessment
        requirements.traffic = self._parse_traffic(checklist_data)
        
        # Parse deployment requirements
        requirements.num_intersections = checklist_data.get('num_intersections', 1)
        requirements.budget_per_intersection = checklist_data.get('budget_per_intersection', 25000.0)
        requirements.explainability_required = checklist_data.get('explainability_required', False)
        requirements.regulatory_compliance_strict = checklist_data.get('regulatory_compliance_strict', False)
        requirements.maintenance_team_available = checklist_data.get('maintenance_team_available', True)
        requirements.ml_team_available = checklist_data.get('ml_team_available', False)
        requirements.edge_deployment_preferred = checklist_data.get('edge_deployment_preferred', False)
        requirements.multi_intersection_coordination = checklist_data.get('multi_intersection_coordination', False)
        requirements.real_time_critical = checklist_data.get('real_time_critical', True)
        requirements.custom_requirements = checklist_data.get('custom_requirements', {})
        
        return requirements
    
    def parse_from_file(self, checklist_path: str) -> RegionalRequirements:
        """Parse checklist from JSON file.
        
        Args:
            checklist_path: Path to checklist JSON file
            
        Returns:
            RegionalRequirements object
        """
        path = Path(checklist_path)
        if not path.exists():
            raise FileNotFoundError(f"Checklist file not found: {checklist_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            checklist_data = json.load(f)
        
        return self.parse_from_dict(checklist_data)
    
    def _parse_infrastructure(self, data: Dict[str, Any]) -> InfrastructureAssessment:
        """Parse infrastructure assessment section."""
        infra = data.get('infrastructure', {})
        power = infra.get('power', {})
        network = infra.get('network', {})
        hardware = infra.get('hardware', {})
        cameras = infra.get('cameras', {})
        
        return InfrastructureAssessment(
            power_availability_hours=power.get('availability_hours', 24.0),
            power_outage_frequency=power.get('outage_frequency', 'rare'),
            network_type=network.get('type', 'fiber'),
            network_bandwidth_mbps=network.get('bandwidth_mbps', 100.0),
            network_reliability_percent=network.get('reliability_percent', 99.9),
            cpu_cores=hardware.get('cpu_cores', 8),
            ram_gb=hardware.get('ram_gb', 16),
            gpu_available=hardware.get('gpu_available', True),
            camera_quality=cameras.get('quality', '4k'),
            existing_cctv=cameras.get('existing_cctv', True)
        )
    
    def _parse_traffic(self, data: Dict[str, Any]) -> TrafficAssessment:
        """Parse traffic assessment section."""
        traffic = data.get('traffic', {})
        patterns = traffic.get('patterns', {})
        volume = traffic.get('volume', {})
        
        return TrafficAssessment(
            peak_hours_defined=patterns.get('peak_hours_defined', True),
            historical_data_months=patterns.get('historical_data_months', 0),
            vehicle_mix_diverse=traffic.get('vehicle_mix_diverse', False),
            lane_discipline_good=traffic.get('lane_discipline_good', True),
            predictable_patterns=patterns.get('predictable', True),
            seasonal_variations=patterns.get('seasonal_variations', False),
            special_events_frequent=patterns.get('special_events_frequent', False),
            average_queue_length=volume.get('average_queue_length', 20.0),
            max_queue_length=volume.get('max_queue_length', 40.0)
        )
    
    def create_template(self) -> Dict[str, Any]:
        """Create a template checklist dictionary.
        
        Returns:
            Template checklist structure
        """
        return {
            "infrastructure": {
                "power": {
                    "availability_hours": 24.0,
                    "outage_frequency": "rare",  # rare, occasional, frequent
                    "backup_available": True
                },
                "network": {
                    "type": "fiber",  # fiber, 5g, 4g, dsl, satellite
                    "bandwidth_mbps": 100.0,
                    "reliability_percent": 99.9,
                    "latency_ms": 10.0
                },
                "hardware": {
                    "cpu_cores": 8,
                    "ram_gb": 16,
                    "gpu_available": True,
                    "storage_gb": 256
                },
                "cameras": {
                    "quality": "4k",  # 4k, 1080p, 720p
                    "existing_cctv": True,
                    "ip_cameras": True
                }
            },
            "traffic": {
                "patterns": {
                    "peak_hours_defined": True,
                    "historical_data_months": 0,
                    "predictable": True,
                    "seasonal_variations": False,
                    "special_events_frequent": False
                },
                "volume": {
                    "average_queue_length": 20.0,
                    "max_queue_length": 40.0,
                    "peak_hour_vehicles_per_hour": 1000
                },
                "vehicle_mix_diverse": False,
                "lane_discipline_good": True
            },
            "deployment": {
                "num_intersections": 1,
                "budget_per_intersection": 25000.0,
                "explainability_required": False,
                "regulatory_compliance_strict": False,
                "maintenance_team_available": True,
                "ml_team_available": False,
                "edge_deployment_preferred": False,
                "multi_intersection_coordination": False,
                "real_time_critical": True
            },
            "custom_requirements": {}
        }

