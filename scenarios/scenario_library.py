"""
Scenario Library for Comprehensive Benchmarking.

This module defines 10 standard traffic scenarios for benchmarking
all technologies across different traffic conditions.
"""

SCENARIOS = {
    "rush_hour": {
        "name": "Rush Hour (High Volume)",
        "description": "High traffic volume during peak hours",
        "duration": 3600,
        "traffic_pattern": "biased_incoming",
        "volume": "high",
        "weather": "clear"
    },
    "off_peak": {
        "name": "Off-Peak (Low Volume)",
        "description": "Low traffic volume during off-peak hours",
        "duration": 3600,
        "traffic_pattern": "uniform",
        "volume": "low",
        "weather": "clear"
    },
    "emergency_vehicle": {
        "name": "Emergency Vehicle Priority",
        "description": "Emergency vehicles requiring priority passage",
        "duration": 300,
        "emergency_frequency": 0.1,
        "priority": "absolute"
    },
    "accident_road_closure": {
        "name": "Accident/Road Closure",
        "description": "Lane closures due to accidents",
        "duration": 1800,
        "lane_closures": [
            2,
            3
        ],
        "detour_activation": True
    },
    "special_event": {
        "name": "Special Event Traffic",
        "description": "High traffic volume from special events",
        "duration": 7200,
        "volume": "extreme",
        "traffic_pattern": "convergent"
    },
    "multi_intersection": {
        "name": "Multi-Intersection Coordination",
        "description": "Coordinated control across multiple intersections",
        "duration": 3600,
        "num_intersections": 4,
        "coordination": True
    },
    "mixed_traffic": {
        "name": "Mixed Traffic (Cars/Buses/Bikes)",
        "description": "Diverse vehicle types with different behaviors",
        "duration": 3600,
        "vehicle_mix": {
            "cars": 0.7,
            "buses": 0.1,
            "trucks": 0.1,
            "motorcycles": 0.05,
            "bicycles": 0.05
        }
    },
    "weather_impacted": {
        "name": "Weather-Impacted Conditions",
        "description": "Adverse weather affecting traffic flow",
        "duration": 3600,
        "weather": "rain",
        "visibility": 0.6,
        "speed_reduction": 0.3
    },
    "construction_zone": {
        "name": "Construction Zone Routing",
        "description": "Lane closures due to construction",
        "duration": 28800,
        "lane_closures": [
            1
        ],
        "work_zone_speed": 25
    },
    "adaptive_timing": {
        "name": "Adaptive Signal Timing",
        "description": "Dynamic signal timing based on real-time conditions",
        "duration": 3600,
        "adaptive": True,
        "learning_rate": 0.1
    }
}

def get_scenario(name: str):
    """Get scenario configuration by name."""
    return SCENARIOS.get(name)

def list_scenarios():
    """List all available scenarios."""
    return list(SCENARIOS.keys())

def get_all_scenarios():
    """Get all scenario configurations."""
    return SCENARIOS
