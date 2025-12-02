#!/usr/bin/env python3
"""
Set up the comprehensive benchmark framework.

Usage:
    python scripts/setup_benchmark_framework.py
    python scripts/setup_benchmark_framework.py --scenarios-only
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# 10 standard scenarios from execution plan
SCENARIOS = {
    "rush_hour": {
        "name": "Rush Hour (High Volume)",
        "description": "High traffic volume during peak hours",
        "duration": 3600,  # 1 hour
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
        "duration": 300,  # 5 min
        "emergency_frequency": 0.1,  # One every 10 episodes
        "priority": "absolute"
    },
    "accident_road_closure": {
        "name": "Accident/Road Closure",
        "description": "Lane closures due to accidents",
        "duration": 1800,  # 30 min
        "lane_closures": [2, 3],
        "detour_activation": True
    },
    "special_event": {
        "name": "Special Event Traffic",
        "description": "High traffic volume from special events",
        "duration": 7200,  # 2 hours
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
        "duration": 28800,  # 8 hours
        "lane_closures": [1],
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

# Expected technologies (from list_all_technologies.py)
EXPECTED_TECHNOLOGIES = [
    "model_based_rl",
    "hierarchical_rl",
    "transformer_agent",
    "dqn",
    "fuzzy_logic",
    "webster",
    "genetic_algorithm",
    "pso",
    "gnn_forecast",
    "lstm_forecast",
    "imitation_learning",
    "bayesian_rl",
    "causal_rl",
]


def create_scenario_library() -> Path:
    """Create scenario library structure."""
    scenarios_dir = PROJECT_ROOT / "scenarios"
    scenarios_dir.mkdir(exist_ok=True)
    
    # Create scenario library file
    scenario_library = scenarios_dir / "scenario_library.py"
    
    content = f'''"""
Scenario Library for Comprehensive Benchmarking.

This module defines 10 standard traffic scenarios for benchmarking
all technologies across different traffic conditions.
"""

SCENARIOS = {json.dumps(SCENARIOS, indent=4)}

def get_scenario(name: str):
    """Get scenario configuration by name."""
    return SCENARIOS.get(name)

def list_scenarios():
    """List all available scenarios."""
    return list(SCENARIOS.keys())

def get_all_scenarios():
    """Get all scenario configurations."""
    return SCENARIOS
'''
    
    with open(scenario_library, 'w') as f:
        f.write(content)
    
    print(f"✅ Created scenario library: {scenario_library}")
    return scenario_library


def create_benchmark_config() -> Path:
    """Create benchmark configuration file."""
    config_dir = PROJECT_ROOT / "configs" / "benchmarking"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = config_dir / "benchmark_config.json"
    
    config = {
        "scenarios": list(SCENARIOS.keys()),
        "technologies": EXPECTED_TECHNOLOGIES,
        "metrics": {
            "avg_wait_time": "Average vehicle wait time in seconds",
            "max_wait_time": "Maximum vehicle wait time in seconds",
            "95th_percentile_wait": "95th percentile wait time in seconds",
            "avg_queue_length": "Average queue length",
            "throughput_vehicles_per_hour": "Vehicles processed per hour",
            "convergence_time_seconds": "Time to convergence in seconds",
            "cpu_usage_percent": "CPU usage percentage",
            "memory_mb": "Memory usage in MB",
            "inference_latency_ms": "Inference latency in milliseconds",
            "adaptation_speed_episodes": "Episodes to adapt"
        },
        "default_episodes": 5000,
        "quick_episodes": 100,
        "statistical_significance": {
            "min_runs": 30,
            "confidence_level": 0.95
        },
        "output_dir": "results/benchmarks"
    }
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created benchmark config: {config_file}")
    return config_file


def create_results_structure():
    """Create results directory structure."""
    results_dir = PROJECT_ROOT / "results"
    benchmarks_dir = results_dir / "benchmarks"
    
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    
    # Create .gitkeep to preserve directory
    (benchmarks_dir / ".gitkeep").touch()
    
    print(f"✅ Created results structure: {benchmarks_dir}")


def create_benchmark_template() -> Path:
    """Create benchmark script template."""
    scripts_dir = PROJECT_ROOT / "scripts"
    template_file = scripts_dir / "benchmark_all_technologies.py.template"
    
    template_content = '''#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite for All Technologies.

This script benchmarks all 13+ technologies across 10 scenarios
as specified in the Perfect Score Execution Plan.
"""

import json
import argparse
from pathlib import Path
from scenarios.scenario_library import SCENARIOS, get_scenario

# TODO: Import all technologies
# from src.rl.dqn_agent import DQNAgent
# from src.control.fuzzy_control import FuzzyController
# ... etc

def benchmark_technology(technology: str, scenario: str, episodes: int = 5000):
    """Benchmark a single technology on a scenario."""
    # TODO: Implement benchmarking logic
    pass

def main():
    parser = argparse.ArgumentParser(description="Benchmark all technologies")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes")
    parser.add_argument("--scenarios", nargs="+", help="Specific scenarios (default: all)")
    parser.add_argument("--technologies", nargs="+", help="Specific technologies (default: all)")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (100 episodes)")
    parser.add_argument("--output", type=Path, help="Output file path")
    
    args = parser.parse_args()
    
    # TODO: Implement main benchmarking loop
    print("Benchmark framework ready. Implementation needed.")

if __name__ == "__main__":
    main()
'''
    
    with open(template_file, 'w') as f:
        f.write(template_content)
    
    print(f"✅ Created benchmark template: {template_file}")
    return template_file


def main():
    parser = argparse.ArgumentParser(description="Set up benchmark framework")
    parser.add_argument("--scenarios-only", action="store_true", help="Only create scenario library")
    
    args = parser.parse_args()
    
    print("\n🚀 Setting up Benchmark Framework")
    print("=" * 80)
    
    # Create scenario library
    create_scenario_library()
    
    if not args.scenarios_only:
        # Create benchmark config
        create_benchmark_config()
        
        # Create results structure
        create_results_structure()
        
        # Create benchmark template
        create_benchmark_template()
        
        print("\n✅ Benchmark framework setup complete!")
        print("\nNext steps:")
        print("  1. Review scenario_library.py")
        print("  2. Implement benchmark_all_technologies.py")
        print("  3. Run: python scripts/benchmark_all_technologies.py --episodes 5000")
    else:
        print("\n✅ Scenario library created!")


if __name__ == "__main__":
    main()

