#!/usr/bin/env python3
"""
Check benchmark completion status.

Usage:
    python scripts/check_benchmark_status.py --output benchmark_status.json
    python scripts/check_benchmark_status.py --detailed
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Expected technologies and scenarios
EXPECTED_TECHNOLOGIES = [
    "model_based_rl",
    "hierarchical_rl",
    "transformer",
    "dqn",
    "fuzzy_logic",
    "webster",
    "genetic_algorithm",
    "pso",
    "gnn",
    "lstm",
    "imitation_learning",
    "bayesian_rl",
    "causal_rl",
]

EXPECTED_SCENARIOS = [
    "rush_hour",
    "off_peak",
    "emergency_vehicle",
    "accident_road_closure",
    "special_event",
    "multi_intersection",
    "mixed_traffic",
    "weather_impacted",
    "construction_zone",
    "adaptive_timing",
]


def parse_benchmark_filename(filename: str) -> tuple:
    """Parse benchmark filename to extract technology and scenario."""
    # Expected format: technology_scenario_timestamp.json
    # or: benchmark_technology_scenario.json
    parts = filename.replace(".json", "").split("_")
    
    technology = None
    scenario = None
    
    for tech in EXPECTED_TECHNOLOGIES:
        if tech in filename.lower():
            technology = tech
            break
    
    for scen in EXPECTED_SCENARIOS:
        if scen in filename.lower():
            scenario = scen
            break
    
    return technology, scenario


def check_benchmark_status() -> Dict:
    """Check benchmark completion status."""
    benchmark_dir = PROJECT_ROOT / "results" / "benchmarks"
    
    if not benchmark_dir.exists():
        benchmark_dir.mkdir(parents=True, exist_ok=True)
    
    benchmark_files = list(benchmark_dir.glob("*.json"))
    
    # Track completed combinations
    completed = set()
    technology_status = {tech: {"completed": 0, "total": len(EXPECTED_SCENARIOS)} for tech in EXPECTED_TECHNOLOGIES}
    scenario_status = {scen: {"completed": 0, "total": len(EXPECTED_TECHNOLOGIES)} for scen in EXPECTED_SCENARIOS}
    
    for benchmark_file in benchmark_files:
        tech, scen = parse_benchmark_filename(benchmark_file.name)
        
        if tech and scen:
            completed.add((tech, scen))
            technology_status[tech]["completed"] += 1
            scenario_status[scen]["completed"] += 1
    
    # Calculate totals
    total_expected = len(EXPECTED_TECHNOLOGIES) * len(EXPECTED_SCENARIOS)
    total_completed = len(completed)
    completion_percentage = (total_completed / total_expected * 100) if total_expected > 0 else 0
    
    # Find missing combinations
    missing = []
    for tech in EXPECTED_TECHNOLOGIES:
        for scen in EXPECTED_SCENARIOS:
            if (tech, scen) not in completed:
                missing.append({"technology": tech, "scenario": scen})
    
    status = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_expected": total_expected,
            "total_completed": total_completed,
            "completion_percentage": completion_percentage,
            "status": "✅" if completion_percentage >= 100 else "⚠️" if completion_percentage >= 80 else "❌"
        },
        "by_technology": technology_status,
        "by_scenario": scenario_status,
        "missing": missing[:50],  # Limit to first 50
        "total_missing": len(missing)
    }
    
    return status


def main():
    parser = argparse.ArgumentParser(description="Check benchmark completion status")
    parser.add_argument("--output", type=Path, help="Output file path")
    parser.add_argument("--detailed", action="store_true", help="Show detailed breakdown")
    
    args = parser.parse_args()
    
    status = check_benchmark_status()
    
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(status, f, indent=2)
        print(f"Benchmark status saved to {args.output}")
    else:
        # Print to console
        print("\n📊 Benchmark Status")
        print("=" * 80)
        summary = status["summary"]
        print(f"Total Expected: {summary['total_expected']}")
        print(f"Total Completed: {summary['total_completed']}")
        print(f"Completion: {summary['completion_percentage']:.2f}%")
        print(f"Status: {summary['status']}")
        print(f"Missing: {status['total_missing']} combinations")
        
        if args.detailed:
            print("\nBy Technology:")
            print("-" * 80)
            for tech, tech_status in status["by_technology"].items():
                completed = tech_status["completed"]
                total = tech_status["total"]
                pct = (completed / total * 100) if total > 0 else 0
                status_icon = "✅" if completed == total else "⚠️" if completed >= total * 0.8 else "❌"
                print(f"{status_icon} {tech:25s}: {completed:2d}/{total:2d} ({pct:5.1f}%)")
            
            print("\nBy Scenario:")
            print("-" * 80)
            for scen, scen_status in status["by_scenario"].items():
                completed = scen_status["completed"]
                total = scen_status["total"]
                pct = (completed / total * 100) if total > 0 else 0
                status_icon = "✅" if completed == total else "⚠️" if completed >= total * 0.8 else "❌"
                print(f"{status_icon} {scen:25s}: {completed:2d}/{total:2d} ({pct:5.1f}%)")
            
            if status["missing"]:
                print("\nMissing Combinations (first 20):")
                print("-" * 80)
                for missing in status["missing"][:20]:
                    print(f"  - {missing['technology']} × {missing['scenario']}")


if __name__ == "__main__":
    main()

