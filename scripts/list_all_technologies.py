#!/usr/bin/env python3
"""
List all technologies and agents implemented in the codebase.

Usage:
    python scripts/list_all_technologies.py --output technologies.json
    python scripts/list_all_technologies.py --detailed
"""

import json
import argparse
import ast
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Set

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def find_agent_files() -> List[Path]:
    """Find all agent-related Python files."""
    agent_files = []
    
    # Search in src/rl/
    rl_dir = PROJECT_ROOT / "src" / "rl"
    if rl_dir.exists():
        agent_files.extend(rl_dir.glob("*agent*.py"))
        agent_files.extend(rl_dir.glob("dqn*.py"))
    
    # Search in src/research/novel_algorithms/
    research_dir = PROJECT_ROOT / "src" / "research" / "novel_algorithms"
    if research_dir.exists():
        agent_files.extend(research_dir.glob("*.py"))
        # Exclude __init__.py
        agent_files = [f for f in agent_files if f.name != "__init__.py"]
    
    # Search in src/control/
    control_dir = PROJECT_ROOT / "src" / "control"
    if control_dir.exists():
        agent_files.extend(control_dir.glob("*.py"))
    
    # Search in src/optimization/
    opt_dir = PROJECT_ROOT / "src" / "optimization"
    if opt_dir.exists():
        agent_files.extend(opt_dir.glob("*.py"))
    
    # Search in src/forecast/
    forecast_dir = PROJECT_ROOT / "src" / "forecast"
    if forecast_dir.exists():
        agent_files.extend(forecast_dir.glob("*.py"))
    
    return agent_files


def extract_class_names(file_path: Path) -> List[str]:
    """Extract class names from a Python file."""
    try:
        with open(file_path) as f:
            content = f.read()
        
        tree = ast.parse(content)
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        return classes
    except Exception:
        return []


def categorize_technology(file_path: Path, classes: List[str]) -> Dict[str, Any]:
    """Categorize a technology based on file path and class names."""
    file_name = file_path.stem.lower()
    file_path_str = str(file_path)
    
    tech_info = {
        "file": str(file_path.relative_to(PROJECT_ROOT)),
        "classes": classes,
        "category": "unknown",
        "technology": "unknown",
        "status": "implemented"
    }
    
    # Categorize based on file path and name
    if "model_based" in file_name or "ModelBased" in str(classes):
        tech_info["category"] = "reinforcement_learning"
        tech_info["technology"] = "model_based_rl"
    elif "hierarchical" in file_name or "Hierarchical" in str(classes):
        tech_info["category"] = "reinforcement_learning"
        tech_info["technology"] = "hierarchical_rl"
    elif "transformer" in file_name or "Transformer" in str(classes):
        tech_info["category"] = "reinforcement_learning"
        tech_info["technology"] = "transformer_agent"
    elif "dqn" in file_name or "DQN" in str(classes):
        tech_info["category"] = "reinforcement_learning"
        tech_info["technology"] = "dqn"
    elif "imitation" in file_name or "Imitation" in str(classes):
        tech_info["category"] = "reinforcement_learning"
        tech_info["technology"] = "imitation_learning"
    elif "bayesian" in file_name or "Bayesian" in str(classes):
        tech_info["category"] = "reinforcement_learning"
        tech_info["technology"] = "bayesian_rl"
    elif "causal" in file_name or "Causal" in str(classes):
        tech_info["category"] = "reinforcement_learning"
        tech_info["technology"] = "causal_rl"
    elif "neuro_symbolic" in file_name or "NeuroSymbolic" in str(classes):
        tech_info["category"] = "reinforcement_learning"
        tech_info["technology"] = "neuro_symbolic"
    elif "meta" in file_name or "Meta" in str(classes):
        tech_info["category"] = "reinforcement_learning"
        tech_info["technology"] = "meta_learning"
    elif "fuzzy" in file_name or "Fuzzy" in str(classes):
        tech_info["category"] = "classical_control"
        tech_info["technology"] = "fuzzy_logic"
    elif "webster" in file_name or "Webster" in str(classes):
        tech_info["category"] = "classical_control"
        tech_info["technology"] = "webster"
    elif "genetic" in file_name or "Genetic" in str(classes):
        tech_info["category"] = "optimization"
        tech_info["technology"] = "genetic_algorithm"
    elif "pso" in file_name or "PSO" in str(classes):
        tech_info["category"] = "optimization"
        tech_info["technology"] = "pso"
    elif "gnn" in file_name or "GNN" in str(classes):
        tech_info["category"] = "forecasting"
        tech_info["technology"] = "gnn_forecast"
    elif "lstm" in file_name or "LSTM" in str(classes) or "forecast" in file_name:
        tech_info["category"] = "forecasting"
        tech_info["technology"] = "lstm_forecast"
    elif "llm" in file_name or "LLM" in str(classes):
        tech_info["category"] = "experimental"
        tech_info["technology"] = "llm_agent"
    elif "diffusion" in file_name or "Diffusion" in str(classes):
        tech_info["category"] = "experimental"
        tech_info["technology"] = "diffusion_agent"
    
    return tech_info


def list_all_technologies(detailed: bool = False) -> Dict[str, Any]:
    """List all technologies in the codebase."""
    agent_files = find_agent_files()
    
    technologies = {}
    all_techs = set()
    
    for file_path in agent_files:
        classes = extract_class_names(file_path)
        tech_info = categorize_technology(file_path, classes)
        
        tech_name = tech_info["technology"]
        all_techs.add(tech_name)
        
        if tech_name not in technologies:
            technologies[tech_name] = {
                "technology": tech_name,
                "category": tech_info["category"],
                "files": [],
                "classes": set(),
                "status": "implemented"
            }
        
        technologies[tech_name]["files"].append(tech_info["file"])
        technologies[tech_name]["classes"].update(classes)
    
    # Convert sets to lists for JSON serialization
    for tech_name, tech_data in technologies.items():
        tech_data["classes"] = list(tech_data["classes"])
    
    # Count by category
    category_counts = {}
    for tech_data in technologies.values():
        category = tech_data["category"]
        category_counts[category] = category_counts.get(category, 0) + 1
    
    result = {
        "total_technologies": len(technologies),
        "categories": category_counts,
        "technologies": technologies,
        "technology_list": sorted(all_techs)
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="List all technologies in the codebase")
    parser.add_argument("--output", type=Path, help="Output JSON file path")
    parser.add_argument("--detailed", action="store_true", help="Show detailed information")
    
    args = parser.parse_args()
    
    result = list_all_technologies(args.detailed)
    
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Technology inventory saved to {args.output}")
    else:
        # Print to console
        print("\n📋 Technology Inventory")
        print("=" * 80)
        print(f"Total Technologies: {result['total_technologies']}")
        print("\nBy Category:")
        print("-" * 80)
        for category, count in result["categories"].items():
            print(f"  {category}: {count}")
        
        print("\nTechnologies:")
        print("-" * 80)
        for tech_name in sorted(result["technology_list"]):
            tech_data = result["technologies"][tech_name]
            print(f"  ✅ {tech_name} ({tech_data['category']})")
            if args.detailed:
                print(f"     Files: {len(tech_data['files'])}")
                print(f"     Classes: {', '.join(tech_data['classes'][:5])}")
                if len(tech_data['classes']) > 5:
                    print(f"     ... and {len(tech_data['classes']) - 5} more")
                print()


if __name__ == "__main__":
    main()

