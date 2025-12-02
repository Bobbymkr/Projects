#!/usr/bin/env python3
"""
Performance Regression Detection Script.

Week 4: Automatically detect performance regressions in CI/CD.

Usage:
    python scripts/check_performance_regression.py --baseline baseline.json --current current.json --threshold 0.15
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def load_results(file_path: Path) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(file_path) as f:
        return json.load(f)


def calculate_regression(baseline: Dict[str, Any], current: Dict[str, Any], threshold: float = 0.15) -> List[Dict[str, Any]]:
    """Calculate performance regressions."""
    regressions = []
    
    baseline_results = baseline.get("results", [])
    current_results = current.get("results", [])
    
    # Group by technology and scenario
    baseline_by_key = {}
    for result in baseline_results:
        if result.get("errors", 0) == 0 and "avg_wait_time" in result:
            key = f"{result['technology']}_{result['scenario']}"
            baseline_by_key[key] = result
    
    current_by_key = {}
    for result in current_results:
        if result.get("errors", 0) == 0 and "avg_wait_time" in result:
            key = f"{result['technology']}_{result['scenario']}"
            current_by_key[key] = result
    
    # Compare
    for key in baseline_by_key:
        if key in current_by_key:
            baseline_result = baseline_by_key[key]
            current_result = current_by_key[key]
            
            baseline_wait = baseline_result.get("avg_wait_time", 0)
            current_wait = current_result.get("avg_wait_time", 0)
            
            if baseline_wait > 0:
                regression_percent = ((current_wait - baseline_wait) / baseline_wait) * 100
                
                if regression_percent > (threshold * 100):
                    regressions.append({
                        "technology": baseline_result["technology"],
                        "scenario": baseline_result["scenario"],
                        "baseline_wait_time": baseline_wait,
                        "current_wait_time": current_wait,
                        "regression_percent": regression_percent,
                        "threshold": threshold * 100
                    })
    
    return regressions


def main():
    parser = argparse.ArgumentParser(description="Check for performance regressions")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline benchmark results")
    parser.add_argument("--current", type=Path, required=True, help="Current benchmark results")
    parser.add_argument("--threshold", type=float, default=0.15, help="Regression threshold (15% default)")
    parser.add_argument("--output", type=Path, help="Output JSON file")
    
    args = parser.parse_args()
    
    # Load results
    baseline = load_results(args.baseline)
    current = load_results(args.current)
    
    # Calculate regressions
    regressions = calculate_regression(baseline, current, args.threshold)
    
    if regressions:
        print(f"\n⚠️ Performance Regressions Detected ({len(regressions)}):")
        print("=" * 80)
        for reg in regressions:
            print(f"  {reg['technology']} on {reg['scenario']}:")
            print(f"    Baseline: {reg['baseline_wait_time']:.2f}s")
            print(f"    Current: {reg['current_wait_time']:.2f}s")
            print(f"    Regression: +{reg['regression_percent']:.2f}% (threshold: {reg['threshold']:.1f}%)")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({"regressions": regressions}, f, indent=2)
        
        sys.exit(1)  # Fail CI/CD
    else:
        print("\n✅ No performance regressions detected")
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({"regressions": []}, f, indent=2)
        sys.exit(0)


if __name__ == "__main__":
    main()

