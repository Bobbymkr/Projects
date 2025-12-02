#!/usr/bin/env python3
"""
Analyze benchmark results and generate performance reports.

Usage:
    python scripts/analyze_benchmark_results.py --input results/benchmarks/benchmark_*.json
    python scripts/analyze_benchmark_results.py --compare baseline.json current.json
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import statistics

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def load_benchmark_results(file_path: Path) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(file_path) as f:
        return json.load(f)


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze benchmark results and generate statistics."""
    if not results.get("results"):
        return {}
    
    # Group by technology
    by_technology = {}
    by_scenario = {}
    
    for result in results["results"]:
        tech = result.get("technology")
        scenario = result.get("scenario")
        
        if tech not in by_technology:
            by_technology[tech] = []
        by_technology[tech].append(result)
        
        if scenario not in by_scenario:
            by_scenario[scenario] = []
        by_scenario[scenario].append(result)
    
    # Calculate statistics
    analysis = {
        "summary": {
            "total_combinations": len(results["results"]),
            "technologies_tested": len(by_technology),
            "scenarios_tested": len(by_scenario),
            "successful": sum(1 for r in results["results"] if r.get("errors", 0) == 0),
            "failed": sum(1 for r in results["results"] if r.get("errors", 0) > 0)
        },
        "by_technology": {},
        "by_scenario": {},
        "best_performers": []
    }
    
    # Analyze by technology
    for tech, tech_results in by_technology.items():
        successful = [r for r in tech_results if r.get("errors", 0) == 0 and "avg_wait_time" in r]
        
        if successful:
            wait_times = [r["avg_wait_time"] for r in successful]
            throughputs = [r.get("throughput_vehicles_per_hour", 0) for r in successful]
            latencies = [r.get("avg_inference_latency_ms", 0) for r in successful]
            
            analysis["by_technology"][tech] = {
                "avg_wait_time": statistics.mean(wait_times),
                "min_wait_time": min(wait_times),
                "max_wait_time": max(wait_times),
                "std_wait_time": statistics.stdev(wait_times) if len(wait_times) > 1 else 0,
                "avg_throughput": statistics.mean(throughputs),
                "avg_latency_ms": statistics.mean(latencies),
                "p95_latency_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0],
                "scenarios_tested": len(set(r["scenario"] for r in successful)),
                "episodes_completed": sum(r.get("episodes_completed", 0) for r in successful)
            }
    
    # Analyze by scenario
    for scenario, scen_results in by_scenario.items():
        successful = [r for r in scen_results if r.get("errors", 0) == 0 and "avg_wait_time" in r]
        
        if successful:
            wait_times = [r["avg_wait_time"] for r in successful]
            
            analysis["by_scenario"][scenario] = {
                "avg_wait_time": statistics.mean(wait_times),
                "min_wait_time": min(wait_times),
                "max_wait_time": max(wait_times),
                "technologies_tested": len(set(r["technology"] for r in successful)),
                "best_technology": min(successful, key=lambda x: x["avg_wait_time"])["technology"]
            }
    
    # Find best performers
    all_successful = [r for r in results["results"] if r.get("errors", 0) == 0 and "avg_wait_time" in r]
    if all_successful:
        sorted_by_wait = sorted(all_successful, key=lambda x: x["avg_wait_time"])
        analysis["best_performers"] = [
            {
                "technology": r["technology"],
                "scenario": r["scenario"],
                "avg_wait_time": r["avg_wait_time"],
                "throughput": r.get("throughput_vehicles_per_hour", 0)
            }
            for r in sorted_by_wait[:10]
        ]
    
    return analysis


def compare_results(baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two benchmark result sets."""
    baseline_analysis = analyze_results(baseline)
    current_analysis = analyze_results(current)
    
    comparison = {
        "baseline": baseline_analysis,
        "current": current_analysis,
        "improvements": {}
    }
    
    # Compare by technology
    for tech in current_analysis.get("by_technology", {}):
        if tech in baseline_analysis.get("by_technology", {}):
            baseline_metrics = baseline_analysis["by_technology"][tech]
            current_metrics = current_analysis["by_technology"][tech]
            
            wait_improvement = ((baseline_metrics["avg_wait_time"] - current_metrics["avg_wait_time"]) 
                              / baseline_metrics["avg_wait_time"] * 100)
            
            comparison["improvements"][tech] = {
                "wait_time_improvement_percent": wait_improvement,
                "baseline_wait_time": baseline_metrics["avg_wait_time"],
                "current_wait_time": current_metrics["avg_wait_time"],
                "throughput_improvement": current_metrics["avg_throughput"] - baseline_metrics.get("avg_throughput", 0),
                "latency_improvement_ms": baseline_metrics.get("avg_latency_ms", 0) - current_metrics.get("avg_latency_ms", 0)
            }
    
    return comparison


def generate_report(analysis: Dict[str, Any], output: Optional[Path] = None) -> str:
    """Generate markdown performance report."""
    report = f"""# Benchmark Performance Analysis Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 Executive Summary

- **Technologies Tested**: {analysis['summary']['technologies_tested']}
- **Scenarios Tested**: {analysis['summary']['scenarios_tested']}
- **Total Combinations**: {analysis['summary']['total_combinations']}
- **Successful**: {analysis['summary']['successful']}
- **Failed**: {analysis['summary']['failed']}

---

## 🏆 Best Performers (Top 10)

"""
    
    for i, performer in enumerate(analysis.get("best_performers", [])[:10], 1):
        report += f"{i}. **{performer['technology']}** on **{performer['scenario']}**\n"
        report += f"   - Wait Time: {performer['avg_wait_time']:.2f}s\n"
        report += f"   - Throughput: {performer['throughput']:.0f} veh/hr\n\n"
    
    report += "\n---\n\n## 📈 Performance by Technology\n\n"
    
    for tech, metrics in analysis.get("by_technology", {}).items():
        report += f"### {tech}\n\n"
        report += f"- **Average Wait Time**: {metrics['avg_wait_time']:.2f}s\n"
        report += f"- **Range**: {metrics['min_wait_time']:.2f}s - {metrics['max_wait_time']:.2f}s\n"
        report += f"- **Std Dev**: {metrics['std_wait_time']:.2f}s\n"
        report += f"- **Average Throughput**: {metrics['avg_throughput']:.0f} veh/hr\n"
        report += f"- **Average Latency**: {metrics['avg_latency_ms']:.3f}ms\n"
        report += f"- **P95 Latency**: {metrics['p95_latency_ms']:.3f}ms\n"
        report += f"- **Scenarios Tested**: {metrics['scenarios_tested']}\n"
        report += f"- **Episodes Completed**: {metrics['episodes_completed']}\n\n"
    
    report += "\n---\n\n## 🎯 Performance by Scenario\n\n"
    
    for scenario, metrics in analysis.get("by_scenario", {}).items():
        report += f"### {scenario}\n\n"
        report += f"- **Average Wait Time**: {metrics['avg_wait_time']:.2f}s\n"
        report += f"- **Range**: {metrics['min_wait_time']:.2f}s - {metrics['max_wait_time']:.2f}s\n"
        report += f"- **Best Technology**: {metrics['best_technology']}\n"
        report += f"- **Technologies Tested**: {metrics['technologies_tested']}\n\n"
    
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to {output}")
    else:
        print(report)
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--input", type=Path, help="Input benchmark JSON file")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BASELINE", "CURRENT"), help="Compare two result files")
    parser.add_argument("--output", type=Path, help="Output report file (markdown)")
    parser.add_argument("--json", type=Path, help="Output analysis JSON file")
    
    args = parser.parse_args()
    
    if args.compare:
        baseline = load_benchmark_results(args.compare[0])
        current = load_benchmark_results(args.compare[1])
        
        comparison = compare_results(baseline, current)
        
        if args.json:
            with open(args.json, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"Comparison saved to {args.json}")
        else:
            print("\n📊 Performance Comparison")
            print("=" * 80)
            for tech, improvement in comparison["improvements"].items():
                print(f"\n{tech}:")
                print(f"  Wait Time Improvement: {improvement['wait_time_improvement_percent']:+.2f}%")
                print(f"  Baseline: {improvement['baseline_wait_time']:.2f}s")
                print(f"  Current: {improvement['current_wait_time']:.2f}s")
    
    elif args.input:
        results = load_benchmark_results(args.input)
        analysis = analyze_results(results)
        
        if args.json:
            with open(args.json, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"Analysis saved to {args.json}")
        
        # Generate report
        report_output = args.output or (PROJECT_ROOT / "execution" / "reports" / f"benchmark_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        generate_report(analysis, report_output)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

