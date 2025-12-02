#!/usr/bin/env python3
"""
Track weekly progress metrics for the Perfect Score Execution Plan.

Usage:
    python scripts/track_progress.py --week 1 --output progress_week1.json
    python scripts/track_progress.py --week 1 --compare week0
"""

import json
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
PROGRESS_DIR = PROJECT_ROOT / "execution" / "progress"


def get_test_coverage() -> float:
    """Get current test coverage percentage."""
    try:
        result = subprocess.run(
            ["pytest", "--cov=src", "--cov-report=term-missing", "--quiet"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        # Parse coverage from output
        for line in result.stdout.split('\n'):
            if 'TOTAL' in line and '%' in line:
                # Extract percentage
                parts = line.split()
                for part in parts:
                    if '%' in part:
                        return float(part.replace('%', ''))
        return 0.0
    except Exception as e:
        print(f"Warning: Could not get test coverage: {e}", file=sys.stderr)
        return 0.0


def get_benchmark_status() -> Dict[str, Any]:
    """Get benchmark completion status."""
    benchmark_dir = PROJECT_ROOT / "results" / "benchmarks"
    if not benchmark_dir.exists():
        return {"total": 0, "completed": 0, "percentage": 0.0}
    
    # Count benchmark result files
    benchmark_files = list(benchmark_dir.glob("*.json"))
    # Expected: 13+ technologies × 10 scenarios = 130+ benchmarks
    expected = 130
    completed = len(benchmark_files)
    
    return {
        "total": expected,
        "completed": completed,
        "percentage": (completed / expected * 100) if expected > 0 else 0.0
    }


def get_performance_improvement() -> Optional[float]:
    """Get performance improvement percentage."""
    # Check for benchmark results
    benchmark_file = PROJECT_ROOT / "results" / "benchmark_report.json"
    if not benchmark_file.exists():
        return None
    
    try:
        with open(benchmark_file) as f:
            data = json.load(f)
        
        # Calculate improvement if baseline exists
        if "baseline" in data and "current" in data:
            baseline = data["baseline"].get("avg_wait_time", 0)
            current = data["current"].get("avg_wait_time", 0)
            if baseline > 0:
                improvement = ((baseline - current) / baseline) * 100
                return improvement
    except Exception:
        pass
    
    return None


def get_deployment_readiness() -> Dict[str, bool]:
    """Check deployment readiness components."""
    readiness = {
        "kubernetes_configs": (PROJECT_ROOT / "deployment" / "kubernetes").exists(),
        "docker_configs": (PROJECT_ROOT / "deployment" / "docker").exists(),
        "monitoring_setup": (PROJECT_ROOT / "src" / "monitoring").exists(),
        "ci_cd_pipeline": (PROJECT_ROOT / ".github" / "workflows").exists(),
        "health_checks": False,  # Would need to check actual health endpoints
    }
    
    # Check for health check endpoints
    api_main = PROJECT_ROOT / "src" / "api" / "main.py"
    if api_main.exists():
        with open(api_main) as f:
            content = f.read()
            readiness["health_checks"] = "/health" in content or "health" in content.lower()
    
    return readiness


def calculate_readiness_score(readiness: Dict[str, bool]) -> float:
    """Calculate deployment readiness score (0-100)."""
    total = len(readiness)
    completed = sum(1 for v in readiness.values() if v)
    return (completed / total * 100) if total > 0 else 0.0


def get_documentation_status() -> Dict[str, Any]:
    """Get documentation completion status."""
    docs_dir = PROJECT_ROOT / "docs"
    readme = PROJECT_ROOT / "README.md"
    
    # Count documentation files
    doc_files = []
    if docs_dir.exists():
        doc_files.extend(list(docs_dir.rglob("*.md")))
    if readme.exists():
        doc_files.append(readme)
    
    # Check for key documentation
    key_docs = {
        "api_documentation": (PROJECT_ROOT / "api" / "openapi.yml").exists(),
        "deployment_guide": any("deploy" in f.name.lower() for f in doc_files),
        "runbooks": any("runbook" in f.name.lower() for f in doc_files),
        "architecture_docs": any("architect" in f.name.lower() for f in doc_files),
    }
    
    return {
        "total_files": len(doc_files),
        "key_docs": key_docs,
        "completion": sum(1 for v in key_docs.values() if v) / len(key_docs) * 100
    }


def track_progress(week: int, output: Optional[Path] = None) -> Dict[str, Any]:
    """Track progress for a specific week."""
    print(f"Tracking progress for Week {week}...")
    
    # Get metrics
    test_coverage = get_test_coverage()
    benchmark_status = get_benchmark_status()
    performance_improvement = get_performance_improvement()
    deployment_readiness = get_deployment_readiness()
    readiness_score = calculate_readiness_score(deployment_readiness)
    documentation_status = get_documentation_status()
    
    progress = {
        "week": week,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "test_coverage": {
                "current": test_coverage,
                "target": 95.0,
                "status": "✅" if test_coverage >= 95 else "⚠️" if test_coverage >= 90 else "❌"
            },
            "benchmark_completion": {
                "completed": benchmark_status["completed"],
                "total": benchmark_status["total"],
                "percentage": benchmark_status["percentage"],
                "status": "✅" if benchmark_status["percentage"] >= 100 else "⚠️" if benchmark_status["percentage"] >= 80 else "❌"
            },
            "performance_improvement": {
                "improvement": performance_improvement,
                "target": 15.0,
                "status": "✅" if performance_improvement and performance_improvement >= 15 else "⚠️" if performance_improvement else "❌"
            },
            "deployment_readiness": {
                "score": readiness_score,
                "components": deployment_readiness,
                "status": "✅" if readiness_score >= 80 else "⚠️" if readiness_score >= 60 else "❌"
            },
            "documentation": {
                "completion": documentation_status["completion"],
                "total_files": documentation_status["total_files"],
                "key_docs": documentation_status["key_docs"],
                "status": "✅" if documentation_status["completion"] >= 80 else "⚠️" if documentation_status["completion"] >= 60 else "❌"
            }
        },
        "overall_score": {
            "current": 92.0,  # Baseline, would be calculated from metrics
            "target": 100.0,
            "progress": 0.0  # Would calculate based on week
        }
    }
    
    # Calculate overall progress
    metrics = progress["metrics"]
    scores = [
        metrics["test_coverage"]["current"] / 100 * 20,  # 20% weight
        metrics["benchmark_completion"]["percentage"] / 100 * 20,  # 20% weight
        (metrics["performance_improvement"]["improvement"] or 0) / 15 * 20 if metrics["performance_improvement"]["improvement"] else 0,  # 20% weight
        metrics["deployment_readiness"]["score"] / 100 * 20,  # 20% weight
        metrics["documentation"]["completion"] / 100 * 20,  # 20% weight
    ]
    overall_progress = sum(scores)
    progress["overall_score"]["progress"] = overall_progress
    
    # Save to file
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w') as f:
            json.dump(progress, f, indent=2)
        print(f"Progress saved to {output}")
    else:
        # Save to default location
        PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
        default_output = PROGRESS_DIR / f"progress_week{week}.json"
        with open(default_output, 'w') as f:
            json.dump(progress, f, indent=2)
        print(f"Progress saved to {default_output}")
    
    return progress


def compare_weeks(week1: int, week2: int):
    """Compare progress between two weeks."""
    file1 = PROGRESS_DIR / f"progress_week{week1}.json"
    file2 = PROGRESS_DIR / f"progress_week{week2}.json"
    
    if not file1.exists():
        print(f"Error: Progress file for week {week1} not found")
        return
    
    if not file2.exists():
        print(f"Error: Progress file for week {week2} not found")
        return
    
    with open(file1) as f:
        progress1 = json.load(f)
    with open(file2) as f:
        progress2 = json.load(f)
    
    print(f"\n📊 Progress Comparison: Week {week1} vs Week {week2}\n")
    print("=" * 60)
    
    for metric_name, metric1 in progress1["metrics"].items():
        metric2 = progress2["metrics"][metric_name]
        
        if "current" in metric1:
            val1 = metric1["current"]
            val2 = metric2["current"]
            change = val2 - val1
            print(f"{metric_name.replace('_', ' ').title()}:")
            print(f"  Week {week1}: {val1:.2f}")
            print(f"  Week {week2}: {val2:.2f}")
            print(f"  Change: {change:+.2f} {'✅' if change > 0 else '⚠️' if change == 0 else '❌'}")
        elif "percentage" in metric1:
            val1 = metric1["percentage"]
            val2 = metric2["percentage"]
            change = val2 - val1
            print(f"{metric_name.replace('_', ' ').title()}:")
            print(f"  Week {week1}: {val1:.2f}%")
            print(f"  Week {week2}: {val2:.2f}%")
            print(f"  Change: {change:+.2f}% {'✅' if change > 0 else '⚠️' if change == 0 else '❌'}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Track weekly progress metrics")
    parser.add_argument("--week", type=int, required=True, help="Week number (0-13)")
    parser.add_argument("--output", type=Path, help="Output file path")
    parser.add_argument("--compare", type=int, help="Compare with another week")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_weeks(args.week, args.compare)
    else:
        progress = track_progress(args.week, args.output)
        
        # Print summary
        print("\n📊 Progress Summary:")
        print("=" * 60)
        for metric_name, metric in progress["metrics"].items():
            status = metric.get("status", "❓")
            if "current" in metric:
                print(f"{status} {metric_name.replace('_', ' ').title()}: {metric['current']:.2f}% (target: {metric['target']:.2f}%)")
            elif "percentage" in metric:
                print(f"{status} {metric_name.replace('_', ' ').title()}: {metric['percentage']:.2f}% ({metric['completed']}/{metric['total']})")
            elif "score" in metric:
                print(f"{status} {metric_name.replace('_', ' ').title()}: {metric['score']:.2f}%")
        
        print(f"\n📈 Overall Progress: {progress['overall_score']['progress']:.2f}%")
        print("=" * 60)


if __name__ == "__main__":
    main()

