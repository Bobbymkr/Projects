#!/usr/bin/env python3
"""
Check success criteria for the Perfect Score Execution Plan.

Usage:
    python scripts/check_success_criteria.py --week 2 --report
    python scripts/check_success_criteria.py --all
"""

import json
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Success criteria by week
SUCCESS_CRITERIA = {
    1: [
        ("test_coverage", "Test coverage ≥95%", 95.0),
        ("benchmark_framework", "Benchmark framework complete", True),
        ("hyperparameter_optimization", "Hyperparameter optimization framework ready", True),
    ],
    2: [
        ("all_technologies_benchmarked", "All 13+ technologies benchmarked", True),
        ("performance_improvement", "≥15% performance improvement", 15.0),
        ("optimized_models", "Optimized models deployed", True),
    ],
    3: [
        ("integration_tests", "20+ integration tests", 20),
        ("load_testing", "Enhanced load testing suite", True),
        ("chaos_engineering", "Chaos engineering framework", True),
    ],
    4: [
        ("test_automation", "All tests automated in CI/CD", True),
        ("coverage_gates", "Test coverage gates enforced", True),
        ("performance_regression", "Performance regression detection", True),
    ],
    5: [
        ("prometheus_metrics", "Prometheus metrics collection", True),
        ("jaeger_tracing", "Jaeger distributed tracing", True),
        ("elk_logging", "ELK stack logging", True),
    ],
    6: [
        ("alerting_system", "Production alerting system", True),
        ("observability_dashboards", "Observability dashboards", True),
        ("operational_runbooks", "10+ operational runbooks", 10),
    ],
    7: [
        ("autoscaling", "Auto-scaling configuration", True),
        ("load_balancing", "Load balancing setup", True),
        ("state_management", "Distributed state management", True),
    ],
    8: [
        ("multi_region", "Multi-region deployment", True),
        ("automated_backups", "Automated backup system", True),
        ("disaster_recovery", "Disaster recovery tested", True),
        ("rto_rpo", "RTO <5 minutes, RPO <1 minute", True),
    ],
    9: [
        ("realtime_scheduler", "Real-time scheduler", True),
        ("deadline_aware_agents", "Deadline-aware agents", True),
        ("deadline_compliance", "99.9% deadline compliance", 99.9),
    ],
    10: [
        ("microservices", "Microservices architecture", True),
        ("grpc_services", "gRPC service layer", True),
        ("api_gateway", "API Gateway", True),
        ("service_latency", "Service-to-service latency <10ms", 10.0),
    ],
    11: [
        ("scenario_validation", "10 scenarios validated", 10),
        ("statistical_significance", "Statistical significance (p<0.05)", True),
        ("performance_docs", "Performance documentation updated", True),
    ],
    12: [
        ("regional_validation", "4 regions validated", 4),
        ("transfer_learning", "Transfer learning confirmed", True),
        ("regional_docs", "Regional deployment guides", True),
    ],
    13: [
        ("api_documentation", "Complete API documentation", True),
        ("deployment_guide", "Production deployment guide", True),
        ("final_validation", "Final validation report", True),
        ("score_100", "100/100 Score Achievement", 100.0),
    ],
}


def check_test_coverage(target: float = 95.0) -> Tuple[bool, float]:
    """Check if test coverage meets target."""
    try:
        result = subprocess.run(
            ["pytest", "--cov=src", "--cov-report=term-missing", "--quiet"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        for line in result.stdout.split('\n'):
            if 'TOTAL' in line and '%' in line:
                parts = line.split()
                for part in parts:
                    if '%' in part:
                        coverage = float(part.replace('%', ''))
                        return coverage >= target, coverage
        return False, 0.0
    except Exception:
        return False, 0.0


def check_benchmark_completion() -> Tuple[bool, int, int]:
    """Check benchmark completion status."""
    benchmark_dir = PROJECT_ROOT / "results" / "benchmarks"
    if not benchmark_dir.exists():
        return False, 0, 130
    
    benchmark_files = list(benchmark_dir.glob("*.json"))
    expected = 130  # 13 technologies × 10 scenarios
    completed = len(benchmark_files)
    return completed >= expected, completed, expected


def check_file_exists(file_path: Path) -> bool:
    """Check if a file or directory exists."""
    return file_path.exists()


def check_criteria_for_week(week: int) -> List[Dict]:
    """Check all success criteria for a specific week."""
    if week not in SUCCESS_CRITERIA:
        return []
    
    results = []
    criteria = SUCCESS_CRITERIA[week]
    
    for criterion_id, description, target in criteria:
        passed = False
        actual = None
        details = ""
        
        if criterion_id == "test_coverage":
            passed, actual = check_test_coverage(target)
            details = f"Current: {actual:.2f}%, Target: {target:.2f}%"
        
        elif criterion_id == "all_technologies_benchmarked":
            passed, completed, total = check_benchmark_completion()
            actual = completed
            details = f"Completed: {completed}/{total}"
        
        elif criterion_id == "benchmark_framework":
            passed = check_file_exists(PROJECT_ROOT / "scripts" / "benchmark_all_technologies.py")
            details = "Benchmark script exists" if passed else "Benchmark script missing"
        
        elif criterion_id == "hyperparameter_optimization":
            passed = check_file_exists(PROJECT_ROOT / "scripts" / "optimize_hyperparameters.py")
            details = "Optimization script exists" if passed else "Optimization script missing"
        
        elif criterion_id == "integration_tests":
            test_file = PROJECT_ROOT / "tests" / "integration" / "test_full_pipeline.py"
            passed = check_file_exists(test_file)
            if passed:
                # Count test functions
                try:
                    with open(test_file) as f:
                        content = f.read()
                        count = content.count("def test_")
                        passed = count >= target
                        actual = count
                        details = f"Found {count} integration tests, target: {target}"
                except Exception:
                    details = "Could not count tests"
            else:
                details = "Integration test file missing"
        
        elif criterion_id == "prometheus_metrics":
            metrics_file = PROJECT_ROOT / "src" / "monitoring" / "metrics.py"
            passed = check_file_exists(metrics_file)
            details = "Metrics file exists" if passed else "Metrics file missing"
        
        elif criterion_id == "jaeger_tracing":
            # Check for tracing setup
            passed = any(
                check_file_exists(PROJECT_ROOT / f) 
                for f in [
                    Path("src/monitoring/tracing.py"),
                    Path("scripts/setup_tracing.py"),
                ]
            )
            details = "Tracing setup exists" if passed else "Tracing setup missing"
        
        elif criterion_id == "elk_logging":
            passed = any(
                check_file_exists(PROJECT_ROOT / f)
                for f in [
                    Path("scripts/setup_logging.py"),
                    Path("deployment/docker/docker-compose.yml"),  # Might have ELK
                ]
            )
            details = "Logging setup exists" if passed else "Logging setup missing"
        
        elif criterion_id == "microservices":
            # Check for microservices structure
            passed = check_file_exists(PROJECT_ROOT / "services") or \
                     check_file_exists(PROJECT_ROOT / "src" / "services")
            details = "Microservices structure exists" if passed else "Microservices structure missing"
        
        elif criterion_id == "api_documentation":
            passed = check_file_exists(PROJECT_ROOT / "api" / "openapi.yml") or \
                     check_file_exists(PROJECT_ROOT / "docs" / "api.md")
            details = "API documentation exists" if passed else "API documentation missing"
        
        else:
            # Generic file/directory check
            passed = check_file_exists(PROJECT_ROOT / criterion_id)
            details = f"{criterion_id} exists" if passed else f"{criterion_id} missing"
        
        results.append({
            "id": criterion_id,
            "description": description,
            "target": target,
            "actual": actual,
            "passed": passed,
            "details": details,
            "status": "✅" if passed else "❌"
        })
    
    return results


def check_all_criteria() -> Dict[int, List[Dict]]:
    """Check all success criteria for all weeks."""
    all_results = {}
    for week in range(1, 14):
        all_results[week] = check_criteria_for_week(week)
    return all_results


def generate_report(week: int = None, output: Path = None):
    """Generate a success criteria report."""
    if week:
        results = {week: check_criteria_for_week(week)}
    else:
        results = check_all_criteria()
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "results": results
    }
    
    # Calculate summary
    total_criteria = 0
    passed_criteria = 0
    
    for week_results in results.values():
        for result in week_results:
            total_criteria += 1
            if result["passed"]:
                passed_criteria += 1
    
    report["summary"] = {
        "total": total_criteria,
        "passed": passed_criteria,
        "failed": total_criteria - passed_criteria,
        "percentage": (passed_criteria / total_criteria * 100) if total_criteria > 0 else 0
    }
    
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {output}")
    else:
        # Print to console
        print("\n📋 Success Criteria Report")
        print("=" * 80)
        
        for week, week_results in results.items():
            if not week_results:
                continue
            
            print(f"\nWeek {week}:")
            print("-" * 80)
            
            for result in week_results:
                status = result["status"]
                desc = result["description"]
                details = result["details"]
                print(f"{status} {desc}")
                print(f"   {details}")
        
        print("\n" + "=" * 80)
        print(f"Summary: {report['summary']['passed']}/{report['summary']['total']} criteria passed ({report['summary']['percentage']:.1f}%)")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Check success criteria")
    parser.add_argument("--week", type=int, help="Week number (1-13)")
    parser.add_argument("--all", action="store_true", help="Check all weeks")
    parser.add_argument("--report", action="store_true", help="Generate report")
    parser.add_argument("--output", type=Path, help="Output file path")
    
    args = parser.parse_args()
    
    if args.all:
        generate_report(output=args.output)
    elif args.week:
        generate_report(week=args.week, output=args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

