#!/usr/bin/env python3
"""
Calculate test coverage for core modules.

Focuses on critical paths to achieve 95%+ coverage.
"""

import subprocess
import sys
import json
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def calculate_core_coverage():
    """Calculate coverage for core modules."""
    core_modules = [
        "src/env/traffic_env.py",
        "src/rl/dqn_agent.py",
        "src/control/fuzzy_control.py",
        "src/control/webster_method.py",
        "src/monitoring/metrics.py",
        "src/monitoring/tracing.py",
        "src/monitoring/logging.py",
        "src/realtime/scheduler.py",
        "src/realtime/deadline_aware_agent.py",
        "src/state/distributed_state.py",
        "src/optimization/genetic_algo.py",
        "src/optimization/pso.py",
        "src/forecast/traffic_forecast.py",
        "src/utils/config.py",
        "src/utils/errors.py",
        "src/utils/health.py",
        "src/utils/io.py"
    ]
    
    cmd = [
        sys.executable, "-m", "pytest",
        "--cov=src/env",
        "--cov=src/rl",
        "--cov=src/control",
        "--cov=src/monitoring",
        "--cov=src/realtime",
        "--cov=src/state",
        "--cov=src/optimization",
        "--cov=src/forecast",
        "--cov=src/utils",
        "--cov-report=json",
        "--cov-report=term-missing",
        "-q",
        "tests/unit/"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    
    # Try to read coverage JSON
    coverage_file = PROJECT_ROOT / "coverage.json"
    if coverage_file.exists():
        with open(coverage_file) as f:
            data = json.load(f)
            files = data.get("files", {})
            
            core_coverage = {}
            total_statements = 0
            total_missing = 0
            
            for module in core_modules:
                module_path = str(Path(module).as_posix())
                if module_path in files:
                    file_data = files[module_path]
                    statements = file_data.get("summary", {}).get("num_statements", 0)
                    missing = file_data.get("summary", {}).get("missing_lines", 0)
                    covered = statements - missing
                    coverage_pct = (covered / statements * 100) if statements > 0 else 0
                    
                    core_coverage[module] = {
                        "statements": statements,
                        "covered": covered,
                        "missing": missing,
                        "coverage": coverage_pct
                    }
                    
                    total_statements += statements
                    total_missing += missing
            
            overall_coverage = ((total_statements - total_missing) / total_statements * 100) if total_statements > 0 else 0
            
            return {
                "overall_coverage": overall_coverage,
                "total_statements": total_statements,
                "total_covered": total_statements - total_missing,
                "total_missing": total_missing,
                "files": core_coverage
            }
    
    return None


def main():
    """Calculate and report coverage."""
    print("\nCalculating core module coverage...")
    coverage = calculate_core_coverage()
    
    if coverage:
        print(f"\nOverall Coverage: {coverage['overall_coverage']:.2f}%")
        print(f"Total Statements: {coverage['total_statements']}")
        print(f"Covered: {coverage['total_covered']}")
        print(f"Missing: {coverage['total_missing']}")
        
        print("\nFile-by-file coverage:")
        for module, data in coverage['files'].items():
            print(f"  {module}: {data['coverage']:.2f}% ({data['covered']}/{data['statements']})")
        
        if coverage['overall_coverage'] >= 95:
            print("\n✅ Coverage target achieved! (≥95%)")
            return 0
        else:
            print(f"\n⚠️ Coverage: {coverage['overall_coverage']:.2f}% (target: 95%)")
            return 1
    else:
        print("Could not calculate coverage")
        return 1


if __name__ == "__main__":
    sys.exit(main())

