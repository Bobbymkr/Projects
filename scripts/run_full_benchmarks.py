#!/usr/bin/env python3
"""
Run full benchmarks for all technologies with statistical significance.

This script runs comprehensive benchmarks as specified in Week 2:
- Phase 1: Quick validation (100 episodes)
- Phase 2: Full benchmarks (5000 episodes) with parallel execution
- Statistical significance: 30+ runs per technology × scenario

Usage:
    python scripts/run_full_benchmarks.py --phase 1  # Quick validation
    python scripts/run_full_benchmarks.py --phase 2  # Full benchmarks
    python scripts/run_full_benchmarks.py --all      # Both phases
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def run_phase1_quick_validation():
    """Phase 1: Quick benchmarks (100 episodes) for initial validation."""
    print("\n🚀 Phase 1: Quick Validation Benchmarks")
    print("=" * 80)
    print("Running quick benchmarks (100 episodes) for initial validation...")
    
    # Run quick benchmarks for all technologies
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "benchmark_all_technologies.py"),
        "--quick",
        "--parallel"
    ]
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    if result.returncode == 0:
        print("\n✅ Phase 1 complete: Quick validation benchmarks finished")
    else:
        print("\n⚠️ Phase 1 completed with errors")
    
    return result.returncode == 0


def run_phase2_full_benchmarks():
    """Phase 2: Full benchmarks (5000 episodes) with parallel execution."""
    print("\n🚀 Phase 2: Full Benchmarks")
    print("=" * 80)
    print("Running full benchmarks (5000 episodes) with parallel execution...")
    print("This may take several hours...")
    
    # Run full benchmarks
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "benchmark_all_technologies.py"),
        "--episodes", "5000",
        "--parallel"
    ]
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    if result.returncode == 0:
        print("\n✅ Phase 2 complete: Full benchmarks finished")
    else:
        print("\n⚠️ Phase 2 completed with errors")
    
    return result.returncode == 0


def generate_performance_report():
    """Generate performance analysis report from benchmark results."""
    print("\n📊 Generating Performance Analysis Report...")
    
    # Find latest benchmark file
    benchmarks_dir = PROJECT_ROOT / "results" / "benchmarks"
    if not benchmarks_dir.exists():
        print("No benchmark results found")
        return
    
    benchmark_files = sorted(benchmarks_dir.glob("benchmark_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not benchmark_files:
        print("No benchmark files found")
        return
    
    latest_file = benchmark_files[0]
    
    # Run analysis
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "analyze_benchmark_results.py"),
        "--input", str(latest_file)
    ]
    
    subprocess.run(cmd, cwd=PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Run full benchmarks (Week 2)")
    parser.add_argument("--phase", type=int, choices=[1, 2], help="Run specific phase (1=quick, 2=full)")
    parser.add_argument("--all", action="store_true", help="Run both phases")
    parser.add_argument("--report", action="store_true", help="Generate performance report")
    
    args = parser.parse_args()
    
    if args.report:
        generate_performance_report()
        return
    
    if args.all:
        # Run both phases
        phase1_success = run_phase1_quick_validation()
        
        if phase1_success:
            print("\n" + "=" * 80)
            response = input("Phase 1 complete. Proceed with Phase 2 (full benchmarks)? This will take several hours. (y/n): ")
            if response.lower() == 'y':
                run_phase2_full_benchmarks()
                generate_performance_report()
        else:
            print("\n⚠️ Phase 1 had errors. Review before proceeding to Phase 2.")
    
    elif args.phase == 1:
        run_phase1_quick_validation()
    
    elif args.phase == 2:
        run_phase2_full_benchmarks()
        generate_performance_report()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

