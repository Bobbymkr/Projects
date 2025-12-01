#!/usr/bin/env python3
"""
Comprehensive Test Runner for Adaptive Traffic Control System.

Provides various test execution modes:
- Unit tests only
- Integration tests only
- Full test suite
- Coverage report
- Load testing
"""

import sys
import subprocess
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent


def run_command(cmd: list, description: str) -> int:
    """Run a command and return exit code."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def run_unit_tests(verbose: bool = False) -> int:
    """Run unit tests only."""
    cmd = ["pytest", "tests/api/test_*.py", "-m", "not integration", "-v" if verbose else ""]
    cmd = [c for c in cmd if c]  # Remove empty strings
    
    return run_command(cmd, "Running Unit Tests")


def run_integration_tests(verbose: bool = False) -> int:
    """Run integration tests only."""
    cmd = [
        "pytest",
        "tests/api/test_integration.py",
        "-m", "integration",
        "-v" if verbose else "",
    ]
    cmd = [c for c in cmd if c]
    
    return run_command(cmd, "Running Integration Tests")


def run_all_tests(verbose: bool = False) -> int:
    """Run all tests."""
    cmd = ["pytest", "tests/api/", "-v" if verbose else ""]
    cmd = [c for c in cmd if c]
    
    return run_command(cmd, "Running All API Tests")


def run_with_coverage(verbose: bool = False) -> int:
    """Run tests with coverage report."""
    cmd = [
        "pytest",
        "tests/api/",
        "--cov=src/api",
        "--cov-report=html",
        "--cov-report=term",
        "--cov-report=xml",
        "-v" if verbose else "",
    ]
    cmd = [c for c in cmd if c]
    
    return run_command(cmd, "Running Tests with Coverage")


def run_load_tests(users: int = 10, spawn_rate: int = 2, duration: str = "1m") -> int:
    """Run Locust load tests."""
    cmd = [
        "locust",
        "-f", "tests/api/load_test.py",
        "--headless",
        "-u", str(users),
        "-r", str(spawn_rate),
        "-t", duration,
        "--host", "http://localhost:8000",
    ]
    
    return run_command(cmd, f"Running Load Tests ({users} users, {duration})")


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description="Test Runner for Adaptive Traffic Control API")
    
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run unit tests only",
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests only",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run tests with coverage report",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Run load tests",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=10,
        help="Number of concurrent users for load tests (default: 10)",
    )
    parser.add_argument(
        "--spawn-rate",
        type=int,
        default=2,
        help="Spawn rate for load tests (default: 2)",
    )
    parser.add_argument(
        "--duration",
        type=str,
        default="1m",
        help="Load test duration (default: 1m)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    # Default: run all tests if no specific option is selected
    if not any([args.unit, args.integration, args.coverage, args.load]):
        args.unit = args.integration = True
    
    exit_code = 0
    
    if args.unit:
        exit_code |= run_unit_tests(args.verbose)
    
    if args.integration:
        exit_code |= run_integration_tests(args.verbose)
    
    if args.coverage:
        exit_code |= run_with_coverage(args.verbose)
    
    if args.load:
        exit_code |= run_load_tests(args.users, args.spawn_rate, args.duration)
    
    if exit_code == 0:
        print("\n" + "="*60)
        print("  ✅ All tests passed!")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("  ❌ Some tests failed!")
        print("="*60 + "\n")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

