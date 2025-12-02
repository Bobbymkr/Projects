#!/usr/bin/env python3
"""
Run Week 3 Testing Infrastructure Tests.

Orchestrates all Week 3 tests:
- Integration tests
- Enhanced load tests
- Chaos engineering tests

Usage:
    python scripts/run_week3_tests.py --all
    python scripts/run_week3_tests.py --integration
    python scripts/run_week3_tests.py --load --test gradual
    python scripts/run_week3_tests.py --chaos
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def run_integration_tests():
    """Run integration test suite."""
    print("\n🧪 Running Integration Tests")
    print("=" * 80)
    
    cmd = [
        sys.executable, "-m", "pytest",
        str(PROJECT_ROOT / "tests" / "integration" / "test_full_pipeline.py"),
        "-v",
        "--tb=short"
    ]
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def run_load_tests(test_type: str = "gradual", short: bool = False):
    """Run enhanced load tests."""
    print(f"\n📊 Running Load Tests: {test_type}")
    print("=" * 80)
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tests" / "performance" / "enhanced_load_tests.py"),
        "--test", test_type
    ]
    
    if short:
        cmd.append("--short")
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def run_chaos_tests(test_type: str = "all", duration: int = 30):
    """Run chaos engineering tests."""
    print(f"\n💥 Running Chaos Tests: {test_type}")
    print("=" * 80)
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tests" / "chaos" / "chaos_engineering.py"),
        "--test", test_type,
        "--duration", str(duration)
    ]
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run Week 3 testing infrastructure tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--load", action="store_true", help="Run load tests")
    parser.add_argument("--chaos", action="store_true", help="Run chaos tests")
    parser.add_argument("--test", choices=["gradual", "spike", "soak"], default="gradual", help="Load test type")
    parser.add_argument("--short", action="store_true", help="Short tests (for development)")
    
    args = parser.parse_args()
    
    results = {}
    
    if args.all or args.integration:
        results["integration"] = run_integration_tests()
    
    if args.all or args.load:
        results["load"] = run_load_tests(args.test, args.short)
    
    if args.all or args.chaos:
        duration = 10 if args.short else 30
        results["chaos"] = run_chaos_tests("all", duration)
    
    if not any([args.all, args.integration, args.load, args.chaos]):
        parser.print_help()
        return
    
    # Print summary
    print("\n📊 Week 3 Test Summary")
    print("=" * 80)
    for test_type, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_type.upper()}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")


if __name__ == "__main__":
    main()

