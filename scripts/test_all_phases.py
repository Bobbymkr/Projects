#!/usr/bin/env python3
"""
Comprehensive Test Suite for All Phases.

Tests all implemented components across Phases 1-5,
identifies issues, and generates a detailed report.
"""

import sys
import importlib
import traceback
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test results storage
test_results: Dict[str, Dict] = {
    "phase1": {"tests": [], "status": "pending", "errors": []},
    "phase2": {"tests": [], "status": "pending", "errors": []},
    "phase3": {"tests": [], "status": "pending", "errors": []},
    "phase4": {"tests": [], "status": "pending", "errors": []},
    "phase5": {"tests": [], "status": "pending", "errors": []},
}


def test_import(module_path: str, description: str, phase: str) -> Tuple[bool, str]:
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_path)
        return True, "OK"
    except ImportError as e:
        return False, f"ImportError: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def test_file_exists(file_path: str, description: str, phase: str) -> Tuple[bool, str]:
    """Test if a file exists."""
    full_path = project_root / file_path
    if full_path.exists():
        return True, "File exists"
    else:
        return False, f"File not found: {file_path}"


def run_test(test_func, *args, **kwargs):
    """Run a test function and record results."""
    try:
        return test_func(*args, **kwargs)
    except Exception as e:
        return False, f"Test failed: {str(e)}\n{traceback.format_exc()}"


def test_phase1():
    """Test Phase 1: Foundation Hardening."""
    print("\n" + "="*60)
    print("Testing Phase 1: Foundation Hardening")
    print("="*60)
    
    phase_results = test_results["phase1"]
    
    # Test 1.1: Production API Layer
    print("\n[1.1] Testing Production API Layer...")
    
    # Test API main module
    success, msg = run_test(test_import, "src.api.main", "API Main", "phase1")
    phase_results["tests"].append({"test": "API Main Import", "status": "PASS" if success else "FAIL", "message": msg})
    if not success:
        phase_results["errors"].append(msg)
    
    # Test API config
    success, msg = run_test(test_import, "src.api.config", "API Config", "phase1")
    phase_results["tests"].append({"test": "API Config Import", "status": "PASS" if success else "FAIL", "message": msg})
    if not success:
        phase_results["errors"].append(msg)
    
    # Test routes
    success, msg = run_test(test_import, "src.api.routes", "API Routes", "phase1")
    phase_results["tests"].append({"test": "API Routes Import", "status": "PASS" if success else "FAIL", "message": msg})
    
    # Test 1.2: Monitoring & Observability
    print("\n[1.2] Testing Monitoring & Observability...")
    
    success, msg = run_test(test_import, "src.api.monitoring", "Monitoring", "phase1")
    phase_results["tests"].append({"test": "Monitoring Import", "status": "PASS" if success else "FAIL", "message": msg})
    
    success, msg = run_test(test_import, "src.api.logging_config", "Logging Config", "phase1")
    phase_results["tests"].append({"test": "Logging Config Import", "status": "PASS" if success else "FAIL", "message": msg})
    
    success, msg = run_test(test_file_exists, "monitoring/prometheus/prometheus.yml", "Prometheus Config", "phase1")
    phase_results["tests"].append({"test": "Prometheus Config File", "status": "PASS" if success else "FAIL", "message": msg})
    
    # Test 1.3: Test Coverage
    print("\n[1.3] Testing Test Infrastructure...")
    
    success, msg = run_test(test_file_exists, "tests/api/conftest.py", "Test Fixtures", "phase1")
    phase_results["tests"].append({"test": "Test Fixtures File", "status": "PASS" if success else "FAIL", "message": msg})
    
    success, msg = run_test(test_file_exists, ".coveragerc", "Coverage Config", "phase1")
    phase_results["tests"].append({"test": "Coverage Config File", "status": "PASS" if success else "FAIL", "message": msg})
    
    # Determine overall status
    failed_tests = [t for t in phase_results["tests"] if t["status"] == "FAIL"]
    phase_results["status"] = "FAIL" if failed_tests else "PASS"
    
    print(f"\nPhase 1 Status: {phase_results['status']} ({len(failed_tests)} failures)")


def test_phase2():
    """Test Phase 2: Performance & Scalability."""
    print("\n" + "="*60)
    print("Testing Phase 2: Performance & Scalability")
    print("="*60)
    
    phase_results = test_results["phase2"]
    
    # Test caching
    print("\n[2.1] Testing Caching Layer...")
    success, msg = run_test(test_import, "src.api.cache", "Cache Module", "phase2")
    phase_results["tests"].append({"test": "Cache Module Import", "status": "PASS" if success else "FAIL", "message": msg})
    
    # Test rate limiting
    print("\n[2.2] Testing Rate Limiting...")
    success, msg = run_test(test_import, "src.api.rate_limiting", "Rate Limiting", "phase2")
    phase_results["tests"].append({"test": "Rate Limiting Import", "status": "PASS" if success else "FAIL", "message": msg})
    
    # Test database
    print("\n[2.3] Testing Database Pooling...")
    success, msg = run_test(test_import, "src.api.database", "Database Module", "phase2")
    phase_results["tests"].append({"test": "Database Module Import", "status": "PASS" if success else "FAIL", "message": msg})
    
    # Test performance utilities
    success, msg = run_test(test_import, "src.api.performance", "Performance Module", "phase2")
    phase_results["tests"].append({"test": "Performance Module Import", "status": "PASS" if success else "FAIL", "message": msg})
    
    # Test deployment files
    success, msg = run_test(test_file_exists, "deployment/docker/Dockerfile.production", "Production Dockerfile", "phase2")
    phase_results["tests"].append({"test": "Production Dockerfile", "status": "PASS" if success else "FAIL", "message": msg})
    
    failed_tests = [t for t in phase_results["tests"] if t["status"] == "FAIL"]
    phase_results["status"] = "FAIL" if failed_tests else "PASS"
    
    print(f"\nPhase 2 Status: {phase_results['status']} ({len(failed_tests)} failures)")


def test_phase3():
    """Test Phase 3: Advanced Features."""
    print("\n" + "="*60)
    print("Testing Phase 3: Advanced Features")
    print("="*60)
    
    phase_results = test_results["phase3"]
    
    # Test GraphQL
    print("\n[3.1] Testing GraphQL...")
    success, msg = run_test(test_import, "src.api.graphql.schema", "GraphQL Schema", "phase3")
    phase_results["tests"].append({"test": "GraphQL Schema Import", "status": "PASS" if success else "FAIL", "message": msg})
    
    # Test Authentication
    print("\n[3.2] Testing Authentication...")
    success, msg = run_test(test_import, "src.api.auth.oauth2", "OAuth2", "phase3")
    phase_results["tests"].append({"test": "OAuth2 Import", "status": "PASS" if success else "FAIL", "message": msg})
    
    # Test Event Streaming
    print("\n[3.3] Testing Event Streaming...")
    success, msg = run_test(test_import, "src.api.events.stream", "Event Streaming", "phase3")
    phase_results["tests"].append({"test": "Event Streaming Import", "status": "PASS" if success else "FAIL", "message": msg})
    
    # Test Versioning
    success, msg = run_test(test_import, "src.api.versioning", "API Versioning", "phase3")
    phase_results["tests"].append({"test": "API Versioning Import", "status": "PASS" if success else "FAIL", "message": msg})
    
    failed_tests = [t for t in phase_results["tests"] if t["status"] == "FAIL"]
    phase_results["status"] = "FAIL" if failed_tests else "PASS"
    
    print(f"\nPhase 3 Status: {phase_results['status']} ({len(failed_tests)} failures)")


def test_phase4():
    """Test Phase 4: Infrastructure Excellence."""
    print("\n" + "="*60)
    print("Testing Phase 4: Infrastructure Excellence")
    print("="*60)
    
    phase_results = test_results["phase4"]
    
    # Test CI/CD files
    print("\n[4.1] Testing CI/CD Configuration...")
    success, msg = run_test(test_file_exists, ".github/workflows/ci-cd.yml", "CI/CD Workflow", "phase4")
    phase_results["tests"].append({"test": "CI/CD Workflow File", "status": "PASS" if success else "FAIL", "message": msg})
    
    # Test Kubernetes files
    print("\n[4.2] Testing Kubernetes Configurations...")
    kubernetes_files = [
        "deployment/kubernetes/api-deployment.yaml",
        "deployment/kubernetes/configmap.yaml",
        "deployment/kubernetes/namespace.yaml",
        "deployment/kubernetes/ingress.yaml",
        "deployment/kubernetes/network-policy.yaml",
    ]
    
    for file_path in kubernetes_files:
        success, msg = run_test(test_file_exists, file_path, f"K8s {Path(file_path).name}", "phase4")
        phase_results["tests"].append({"test": f"K8s File: {Path(file_path).name}", "status": "PASS" if success else "FAIL", "message": msg})
    
    # Test Helm charts
    print("\n[4.3] Testing Helm Charts...")
    helm_files = [
        "deployment/helm/adaptive-traffic/Chart.yaml",
        "deployment/helm/adaptive-traffic/values.yaml",
    ]
    
    for file_path in helm_files:
        success, msg = run_test(test_file_exists, file_path, f"Helm {Path(file_path).name}", "phase4")
        phase_results["tests"].append({"test": f"Helm File: {Path(file_path).name}", "status": "PASS" if success else "FAIL", "message": msg})
    
    # Test Terraform
    print("\n[4.4] Testing Terraform Configuration...")
    success, msg = run_test(test_file_exists, "deployment/terraform/main.tf", "Terraform Main", "phase4")
    phase_results["tests"].append({"test": "Terraform Main File", "status": "PASS" if success else "FAIL", "message": msg})
    
    # Test migrations
    print("\n[4.5] Testing Database Migrations...")
    success, msg = run_test(test_file_exists, "src/api/database/migrations/alembic.ini", "Alembic Config", "phase4")
    phase_results["tests"].append({"test": "Alembic Config File", "status": "PASS" if success else "FAIL", "message": msg})
    
    failed_tests = [t for t in phase_results["tests"] if t["status"] == "FAIL"]
    phase_results["status"] = "FAIL" if failed_tests else "PASS"
    
    print(f"\nPhase 4 Status: {phase_results['status']} ({len(failed_tests)} failures)")


def test_phase5():
    """Test Phase 5: Innovation & Research (if implemented)."""
    print("\n" + "="*60)
    print("Testing Phase 5: Innovation & Research")
    print("="*60)
    
    phase_results = test_results["phase5"]
    
    # Phase 5 not yet implemented
    phase_results["tests"].append({"test": "Phase 5 Implementation", "status": "SKIP", "message": "Phase 5 not yet implemented"})
    phase_results["status"] = "SKIP"
    
    print("\nPhase 5 Status: SKIP (Not yet implemented)")


def generate_report():
    """Generate comprehensive test report."""
    print("\n" + "="*60)
    print("Generating Test Report")
    print("="*60)
    
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "summary": {
            "total_tests": sum(len(p["tests"]) for p in test_results.values()),
            "passed": sum(len([t for t in p["tests"] if t["status"] == "PASS"]) for p in test_results.values()),
            "failed": sum(len([t for t in p["tests"] if t["status"] == "FAIL"]) for p in test_results.values()),
            "skipped": sum(len([t for t in p["tests"] if t["status"] == "SKIP"]) for p in test_results.values()),
        },
        "phases": test_results,
    }
    
    # Calculate overall status
    phase_statuses = [p["status"] for p in test_results.values() if p["status"] != "SKIP"]
    if all(status == "PASS" for status in phase_statuses):
        overall_status = "PASS"
    elif any(status == "FAIL" for status in phase_statuses):
        overall_status = "FAIL"
    else:
        overall_status = "PARTIAL"
    
    report["overall_status"] = overall_status
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Overall Status: {overall_status}")
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Skipped: {report['summary']['skipped']}")
    print(f"\n{'='*60}\n")
    
    # Print phase summaries
    for phase, results in test_results.items():
        print(f"{phase.upper()}: {results['status']}")
        if results['errors']:
            print(f"  Errors: {len(results['errors'])}")
            for error in results['errors'][:3]:  # Show first 3 errors
                print(f"    - {error[:100]}...")
    
    # Save report to file
    report_file = project_root / "TEST_REPORT.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    return report, overall_status


def main():
    """Main test runner."""
    print("="*60)
    print("COMPREHENSIVE TEST SUITE FOR ALL PHASES")
    print("="*60)
    print(f"Project Root: {project_root}")
    print(f"Python Version: {sys.version}")
    print(f"Test Time: {datetime.now().isoformat()}")
    
    # Run all phase tests
    test_phase1()
    test_phase2()
    test_phase3()
    test_phase4()
    test_phase5()
    
    # Generate report
    report, status = generate_report()
    
    # Return exit code
    if status == "FAIL":
        print("\n[FAIL] TESTS FAILED - Issues need to be resolved")
        return 1
    else:
        print("\n[PASS] ALL TESTS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())

