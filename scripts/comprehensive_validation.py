#!/usr/bin/env python3
"""
Comprehensive Testing and Validation Script.

Runs all tests systematically and generates validation reports.
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def run_command(cmd: List[str], description: str) -> Dict[str, Any]:
    """Run a command and capture results."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*80)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        return {
            "description": description,
            "command": ' '.join(cmd),
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "output_lines": result.stdout.split('\n')[:50]  # First 50 lines
        }
    except subprocess.TimeoutExpired:
        return {
            "description": description,
            "command": ' '.join(cmd),
            "success": False,
            "error": "Timeout after 600 seconds"
        }
    except Exception as e:
        return {
            "description": description,
            "command": ' '.join(cmd),
            "success": False,
            "error": str(e)
        }


def run_unit_tests() -> Dict[str, Any]:
    """Run unit tests."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit/",
        "-v",
        "--tb=short",
        "--maxfail=10"
    ]
    return run_command(cmd, "Unit Tests")


def run_integration_tests() -> Dict[str, Any]:
    """Run integration tests."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/integration/",
        "-v",
        "--tb=short",
        "--maxfail=10"
    ]
    return run_command(cmd, "Integration Tests")


def run_coverage_analysis() -> Dict[str, Any]:
    """Run coverage analysis."""
    cmd = [
        sys.executable, "-m", "pytest",
        "--cov=src",
        "--cov-report=json",
        "--cov-report=term-missing",
        "-q",
        "tests/unit/",
        "tests/integration/"
    ]
    return run_command(cmd, "Coverage Analysis")


def validate_scripts() -> Dict[str, Any]:
    """Validate all scripts are executable."""
    scripts_dir = PROJECT_ROOT / "scripts"
    scripts = list(scripts_dir.glob("*.py"))
    
    results = {
        "total_scripts": len(scripts),
        "validated": 0,
        "errors": []
    }
    
    for script in scripts:
        try:
            # Try to import/parse the script
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(script)],
                capture_output=True,
                timeout=10
            )
            if result.returncode == 0:
                results["validated"] += 1
            else:
                results["errors"].append({
                    "script": script.name,
                    "error": result.stderr.decode()
                })
        except Exception as e:
            results["errors"].append({
                "script": script.name,
                "error": str(e)
            })
    
    results["success"] = results["validated"] == results["total_scripts"]
    return results


def validate_configurations() -> Dict[str, Any]:
    """Validate configuration files."""
    configs = {
        "prometheus": PROJECT_ROOT / "monitoring" / "prometheus" / "prometheus.yml",
        "alerts": PROJECT_ROOT / "monitoring" / "prometheus" / "alerts" / "api_alerts.yml",
        "hpa": PROJECT_ROOT / "deployment" / "kubernetes" / "hpa-enhanced.yaml",
        "load_balancer": PROJECT_ROOT / "deployment" / "kubernetes" / "load-balancer.yaml"
    }
    
    results = {
        "total_configs": len(configs),
        "validated": 0,
        "missing": [],
        "errors": []
    }
    
    for name, path in configs.items():
        if path.exists():
            results["validated"] += 1
        else:
            results["missing"].append(name)
    
    results["success"] = results["validated"] == results["total_configs"]
    return results


def generate_validation_report(results: Dict[str, Any]) -> str:
    """Generate comprehensive validation report."""
    report = f"""# Comprehensive Validation Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

- **Unit Tests**: {'✅ PASSED' if results.get('unit_tests', {}).get('success') else '❌ FAILED'}
- **Integration Tests**: {'✅ PASSED' if results.get('integration_tests', {}).get('success') else '❌ FAILED'}
- **Coverage Analysis**: {'✅ COMPLETE' if results.get('coverage', {}).get('success') else '❌ FAILED'}
- **Scripts Validation**: {'✅ PASSED' if results.get('scripts', {}).get('success') else '❌ FAILED'}
- **Configurations**: {'✅ VALID' if results.get('configs', {}).get('success') else '❌ INVALID'}

---

## Detailed Results

### Unit Tests
"""
    
    unit_results = results.get('unit_tests', {})
    if unit_results.get('success'):
        report += "✅ **Status**: PASSED\n\n"
    else:
        report += f"❌ **Status**: FAILED (Return code: {unit_results.get('returncode')})\n\n"
        if unit_results.get('stderr'):
            report += f"**Errors**:\n```\n{unit_results['stderr'][:500]}\n```\n\n"
    
    report += "\n### Integration Tests\n"
    integration_results = results.get('integration_tests', {})
    if integration_results.get('success'):
        report += "✅ **Status**: PASSED\n\n"
    else:
        report += f"❌ **Status**: FAILED (Return code: {integration_results.get('returncode')})\n\n"
        if integration_results.get('stderr'):
            report += f"**Errors**:\n```\n{integration_results['stderr'][:500]}\n```\n\n"
    
    report += "\n### Coverage Analysis\n"
    coverage_results = results.get('coverage', {})
    if coverage_results.get('success'):
        report += "✅ **Status**: COMPLETE\n\n"
        # Try to extract coverage percentage
        if 'stdout' in coverage_results:
            for line in coverage_results['stdout'].split('\n'):
                if 'TOTAL' in line or 'total' in line.lower():
                    report += f"**Coverage**: {line}\n\n"
    else:
        report += f"❌ **Status**: FAILED\n\n"
    
    report += "\n### Scripts Validation\n"
    scripts_results = results.get('scripts', {})
    report += f"- **Total Scripts**: {scripts_results.get('total_scripts', 0)}\n"
    report += f"- **Validated**: {scripts_results.get('validated', 0)}\n"
    if scripts_results.get('errors'):
        report += f"- **Errors**: {len(scripts_results['errors'])}\n"
    
    report += "\n### Configuration Validation\n"
    configs_results = results.get('configs', {})
    report += f"- **Total Configs**: {configs_results.get('total_configs', 0)}\n"
    report += f"- **Validated**: {configs_results.get('validated', 0)}\n"
    if configs_results.get('missing'):
        report += f"- **Missing**: {', '.join(configs_results['missing'])}\n"
    
    report += "\n---\n\n## Recommendations\n\n"
    
    # Add recommendations based on results
    if not unit_results.get('success'):
        report += "1. Fix unit test failures\n"
    if not integration_results.get('success'):
        report += "2. Fix integration test failures\n"
    if not coverage_results.get('success'):
        report += "3. Run coverage analysis\n"
    if configs_results.get('missing'):
        report += "4. Create missing configuration files\n"
    
    report += "\n---\n\n*Generated by Comprehensive Validation Script*\n"
    
    return report


def main():
    """Run comprehensive validation."""
    print("\n" + "="*80)
    print("COMPREHENSIVE TESTING AND VALIDATION")
    print("="*80)
    
    results = {}
    
    # Run tests
    print("\n[1/5] Running Unit Tests...")
    results['unit_tests'] = run_unit_tests()
    
    print("\n[2/5] Running Integration Tests...")
    results['integration_tests'] = run_integration_tests()
    
    print("\n[3/5] Running Coverage Analysis...")
    results['coverage'] = run_coverage_analysis()
    
    print("\n[4/5] Validating Scripts...")
    results['scripts'] = validate_scripts()
    
    print("\n[5/5] Validating Configurations...")
    results['configs'] = validate_configurations()
    
    # Generate report
    report = generate_validation_report(results)
    
    # Save report
    output_file = PROJECT_ROOT / "execution" / "reports" / f"comprehensive_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Save JSON results
    json_file = PROJECT_ROOT / "execution" / "reports" / f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\nReport saved to: {output_file}")
    print(f"Results saved to: {json_file}")
    # Print summary without Unicode characters
    print("\nSummary:")
    print(f"  Unit Tests: {'PASSED' if results.get('unit_tests', {}).get('success') else 'FAILED'}")
    print(f"  Integration Tests: {'PASSED' if results.get('integration_tests', {}).get('success') else 'FAILED'}")
    print(f"  Coverage: {'COMPLETE' if results.get('coverage', {}).get('success') else 'FAILED'}")
    print(f"  Scripts: {'VALID' if results.get('scripts', {}).get('success') else 'INVALID'}")
    print(f"  Configs: {'VALID' if results.get('configs', {}).get('success') else 'INVALID'}")
    
    # Return success status
    all_passed = (
        results.get('unit_tests', {}).get('success', False) and
        results.get('integration_tests', {}).get('success', False) and
        results.get('scripts', {}).get('success', False) and
        results.get('configs', {}).get('success', False)
    )
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

