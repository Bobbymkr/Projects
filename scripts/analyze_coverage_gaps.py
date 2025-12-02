#!/usr/bin/env python3
"""
Analyze test coverage gaps and identify files needing tests.

Usage:
    python scripts/analyze_coverage_gaps.py --output coverage_gaps.json
    python scripts/analyze_coverage_gaps.py --threshold 80 --detailed
"""

import json
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def run_coverage_analysis() -> Dict[str, Any]:
    """Run pytest with coverage and return results."""
    print("Running coverage analysis...")
    
    try:
        # Run pytest with JSON coverage report
        result = subprocess.run(
            ["pytest", "--cov=src", "--cov-report=json", "--cov-report=term-missing", "--quiet"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        
        if result.returncode != 0:
            print(f"Warning: pytest returned non-zero exit code: {result.returncode}", file=sys.stderr)
        
        # Load coverage JSON
        coverage_file = PROJECT_ROOT / "coverage.json"
        if not coverage_file.exists():
            print("Error: coverage.json not found. Run pytest with --cov-report=json first.", file=sys.stderr)
            return {}
        
        with open(coverage_file) as f:
            return json.load(f)
    
    except FileNotFoundError:
        print("Error: pytest not found. Please install pytest and pytest-cov.", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"Error running coverage analysis: {e}", file=sys.stderr)
        return {}


def analyze_gaps(coverage_data: Dict[str, Any], threshold: float = 95.0) -> List[Dict[str, Any]]:
    """Analyze coverage gaps and identify files below threshold."""
    if not coverage_data or "files" not in coverage_data:
        return []
    
    gaps = []
    
    for file_path, file_data in coverage_data["files"].items():
        # Skip test files and __pycache__
        if "test" in file_path.lower() or "__pycache__" in file_path:
            continue
        
        summary = file_data.get("summary", {})
        coverage = summary.get("percent_covered", 0.0)
        missing_lines = file_data.get("missing_lines", [])
        excluded_lines = file_data.get("excluded_lines", [])
        
        if coverage < threshold:
            gap = threshold - coverage
            
            # Calculate line statistics
            num_statements = summary.get("num_statements", 0)
            missing_count = summary.get("missing_lines", 0)
            covered_count = summary.get("covered_lines", 0)
            
            gaps.append({
                "file": file_path,
                "coverage": coverage,
                "target": threshold,
                "gap": gap,
                "statements": num_statements,
                "covered": covered_count,
                "missing": missing_count,
                "missing_lines": missing_lines[:50],  # Limit to first 50
                "excluded_lines": excluded_lines[:20],  # Limit to first 20
                "priority": "high" if gap > 20 else "medium" if gap > 10 else "low"
            })
    
    # Sort by gap (largest gap first)
    gaps.sort(key=lambda x: x["gap"], reverse=True)
    
    return gaps


def generate_report(gaps: List[Dict[str, Any]], output: Path = None, detailed: bool = False) -> Dict[str, Any]:
    """Generate coverage gap report."""
    total_files = len(gaps)
    high_priority = sum(1 for g in gaps if g["priority"] == "high")
    medium_priority = sum(1 for g in gaps if g["priority"] == "medium")
    low_priority = sum(1 for g in gaps if g["priority"] == "low")
    
    # Calculate statistics
    total_statements = sum(g["statements"] for g in gaps)
    total_missing = sum(g["missing"] for g in gaps)
    avg_coverage = sum(g["coverage"] for g in gaps) / total_files if total_files > 0 else 0
    
    report = {
        "summary": {
            "total_files_below_threshold": total_files,
            "high_priority": high_priority,
            "medium_priority": medium_priority,
            "low_priority": low_priority,
            "total_statements": total_statements,
            "total_missing": total_missing,
            "average_coverage": avg_coverage
        },
        "gaps": gaps if detailed else gaps[:50]  # Limit if not detailed
    }
    
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Coverage gap report saved to {output}")
    else:
        # Print to console
        print("\n📊 Coverage Gap Analysis")
        print("=" * 80)
        print(f"Files below threshold: {total_files}")
        print(f"High priority: {high_priority}, Medium: {medium_priority}, Low: {low_priority}")
        print(f"Total statements: {total_statements}, Missing: {total_missing}")
        print(f"Average coverage: {avg_coverage:.2f}%")
        print("\nTop 20 files needing coverage:")
        print("-" * 80)
        
        for i, gap in enumerate(gaps[:20], 1):
            priority_icon = "🔴" if gap["priority"] == "high" else "🟡" if gap["priority"] == "medium" else "🟢"
            print(f"{i:2d}. {priority_icon} {gap['file']}")
            print(f"    Coverage: {gap['coverage']:.2f}% (Gap: {gap['gap']:.2f}%)")
            print(f"    Missing: {gap['missing']}/{gap['statements']} statements")
            if detailed and gap["missing_lines"]:
                print(f"    Missing lines: {gap['missing_lines'][:10]}...")
            print()
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze test coverage gaps")
    parser.add_argument("--output", type=Path, help="Output JSON file path")
    parser.add_argument("--threshold", type=float, default=95.0, help="Coverage threshold (default: 95.0)")
    parser.add_argument("--detailed", action="store_true", help="Include detailed missing lines")
    parser.add_argument("--no-run", action="store_true", help="Don't run coverage, use existing coverage.json")
    
    args = parser.parse_args()
    
    # Run coverage analysis
    if not args.no_run:
        coverage_data = run_coverage_analysis()
    else:
        coverage_file = PROJECT_ROOT / "coverage.json"
        if not coverage_file.exists():
            print("Error: coverage.json not found. Run without --no-run first.", file=sys.stderr)
            sys.exit(1)
        with open(coverage_file) as f:
            coverage_data = json.load(f)
    
    if not coverage_data:
        print("Error: Could not get coverage data.", file=sys.stderr)
        sys.exit(1)
    
    # Analyze gaps
    gaps = analyze_gaps(coverage_data, args.threshold)
    
    if not gaps:
        print(f"✅ All files meet the {args.threshold}% coverage threshold!")
        return
    
    # Generate report
    generate_report(gaps, args.output, args.detailed)


if __name__ == "__main__":
    main()

