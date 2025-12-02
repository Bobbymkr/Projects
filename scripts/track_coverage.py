#!/usr/bin/env python3
"""
Track test coverage progress.

Usage:
    python scripts/track_coverage.py --target 95 --current 90
    python scripts/track_coverage.py --analyze --output coverage_gaps.json
"""

import json
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
COVERAGE_DIR = PROJECT_ROOT / "execution" / "coverage"


def get_current_coverage() -> float:
    """Get current test coverage percentage."""
    try:
        result = subprocess.run(
            ["pytest", "--cov=src", "--cov-report=json", "--quiet"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        
        coverage_file = PROJECT_ROOT / "coverage.json"
        if coverage_file.exists():
            with open(coverage_file) as f:
                data = json.load(f)
                return data["totals"]["percent_covered"]
        
        # Fallback: parse from term output
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
                        return float(part.replace('%', ''))
        
        return 0.0
    except Exception as e:
        print(f"Warning: Could not get test coverage: {e}", file=sys.stderr)
        return 0.0


def analyze_coverage_gaps() -> List[Dict]:
    """Analyze coverage gaps and identify files needing tests."""
    try:
        result = subprocess.run(
            ["pytest", "--cov=src", "--cov-report=json", "--quiet"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        
        coverage_file = PROJECT_ROOT / "coverage.json"
        if not coverage_file.exists():
            return []
        
        with open(coverage_file) as f:
            data = json.load(f)
        
        gaps = []
        for file_path, file_data in data["files"].items():
            coverage = file_data["summary"]["percent_covered"]
            missing_lines = file_data.get("missing_lines", [])
            
            if coverage < 95:  # Target threshold
                gaps.append({
                    "file": file_path,
                    "coverage": coverage,
                    "missing_lines": missing_lines,
                    "missing_count": len(missing_lines),
                    "gap": 95 - coverage
                })
        
        # Sort by gap (largest gap first)
        gaps.sort(key=lambda x: x["gap"], reverse=True)
        
        return gaps
    except Exception as e:
        print(f"Warning: Could not analyze coverage gaps: {e}", file=sys.stderr)
        return []


def track_coverage(target: float = 95.0, current: Optional[float] = None) -> Dict:
    """Track test coverage progress."""
    if current is None:
        current = get_current_coverage()
    
    gap = target - current
    progress = (current / target * 100) if target > 0 else 0
    
    coverage_data = {
        "timestamp": datetime.now().isoformat(),
        "current": current,
        "target": target,
        "gap": gap,
        "progress": progress,
        "status": "✅" if current >= target else "⚠️" if current >= target * 0.9 else "❌"
    }
    
    # Save to file
    COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
    coverage_file = COVERAGE_DIR / f"coverage_{datetime.now().strftime('%Y%m%d')}.json"
    with open(coverage_file, 'w') as f:
        json.dump(coverage_data, f, indent=2)
    
    return coverage_data


def main():
    parser = argparse.ArgumentParser(description="Track test coverage")
    parser.add_argument("--target", type=float, default=95.0, help="Target coverage percentage")
    parser.add_argument("--current", type=float, help="Current coverage percentage (auto-detected if not provided)")
    parser.add_argument("--analyze", action="store_true", help="Analyze coverage gaps")
    parser.add_argument("--output", type=Path, help="Output file path")
    
    args = parser.parse_args()
    
    if args.analyze:
        gaps = analyze_coverage_gaps()
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(gaps, f, indent=2)
            print(f"Coverage gaps saved to {args.output}")
        else:
            print("\n📊 Coverage Gap Analysis")
            print("=" * 80)
            print(f"Found {len(gaps)} files below 95% coverage:\n")
            
            for i, gap in enumerate(gaps[:20], 1):  # Top 20
                print(f"{i:2d}. {gap['file']}")
                print(f"    Coverage: {gap['coverage']:.2f}% (Gap: {gap['gap']:.2f}%)")
                print(f"    Missing lines: {gap['missing_count']}")
                print()
    else:
        coverage_data = track_coverage(args.target, args.current)
        
        # Print summary
        print("\n📊 Test Coverage Tracking")
        print("=" * 60)
        print(f"Current: {coverage_data['current']:.2f}%")
        print(f"Target: {coverage_data['target']:.2f}%")
        print(f"Gap: {coverage_data['gap']:.2f}%")
        print(f"Progress: {coverage_data['progress']:.2f}%")
        print(f"Status: {coverage_data['status']}")
        print("=" * 60)


if __name__ == "__main__":
    main()

