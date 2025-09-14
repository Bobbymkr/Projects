#!/usr/bin/env python3
"""
Generate CI summary report from all test artifacts.

This script aggregates results from various test stages and generates
a comprehensive summary for CI/CD pipeline reporting.
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import re


class CISummaryGenerator:
    """Generate comprehensive CI summary from test artifacts."""
    
    def __init__(self, artifacts_dir: Path, output_file: Path):
        self.artifacts_dir = artifacts_dir
        self.output_file = output_file
        self.summary_data = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'unknown',
            'test_results': {},
            'coverage_data': {},
            'performance_data': {},
            'quality_metrics': {}
        }
    
    def collect_junit_results(self) -> Dict[str, Any]:
        """Collect all JUnit XML test results."""
        results = {}
        
        # Find all JUnit XML files in artifacts
        junit_files = list(self.artifacts_dir.rglob("test-results-*.xml"))
        
        for junit_file in junit_files:
            try:
                tree = ET.parse(junit_file)
                root = tree.getroot()
                
                # Extract test type from filename
                test_type = "unknown"
                if "unit" in junit_file.name:
                    test_type = "unit"
                elif "integration" in junit_file.name:
                    test_type = "integration"
                elif "system" in junit_file.name:
                    test_type = "system"
                elif "sumo" in junit_file.name:
                    test_type = "sumo"
                elif "perf" in junit_file.name:
                    test_type = "performance"
                
                results[test_type] = {
                    'total': int(root.get('tests', 0)),
                    'passed': int(root.get('tests', 0)) - int(root.get('failures', 0)) - int(root.get('errors', 0)) - int(root.get('skipped', 0)),
                    'failed': int(root.get('failures', 0)),
                    'errors': int(root.get('errors', 0)),
                    'skipped': int(root.get('skipped', 0)),
                    'time': float(root.get('time', 0)),
                    'file': str(junit_file)
                }
                
            except Exception as e:
                print(f"Error parsing {junit_file}: {e}")
                
        return results
    
    def collect_coverage_data(self) -> Dict[str, Any]:
        """Collect coverage information from coverage.xml files."""
        coverage_data = {}
        
        # Find coverage XML files
        coverage_files = list(self.artifacts_dir.rglob("coverage.xml"))
        
        for coverage_file in coverage_files:
            try:
                tree = ET.parse(coverage_file)
                root = tree.getroot()
                
                # Extract overall coverage
                coverage_elem = root.find('.//coverage')
                if coverage_elem is not None:
                    lines_covered = int(coverage_elem.get('lines-covered', 0))
                    lines_valid = int(coverage_elem.get('lines-valid', 1))
                    line_rate = float(coverage_elem.get('line-rate', 0))
                    branch_rate = float(coverage_elem.get('branch-rate', 0))
                    
                    coverage_data['overall'] = {
                        'line_coverage': line_rate * 100,
                        'branch_coverage': branch_rate * 100,
                        'lines_covered': lines_covered,
                        'lines_total': lines_valid
                    }
                
                # Extract package-level coverage
                packages = {}
                for package in root.findall('.//package'):
                    pkg_name = package.get('name', '')
                    pkg_line_rate = float(package.get('line-rate', 0))
                    pkg_branch_rate = float(package.get('branch-rate', 0))
                    
                    packages[pkg_name] = {
                        'line_coverage': pkg_line_rate * 100,
                        'branch_coverage': pkg_branch_rate * 100
                    }
                
                coverage_data['packages'] = packages
                
            except Exception as e:
                print(f"Error parsing coverage file {coverage_file}: {e}")
        
        return coverage_data
    
    def collect_performance_data(self) -> Dict[str, Any]:
        """Collect performance benchmark data."""
        perf_data = {}
        
        # Find benchmark JSON files
        benchmark_files = list(self.artifacts_dir.rglob("benchmark-results.json"))
        
        for benchmark_file in benchmark_files:
            try:
                with open(benchmark_file, 'r') as f:
                    data = json.load(f)
                
                benchmarks = data.get('benchmarks', [])
                
                for benchmark in benchmarks:
                    test_name = benchmark.get('name', 'unknown')
                    stats = benchmark.get('stats', {})
                    
                    perf_data[test_name] = {
                        'mean': stats.get('mean', 0),
                        'stddev': stats.get('stddev', 0),
                        'min': stats.get('min', 0),
                        'max': stats.get('max', 0),
                        'rounds': stats.get('rounds', 0)
                    }
                    
            except Exception as e:
                print(f"Error parsing benchmark file {benchmark_file}: {e}")
        
        return perf_data
    
    def calculate_overall_status(self, test_results: Dict) -> str:
        """Calculate overall CI status from test results."""
        total_failures = 0
        total_errors = 0
        total_tests = 0
        
        for test_type, results in test_results.items():
            if test_type != 'performance':  # Performance tests are optional
                total_failures += results.get('failed', 0)
                total_errors += results.get('errors', 0)
                total_tests += results.get('total', 0)
        
        if total_failures == 0 and total_errors == 0 and total_tests > 0:
            return 'success'
        elif total_tests == 0:
            return 'no_tests'
        else:
            return 'failure'
    
    def generate_summary(self) -> str:
        """Generate markdown summary report."""
        # Collect all data
        test_results = self.collect_junit_results()
        coverage_data = self.collect_coverage_data()
        performance_data = self.collect_performance_data()
        
        # Store in summary data
        self.summary_data['test_results'] = test_results
        self.summary_data['coverage_data'] = coverage_data
        self.summary_data['performance_data'] = performance_data
        self.summary_data['overall_status'] = self.calculate_overall_status(test_results)
        
        # Generate report
        report = []
        report.append("# CI Pipeline Summary")
        report.append(f"**Generated:** {self.summary_data['timestamp']}")
        
        # Overall status
        status_icon = "✅" if self.summary_data['overall_status'] == 'success' else "❌"
        report.append(f"**Overall Status:** {status_icon} {self.summary_data['overall_status'].title()}")
        report.append("")
        
        # Test Results Summary
        report.append("## Test Results")
        if test_results:
            report.append("| Test Type | Total | Passed | Failed | Errors | Skipped | Time (s) |")
            report.append("|-----------|-------|--------|--------|--------|---------|----------|")
            
            total_all = 0
            passed_all = 0
            failed_all = 0
            errors_all = 0
            skipped_all = 0
            time_all = 0
            
            for test_type, results in test_results.items():
                total = results.get('total', 0)
                passed = results.get('passed', 0)
                failed = results.get('failed', 0)
                errors = results.get('errors', 0)
                skipped = results.get('skipped', 0)
                time = results.get('time', 0)
                
                status_icon = "✅" if failed == 0 and errors == 0 else "❌"
                report.append(f"| {status_icon} {test_type.title()} | {total} | {passed} | {failed} | {errors} | {skipped} | {time:.1f} |")
                
                total_all += total
                passed_all += passed
                failed_all += failed
                errors_all += errors
                skipped_all += skipped
                time_all += time
            
            # Summary row
            overall_status = "✅" if failed_all == 0 and errors_all == 0 else "❌"
            report.append(f"| **{overall_status} TOTAL** | **{total_all}** | **{passed_all}** | **{failed_all}** | **{errors_all}** | **{skipped_all}** | **{time_all:.1f}** |")
        else:
            report.append("No test results found.")
        report.append("")
        
        # Coverage Summary
        report.append("## Coverage Summary")
        if coverage_data and 'overall' in coverage_data:
            overall_cov = coverage_data['overall']
            line_cov = overall_cov['line_coverage']
            branch_cov = overall_cov['branch_coverage']
            
            # Check against thresholds from strategy
            line_status = "✅" if line_cov >= 85 else "❌"
            branch_status = "✅" if branch_cov >= 80 else "❌"
            
            report.append(f"- {line_status} **Line Coverage:** {line_cov:.1f}% (Target: 85%)")
            report.append(f"- {branch_status} **Branch Coverage:** {branch_cov:.1f}% (Target: 80%)")
            report.append(f"- **Lines Covered:** {overall_cov['lines_covered']:,} / {overall_cov['lines_total']:,}")
            
            # Package breakdown for critical components
            if 'packages' in coverage_data:
                report.append("")
                report.append("### Core Component Coverage")
                core_packages = ['src.rl', 'src.env', 'src.sumo_integration', 'src.forecast', 'src.vision']
                
                for pkg_name, pkg_data in coverage_data['packages'].items():
                    if any(core in pkg_name for core in core_packages):
                        pkg_line_cov = pkg_data['line_coverage']
                        pkg_status = "✅" if pkg_line_cov >= 90 else "❌"
                        report.append(f"- {pkg_status} **{pkg_name}:** {pkg_line_cov:.1f}%")
        else:
            report.append("No coverage data available.")
        report.append("")
        
        # Performance Summary
        report.append("## Performance Benchmarks")
        if performance_data:
            report.append("| Benchmark | Mean (s) | StdDev | Min | Max | Rounds |")
            report.append("|-----------|----------|--------|-----|-----|--------|")
            
            for test_name, perf in performance_data.items():
                mean = perf['mean']
                stddev = perf['stddev']
                min_time = perf['min']
                max_time = perf['max']
                rounds = perf['rounds']
                
                # Performance status based on rough thresholds
                status = "✅" if mean < 1.0 else "⚠️" if mean < 5.0 else "❌"
                clean_name = test_name.replace('test_', '').replace('_', ' ').title()
                
                report.append(f"| {status} {clean_name} | {mean:.3f} | {stddev:.3f} | {min_time:.3f} | {max_time:.3f} | {rounds} |")
        else:
            report.append("No performance benchmarks available.")
        report.append("")
        
        # Quality Gates Status
        report.append("## Quality Gates")
        gates = []
        
        # Test success gate
        if test_results:
            total_failures = sum(r.get('failed', 0) + r.get('errors', 0) for r in test_results.values())
            test_gate_status = "✅" if total_failures == 0 else "❌"
            gates.append(f"- {test_gate_status} All tests pass: {total_failures == 0}")
        
        # Coverage gate
        if coverage_data and 'overall' in coverage_data:
            line_cov = coverage_data['overall']['line_coverage']
            cov_gate_status = "✅" if line_cov >= 85 else "❌"
            gates.append(f"- {cov_gate_status} Coverage threshold (85%): {line_cov:.1f}%")
        
        # Performance gate (if available)
        if performance_data:
            slow_benchmarks = sum(1 for p in performance_data.values() if p['mean'] > 5.0)
            perf_gate_status = "✅" if slow_benchmarks == 0 else "⚠️"
            gates.append(f"- {perf_gate_status} Performance acceptable: {slow_benchmarks} slow benchmarks")
        
        report.extend(gates)
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        recommendations = []
        
        if self.summary_data['overall_status'] != 'success':
            recommendations.append("- ❗ **Fix failing tests before merging**")
        
        if coverage_data and 'overall' in coverage_data:
            line_cov = coverage_data['overall']['line_coverage']
            if line_cov < 85:
                recommendations.append(f"- 📈 **Increase test coverage** from {line_cov:.1f}% to 85% minimum")
        
        if performance_data:
            slow_tests = [name for name, perf in performance_data.items() if perf['mean'] > 5.0]
            if slow_tests:
                recommendations.append(f"- ⚡ **Optimize slow benchmarks:** {', '.join(slow_tests[:3])}")
        
        if not recommendations:
            recommendations.append("- ✨ **All quality gates passed!** Ready for merge.")
        
        report.extend(recommendations)
        report.append("")
        
        # Artifacts Information
        report.append("## Available Artifacts")
        artifact_dirs = [d for d in self.artifacts_dir.iterdir() if d.is_dir()]
        for artifact_dir in sorted(artifact_dirs):
            files = list(artifact_dir.glob("*"))
            if files:
                report.append(f"- **{artifact_dir.name}:** {len(files)} files")
        
        return "\n".join(report)
    
    def save_summary(self):
        """Generate and save the CI summary."""
        summary_content = self.generate_summary()
        
        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        # Also save raw data as JSON for programmatic access
        json_file = self.output_file.with_suffix('.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.summary_data, f, indent=2, default=str)
        
        print(f"CI summary saved to: {self.output_file}")
        print(f"Raw data saved to: {json_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate CI summary report from test artifacts"
    )
    parser.add_argument(
        '--artifacts-dir',
        type=Path,
        required=True,
        help='Directory containing all test artifacts'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output markdown summary file path'
    )
    
    args = parser.parse_args()
    
    generator = CISummaryGenerator(args.artifacts_dir, args.output)
    generator.save_summary()


if __name__ == '__main__':
    main()
