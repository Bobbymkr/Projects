#!/usr/bin/env python3
"""
Generate system test KPI report from JUnit XML results.

This script parses system test results and generates a markdown report
with key performance indicators and baseline comparisons.
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
import json
import re
from typing import Dict, List, Optional, Tuple


class SystemTestReportGenerator:
    """Generate system test reports from JUnit XML results."""
    
    def __init__(self, input_file: Path, output_file: Path):
        self.input_file = input_file
        self.output_file = output_file
        self.test_results = {}
        self.kpis = {}
        
    def parse_junit_xml(self) -> Dict:
        """Parse JUnit XML file and extract test results."""
        if not self.input_file.exists():
            print(f"Warning: Input file {self.input_file} does not exist")
            return {}
            
        try:
            tree = ET.parse(self.input_file)
            root = tree.getroot()
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'total_tests': int(root.get('tests', 0)),
                'failures': int(root.get('failures', 0)),
                'errors': int(root.get('errors', 0)),
                'skipped': int(root.get('skipped', 0)),
                'time': float(root.get('time', 0)),
                'testcases': []
            }
            
            for testcase in root.findall('.//testcase'):
                test_data = {
                    'name': testcase.get('name'),
                    'classname': testcase.get('classname'),
                    'time': float(testcase.get('time', 0)),
                    'status': 'passed'
                }
                
                # Check for failures or errors
                if testcase.find('failure') is not None:
                    test_data['status'] = 'failed'
                    test_data['failure'] = testcase.find('failure').text
                elif testcase.find('error') is not None:
                    test_data['status'] = 'error'
                    test_data['error'] = testcase.find('error').text
                elif testcase.find('skipped') is not None:
                    test_data['status'] = 'skipped'
                    
                # Extract KPI data from test properties if available
                properties = testcase.find('properties')
                if properties is not None:
                    test_data['kpis'] = {}
                    for prop in properties.findall('property'):
                        key = prop.get('name')
                        value = prop.get('value')
                        if key and value:
                            # Try to parse numeric values
                            try:
                                test_data['kpis'][key] = float(value)
                            except ValueError:
                                test_data['kpis'][key] = value
                
                results['testcases'].append(test_data)
                
            return results
            
        except ET.ParseError as e:
            print(f"Error parsing XML file: {e}")
            return {}
    
    def extract_kpis(self, results: Dict) -> Dict:
        """Extract and aggregate KPIs from test results."""
        kpis = {
            'scenario_results': {},
            'performance_metrics': {},
            'baseline_comparisons': {}
        }
        
        for test in results.get('testcases', []):
            test_name = test['name']
            test_kpis = test.get('kpis', {})
            
            # Group KPIs by scenario
            scenario_match = re.match(r'test_(.+?)_scenario', test_name)
            if scenario_match:
                scenario = scenario_match.group(1)
                if scenario not in kpis['scenario_results']:
                    kpis['scenario_results'][scenario] = {
                        'status': test['status'],
                        'execution_time': test['time'],
                        'metrics': test_kpis
                    }
            
            # Extract performance metrics
            for key, value in test_kpis.items():
                if isinstance(value, (int, float)):
                    if 'performance' in key.lower() or 'throughput' in key.lower():
                        kpis['performance_metrics'][key] = value
                    elif 'baseline' in key.lower() or 'comparison' in key.lower():
                        kpis['baseline_comparisons'][key] = value
        
        return kpis
    
    def generate_report(self) -> str:
        """Generate markdown report from parsed results."""
        results = self.parse_junit_xml()
        if not results:
            return "# System Test Report\n\nNo test results found.\n"
        
        kpis = self.extract_kpis(results)
        
        report = []
        report.append("# System Test KPI Report")
        report.append(f"**Generated:** {results['timestamp']}")
        report.append(f"**Total Tests:** {results['total_tests']}")
        report.append(f"**Execution Time:** {results['time']:.2f}s")
        report.append("")
        
        # Overall Status
        report.append("## Overall Status")
        passed = results['total_tests'] - results['failures'] - results['errors'] - results['skipped']
        pass_rate = (passed / results['total_tests'] * 100) if results['total_tests'] > 0 else 0
        
        status_icon = "✅" if results['failures'] == 0 and results['errors'] == 0 else "❌"
        report.append(f"{status_icon} **Pass Rate:** {pass_rate:.1f}% ({passed}/{results['total_tests']})")
        report.append(f"- **Passed:** {passed}")
        report.append(f"- **Failed:** {results['failures']}")
        report.append(f"- **Errors:** {results['errors']}")
        report.append(f"- **Skipped:** {results['skipped']}")
        report.append("")
        
        # Scenario Results
        if kpis['scenario_results']:
            report.append("## Scenario Results")
            report.append("| Scenario | Status | Time (s) | Key Metrics |")
            report.append("|----------|--------|----------|-------------|")
            
            for scenario, data in kpis['scenario_results'].items():
                status_icon = "✅" if data['status'] == 'passed' else "❌"
                metrics_summary = self._format_metrics_summary(data['metrics'])
                report.append(f"| {scenario} | {status_icon} {data['status']} | {data['execution_time']:.2f} | {metrics_summary} |")
            report.append("")
        
        # Performance Metrics
        if kpis['performance_metrics']:
            report.append("## Performance Metrics")
            for metric, value in kpis['performance_metrics'].items():
                report.append(f"- **{metric}:** {value}")
            report.append("")
        
        # Baseline Comparisons
        if kpis['baseline_comparisons']:
            report.append("## Baseline Comparisons")
            report.append("| Metric | Current | Baseline | Change | Status |")
            report.append("|--------|---------|----------|--------|--------|")
            
            for metric, data in kpis['baseline_comparisons'].items():
                if isinstance(data, dict) and 'current' in data:
                    current = data['current']
                    baseline = data.get('baseline', 0)
                    change = ((current - baseline) / baseline * 100) if baseline != 0 else 0
                    status = "⚠️" if abs(change) > 5 else "✅"
                    report.append(f"| {metric} | {current:.2f} | {baseline:.2f} | {change:+.1f}% | {status} |")
            report.append("")
        
        # Failed Tests Details
        failed_tests = [t for t in results['testcases'] if t['status'] in ['failed', 'error']]
        if failed_tests:
            report.append("## Failed Tests")
            for test in failed_tests:
                report.append(f"### {test['name']}")
                report.append(f"**Class:** {test['classname']}")
                report.append(f"**Status:** {test['status']}")
                report.append(f"**Time:** {test['time']:.2f}s")
                
                if 'failure' in test:
                    report.append("**Failure:**")
                    report.append("```")
                    report.append(test['failure'])
                    report.append("```")
                elif 'error' in test:
                    report.append("**Error:**")
                    report.append("```")
                    report.append(test['error'])
                    report.append("```")
                report.append("")
        
        # System Requirements Validation
        report.append("## System Requirements")
        report.append(self._validate_system_requirements(kpis))
        
        return "\\n".join(report)
    
    def _format_metrics_summary(self, metrics: Dict) -> str:
        """Format metrics dictionary into a summary string."""
        if not metrics:
            return "N/A"
        
        key_metrics = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if 'delay' in key.lower():
                    key_metrics.append(f"Delay: {value:.1f}s")
                elif 'throughput' in key.lower():
                    key_metrics.append(f"Throughput: {value:.0f}")
                elif 'queue' in key.lower():
                    key_metrics.append(f"Queue: {value:.0f}")
        
        return ", ".join(key_metrics[:3]) if key_metrics else "No metrics"
    
    def _validate_system_requirements(self, kpis: Dict) -> str:
        """Validate system performance against defined requirements."""
        requirements = []
        
        # Example requirements from strategy document
        perf_metrics = kpis.get('performance_metrics', {})
        
        # Check policy inference time
        if 'policy_inference_ms' in perf_metrics:
            inference_time = perf_metrics['policy_inference_ms']
            status = "✅" if inference_time < 10 else "❌"
            requirements.append(f"- Policy Inference (<10ms): {status} {inference_time:.1f}ms")
        
        # Check environment step time
        if 'env_step_ms' in perf_metrics:
            step_time = perf_metrics['env_step_ms']
            status = "✅" if step_time < 5 else "❌"
            requirements.append(f"- Environment Step (<5ms): {status} {step_time:.1f}ms")
        
        # Check vision processing FPS
        if 'vision_fps' in perf_metrics:
            fps = perf_metrics['vision_fps']
            status = "✅" if fps > 30 else "❌"
            requirements.append(f"- Vision Processing (>30 FPS): {status} {fps:.0f} FPS")
        
        # Check episode success rate
        success_rates = []
        for scenario, data in kpis.get('scenario_results', {}).items():
            if 'success_rate' in data.get('metrics', {}):
                success_rates.append(data['metrics']['success_rate'])
        
        if success_rates:
            avg_success = sum(success_rates) / len(success_rates)
            status = "✅" if avg_success > 95 else "❌"
            requirements.append(f"- Episode Success Rate (>95%): {status} {avg_success:.1f}%")
        
        return "\\n".join(requirements) if requirements else "No performance requirements validated."
    
    def save_report(self):
        """Generate and save the report to file."""
        report_content = self.generate_report()
        
        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"System test report saved to: {self.output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate system test KPI report from JUnit XML results"
    )
    parser.add_argument(
        '--input', 
        type=Path, 
        required=True,
        help='Input JUnit XML file path'
    )
    parser.add_argument(
        '--output', 
        type=Path, 
        required=True,
        help='Output markdown report file path'
    )
    
    args = parser.parse_args()
    
    generator = SystemTestReportGenerator(args.input, args.output)
    generator.save_report()


if __name__ == '__main__':
    main()
