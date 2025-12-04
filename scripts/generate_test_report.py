"""
Automated Test Report Generator.

Generates comprehensive test reports with coverage, performance,
and quality metrics.
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import xml.etree.ElementTree as ET


class TestReportGenerator:
    """Generate comprehensive test reports."""
    
    def __init__(self, output_dir: Path = Path("reports")):
        """Initialize report generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate complete test report."""
        print("Generating test report...")
        
        # Run tests and collect data
        test_results = self._run_tests()
        coverage_data = self._get_coverage()
        performance_data = self._get_performance_metrics()
        
        # Compile report
        report = {
            "timestamp": datetime.now().isoformat(),
            "test_results": test_results,
            "coverage": coverage_data,
            "performance": performance_data,
            "summary": self._generate_summary(test_results, coverage_data),
        }
        
        # Save report
        report_file = self.output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate HTML report
        html_report = self._generate_html_report(report)
        html_file = self.output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(html_file, 'w') as f:
            f.write(html_report)
        
        print(f"Report generated: {report_file}")
        print(f"HTML report: {html_file}")
        
        return report
    
    def _run_tests(self) -> Dict[str, Any]:
        """Run tests and collect results."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "--json-report", "--json-report-file=test_results.json"],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            # Parse results
            if Path("test_results.json").exists():
                with open("test_results.json") as f:
                    return json.load(f)
            else:
                return {
                    "passed": result.returncode == 0,
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_coverage(self) -> Dict[str, Any]:
        """Get coverage data."""
        try:
            # Run coverage
            subprocess.run(
                [sys.executable, "-m", "pytest", "--cov=src", "--cov-report=xml", "--cov-report=term", "tests/"],
                capture_output=True,
                timeout=600
            )
            
            # Parse coverage XML
            if Path("coverage.xml").exists():
                tree = ET.parse("coverage.xml")
                root = tree.getroot()
                
                return {
                    "line_rate": float(root.attrib.get("line-rate", 0)) * 100,
                    "branch_rate": float(root.attrib.get("branch-rate", 0)) * 100,
                }
        except Exception as e:
            return {"error": str(e)}
        
        return {}
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/performance/", "--benchmark-only", "--benchmark-json=benchmark.json"],
                capture_output=True,
                timeout=300
            )
            
            if Path("benchmark.json").exists():
                with open("benchmark.json") as f:
                    return json.load(f)
        except Exception as e:
            return {"error": str(e)}
        
        return {}
    
    def _generate_summary(
        self,
        test_results: Dict[str, Any],
        coverage: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            "total_tests": test_results.get("summary", {}).get("total", 0),
            "passed": test_results.get("summary", {}).get("passed", 0),
            "failed": test_results.get("summary", {}).get("failed", 0),
            "coverage": coverage.get("line_rate", 0),
            "status": "PASS" if test_results.get("passed", False) else "FAIL",
        }
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report."""
        summary = report.get("summary", {})
        coverage = report.get("coverage", {})
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Test Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #4CAF50; color: white; padding: 20px; }}
        .summary {{ margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f0f0f0; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Test Report</h1>
        <p>Generated: {report.get('timestamp', 'Unknown')}</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <div class="metric">
            <strong>Total Tests:</strong> {summary.get('total_tests', 0)}
        </div>
        <div class="metric">
            <strong>Passed:</strong> <span class="pass">{summary.get('passed', 0)}</span>
        </div>
        <div class="metric">
            <strong>Failed:</strong> <span class="fail">{summary.get('failed', 0)}</span>
        </div>
        <div class="metric">
            <strong>Coverage:</strong> {coverage.get('line_rate', 0):.2f}%
        </div>
        <div class="metric">
            <strong>Status:</strong> <span class="{summary.get('status', 'FAIL').lower()}">{summary.get('status', 'FAIL')}</span>
        </div>
    </div>
</body>
</html>
"""
        return html


if __name__ == "__main__":
    generator = TestReportGenerator()
    report = generator.generate_report()
    print("\nReport Summary:")
    print(json.dumps(report.get("summary", {}), indent=2))

