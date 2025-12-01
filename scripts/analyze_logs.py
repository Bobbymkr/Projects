"""
Log Analysis Script.

Analyze application logs for errors, patterns, and performance issues.
"""

import argparse
import re
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter, defaultdict
from datetime import datetime
import json


class LogAnalyzer:
    """Analyze application logs."""
    
    def __init__(self, log_dir: Path):
        """Initialize log analyzer."""
        self.log_dir = log_dir
        self.error_patterns = [
            r"ERROR",
            r"Exception",
            r"Traceback",
            r"Failed",
            r"Error:",
        ]
        self.warning_patterns = [
            r"WARNING",
            r"Warning:",
        ]
    
    def find_log_files(self) -> List[Path]:
        """Find all log files in directory."""
        log_files = []
        for pattern in ["*.log", "*.txt"]:
            log_files.extend(self.log_dir.glob(pattern))
            log_files.extend(self.log_dir.glob(f"**/{pattern}"))
        return sorted(set(log_files))
    
    def parse_log_line(self, line: str) -> Dict[str, Any]:
        """Parse a single log line."""
        # Try to parse structured JSON logs
        try:
            return json.loads(line)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Parse standard log format
        # Format: TIMESTAMP LEVEL MODULE: MESSAGE
        timestamp_match = re.search(r'\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}', line)
        level_match = re.search(r'\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b', line)
        
        return {
            "raw": line,
            "timestamp": timestamp_match.group() if timestamp_match else None,
            "level": level_match.group() if level_match else "UNKNOWN",
            "message": line,
        }
    
    def analyze_errors(self, log_files: List[Path]) -> Dict[str, Any]:
        """Analyze errors in log files."""
        errors = []
        error_counts = Counter()
        error_types = defaultdict(int)
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        parsed = self.parse_log_line(line.strip())
                        
                        # Check for errors
                        if any(re.search(pattern, line, re.IGNORECASE) for pattern in self.error_patterns):
                            errors.append({
                                "file": str(log_file),
                                "line": line_num,
                                "level": parsed.get("level", "UNKNOWN"),
                                "message": parsed.get("message", line)[:200],
                            })
                            error_counts[parsed.get("level", "UNKNOWN")] += 1
                            
                            # Categorize error
                            if "Exception" in line or "Traceback" in line:
                                error_types["Exception"] += 1
                            elif "Timeout" in line:
                                error_types["Timeout"] += 1
                            elif "Connection" in line:
                                error_types["Connection"] += 1
                            else:
                                error_types["Other"] += 1
            except Exception as e:
                print(f"Error reading {log_file}: {e}")
        
        return {
            "total_errors": len(errors),
            "error_counts": dict(error_counts),
            "error_types": dict(error_types),
            "errors": errors[:100],  # Limit to first 100
        }
    
    def analyze_warnings(self, log_files: List[Path]) -> Dict[str, Any]:
        """Analyze warnings in log files."""
        warnings = []
        warning_counts = Counter()
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if any(re.search(pattern, line, re.IGNORECASE) for pattern in self.warning_patterns):
                            warnings.append({
                                "file": str(log_file),
                                "line": line_num,
                                "message": line.strip()[:200],
                            })
                            warning_counts["WARNING"] += 1
            except Exception as e:
                print(f"Error reading {log_file}: {e}")
        
        return {
            "total_warnings": len(warnings),
            "warning_counts": dict(warning_counts),
            "warnings": warnings[:100],
        }
    
    def analyze_performance(self, log_files: List[Path]) -> Dict[str, Any]:
        """Analyze performance metrics from logs."""
        performance_metrics = []
        latencies = []
        
        # Look for timing information
        latency_pattern = re.compile(r'(\d+\.?\d*)\s*(ms|s)', re.IGNORECASE)
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        # Extract latencies
                        matches = latency_pattern.findall(line)
                        for value, unit in matches:
                            try:
                                latency = float(value)
                                if unit.lower() == 's':
                                    latency *= 1000  # Convert to ms
                                if 0 < latency < 10000:  # Reasonable range
                                    latencies.append(latency)
                            except ValueError:
                                pass
            except Exception as e:
                print(f"Error reading {log_file}: {e}")
        
        if latencies:
            latencies.sort()
            return {
                "count": len(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "avg": sum(latencies) / len(latencies),
                "p50": latencies[len(latencies) // 2],
                "p95": latencies[int(len(latencies) * 0.95)],
                "p99": latencies[int(len(latencies) * 0.99)],
            }
        return {}
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive log analysis report."""
        log_files = self.find_log_files()
        
        if not log_files:
            return {"error": "No log files found"}
        
        print(f"Analyzing {len(log_files)} log files...")
        
        report = {
            "analysis_date": datetime.now().isoformat(),
            "log_files_analyzed": len(log_files),
            "errors": self.analyze_errors(log_files),
            "warnings": self.analyze_warnings(log_files),
            "performance": self.analyze_performance(log_files),
        }
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted report."""
        print("\n" + "="*60)
        print("LOG ANALYSIS REPORT")
        print("="*60)
        
        print(f"\nLog Files Analyzed: {report.get('log_files_analyzed', 0)}")
        
        # Errors
        errors = report.get("errors", {})
        print(f"\nErrors Found: {errors.get('total_errors', 0)}")
        if errors.get("error_counts"):
            print("  By Level:")
            for level, count in errors["error_counts"].items():
                print(f"    {level}: {count}")
        if errors.get("error_types"):
            print("  By Type:")
            for error_type, count in errors["error_types"].items():
                print(f"    {error_type}: {count}")
        
        # Warnings
        warnings = report.get("warnings", {})
        print(f"\nWarnings Found: {warnings.get('total_warnings', 0)}")
        
        # Performance
        perf = report.get("performance", {})
        if perf:
            print("\nPerformance Metrics:")
            print(f"  Latency Count: {perf.get('count', 0)}")
            if perf.get('avg'):
                print(f"  Average: {perf['avg']:.2f} ms")
                print(f"  P50: {perf.get('p50', 0):.2f} ms")
                print(f"  P95: {perf.get('p95', 0):.2f} ms")
                print(f"  P99: {perf.get('p99', 0):.2f} ms")
        
        print("\n" + "="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze application logs")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing log files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON report file"
    )
    
    args = parser.parse_args()
    
    analyzer = LogAnalyzer(args.log_dir)
    report = analyzer.generate_report()
    
    analyzer.print_report(report)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()

