"""
Enhanced Load Testing Suite.

Week 3: Comprehensive load testing scenarios:
- Gradual load increase (0 → 1000 req/s)
- Spike test (sudden 10x traffic)
- Soak test (24-hour sustained load)
"""

import time
import statistics
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import requests
import threading
import concurrent.futures

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


class LoadTestRunner:
    """Enhanced load testing runner."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
    
    def make_request(self, endpoint: str, method: str = "GET", data: dict = None) -> Dict[str, Any]:
        """Make a single request and measure latency."""
        start_time = time.time()
        
        try:
            if method == "GET":
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
            elif method == "POST":
                response = requests.post(f"{self.base_url}{endpoint}", json=data, timeout=5)
            else:
                return {"error": f"Unsupported method: {method}"}
            
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                "endpoint": endpoint,
                "method": method,
                "status_code": response.status_code,
                "latency_ms": latency,
                "success": 200 <= response.status_code < 300,
                "timestamp": time.time()
            }
        
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return {
                "endpoint": endpoint,
                "method": method,
                "error": str(e),
                "latency_ms": latency,
                "success": False,
                "timestamp": time.time()
            }
    
    def gradual_load_increase(self, max_rps: int = 1000, duration_seconds: int = 300):
        """Gradual load increase test (0 → max_rps)."""
        print(f"\n📈 Gradual Load Increase Test")
        print(f"   Target: 0 → {max_rps} req/s over {duration_seconds}s")
        print("=" * 80)
        
        start_time = time.time()
        results = []
        
        while time.time() - start_time < duration_seconds:
            elapsed = time.time() - start_time
            current_rps = int((max_rps / duration_seconds) * elapsed)
            
            # Make requests at current rate
            requests_this_second = []
            request_start = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(current_rps, 100)) as executor:
                futures = []
                for _ in range(current_rps):
                    future = executor.submit(
                        self.make_request,
                        "/api/v1/traffic/decision",
                        "POST",
                        {
                            "intersection_id": "test",
                            "queue_lengths": [5, 8, 3, 6],
                            "wait_times": [10, 15, 8, 12],
                            "current_phase": 0
                        }
                    )
                    futures.append(future)
                
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    requests_this_second.append(result)
            
            # Wait until next second
            elapsed_in_second = time.time() - request_start
            if elapsed_in_second < 1.0:
                time.sleep(1.0 - elapsed_in_second)
            
            # Calculate metrics
            if requests_this_second:
                successful = sum(1 for r in requests_this_second if r.get("success"))
                latencies = [r["latency_ms"] for r in requests_this_second if "latency_ms" in r]
                
                results.append({
                    "time_elapsed": elapsed,
                    "target_rps": current_rps,
                    "actual_rps": len(requests_this_second),
                    "successful": successful,
                    "failed": len(requests_this_second) - successful,
                    "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
                    "p95_latency_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0] if latencies else 0,
                    "max_latency_ms": max(latencies) if latencies else 0
                })
                
                print(f"  {elapsed:.1f}s: {current_rps} req/s, "
                      f"Success: {successful}/{len(requests_this_second)}, "
                      f"Latency: {statistics.mean(latencies):.2f}ms (p95: {statistics.quantiles(latencies, n=20)[18]:.2f}ms)" if latencies else "N/A")
        
        return results
    
    def spike_test(self, base_rps: int = 100, spike_multiplier: int = 10, duration_seconds: int = 60):
        """Spike test: sudden 10x traffic increase."""
        print(f"\n⚡ Spike Test")
        print(f"   Base: {base_rps} req/s → Spike: {base_rps * spike_multiplier} req/s")
        print("=" * 80)
        
        results = []
        
        # Phase 1: Baseline (30 seconds)
        print("Phase 1: Baseline load...")
        baseline_results = self._run_constant_load(base_rps, duration_seconds=30)
        results.extend(baseline_results)
        
        # Phase 2: Spike (10 seconds)
        print(f"Phase 2: Spike ({base_rps * spike_multiplier} req/s)...")
        spike_results = self._run_constant_load(base_rps * spike_multiplier, duration_seconds=10)
        results.extend(spike_results)
        
        # Phase 3: Recovery (20 seconds)
        print("Phase 3: Recovery...")
        recovery_results = self._run_constant_load(base_rps, duration_seconds=20)
        results.extend(recovery_results)
        
        return results
    
    def _run_constant_load(self, rps: int, duration_seconds: int) -> List[Dict[str, Any]]:
        """Run constant load for specified duration."""
        results = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            second_start = time.time()
            requests_this_second = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(rps, 100)) as executor:
                futures = []
                for _ in range(rps):
                    future = executor.submit(
                        self.make_request,
                        "/api/v1/traffic/decision",
                        "POST",
                        {
                            "intersection_id": "test",
                            "queue_lengths": [5, 8, 3, 6],
                            "wait_times": [10, 15, 8, 12],
                            "current_phase": 0
                        }
                    )
                    futures.append(future)
                
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    requests_this_second.append(result)
            
            elapsed = time.time() - start_time
            latencies = [r["latency_ms"] for r in requests_this_second if "latency_ms" in r]
            
            if requests_this_second:
                results.append({
                    "time_elapsed": elapsed,
                    "target_rps": rps,
                    "actual_rps": len(requests_this_second),
                    "successful": sum(1 for r in requests_this_second if r.get("success")),
                    "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
                    "p95_latency_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0] if latencies else 0
                })
            
            # Wait until next second
            elapsed_in_second = time.time() - second_start
            if elapsed_in_second < 1.0:
                time.sleep(1.0 - elapsed_in_second)
        
        return results
    
    def soak_test(self, rps: int = 100, duration_hours: int = 24):
        """Soak test: sustained load for extended period."""
        print(f"\n💧 Soak Test")
        print(f"   Load: {rps} req/s for {duration_hours} hours")
        print("=" * 80)
        print("Note: This is a long-running test. Use --short for testing.")
        
        duration_seconds = duration_hours * 3600
        return self._run_constant_load(rps, duration_seconds)
    
    def generate_report(self, results: List[Dict[str, Any]], test_name: str) -> Dict[str, Any]:
        """Generate load test report."""
        if not results:
            return {}
        
        latencies = [r.get("avg_latency_ms", 0) for r in results if "avg_latency_ms" in r]
        p95_latencies = [r.get("p95_latency_ms", 0) for r in results if "p95_latency_ms" in r]
        success_rates = [
            r.get("successful", 0) / max(r.get("actual_rps", 1), 1) * 100
            for r in results
            if r.get("actual_rps", 0) > 0
        ]
        
        report = {
            "test_name": test_name,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_samples": len(results),
                "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
                "p95_latency_ms": statistics.mean(p95_latencies) if p95_latencies else 0,
                "max_latency_ms": max(latencies) if latencies else 0,
                "min_latency_ms": min(latencies) if latencies else 0,
                "avg_success_rate": statistics.mean(success_rates) if success_rates else 0,
                "min_success_rate": min(success_rates) if success_rates else 0
            },
            "results": results
        }
        
        return report


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced load testing")
    parser.add_argument("--test", choices=["gradual", "spike", "soak"], default="gradual", help="Test type")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL")
    parser.add_argument("--max-rps", type=int, default=1000, help="Max requests per second")
    parser.add_argument("--duration", type=int, default=300, help="Duration in seconds")
    parser.add_argument("--short", action="store_true", help="Short test (for development)")
    parser.add_argument("--output", type=Path, help="Output JSON file")
    
    args = parser.parse_args()
    
    runner = LoadTestRunner(args.url)
    
    if args.short:
        args.duration = 10  # Short test
    
    if args.test == "gradual":
        results = runner.gradual_load_increase(args.max_rps, args.duration)
    elif args.test == "spike":
        results = runner.spike_test(base_rps=args.max_rps // 10, spike_multiplier=10, duration_seconds=args.duration)
    elif args.test == "soak":
        duration_hours = 1 if args.short else 24
        results = runner.soak_test(rps=args.max_rps // 10, duration_hours=duration_hours)
    
    # Generate report
    report = runner.generate_report(results, args.test)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ Report saved to {args.output}")
    else:
        output_file = PROJECT_ROOT / "results" / "load_tests" / f"load_test_{args.test}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ Report saved to {output_file}")
    
    # Print summary
    if report.get("summary"):
        summary = report["summary"]
        print("\n📊 Load Test Summary")
        print("=" * 80)
        print(f"Average Latency: {summary['avg_latency_ms']:.2f}ms")
        print(f"P95 Latency: {summary['p95_latency_ms']:.2f}ms")
        print(f"Max Latency: {summary['max_latency_ms']:.2f}ms")
        print(f"Average Success Rate: {summary['avg_success_rate']:.2f}%")


if __name__ == "__main__":
    main()

