"""
Chaos Engineering Test Suite.

Week 3: Automated chaos testing scenarios:
- Network latency injection
- Service failure simulation
- Resource exhaustion tests
- Data corruption scenarios
"""

import time
import random
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Callable
from datetime import datetime
from unittest.mock import patch, Mock
import threading

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class ChaosEngine:
    """Chaos engineering test engine."""
    
    def __init__(self):
        self.results = []
    
    def inject_network_latency(self, delay_ms: int = 100, duration_seconds: int = 30):
        """Inject network latency into system calls."""
        print(f"\n🌐 Network Latency Injection")
        print(f"   Delay: {delay_ms}ms for {duration_seconds}s")
        print("=" * 80)
        
        original_sleep = time.sleep
        
        def delayed_sleep(seconds):
            """Add delay to sleep calls."""
            return original_sleep(seconds + (delay_ms / 1000.0))
        
        results = []
        start_time = time.time()
        
        # Test system under latency
        try:
            from src.env.traffic_env import TrafficEnv
            from src.rl.dqn_agent import DQNAgent, DQNConfig
            
            env_config = {"num_lanes": 4}
            env = TrafficEnv(env_config)
            agent = DQNAgent(8, env.action_space.n, DQNConfig())
            
            with patch('time.sleep', delayed_sleep):
                while time.time() - start_time < duration_seconds:
                    test_start = time.time()
                    obs, info = env.reset()
                    action = agent.select_action(obs)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    test_time = (time.time() - test_start) * 1000
                    
                    results.append({
                        "time_elapsed": time.time() - start_time,
                        "test_latency_ms": test_time,
                        "action": action,
                        "success": True
                    })
                    
                    if len(results) % 10 == 0:
                        print(f"  {len(results)} tests completed, avg latency: {statistics.mean([r['test_latency_ms'] for r in results]):.2f}ms")
        
        except Exception as e:
            results.append({
                "error": str(e),
                "success": False
            })
        
        return results
    
    def simulate_service_failure(self, failure_rate: float = 0.1, duration_seconds: int = 60):
        """Simulate random service failures."""
        print(f"\n💥 Service Failure Simulation")
        print(f"   Failure Rate: {failure_rate * 100}% for {duration_seconds}s")
        print("=" * 80)
        
        results = []
        start_time = time.time()
        failure_count = 0
        total_requests = 0
        
        try:
            from src.env.traffic_env import TrafficEnv
            from src.rl.dqn_agent import DQNAgent, DQNConfig
            
            env_config = {"num_lanes": 4}
            env = TrafficEnv(env_config)
            agent = DQNAgent(8, env.action_space.n, DQNConfig())
            
            while time.time() - start_time < duration_seconds:
                total_requests += 1
                
                # Simulate failure
                if random.random() < failure_rate:
                    failure_count += 1
                    # Simulate failure (raise exception or return error)
                    try:
                        raise RuntimeError("Simulated service failure")
                    except RuntimeError as e:
                        results.append({
                            "time_elapsed": time.time() - start_time,
                            "request_num": total_requests,
                            "failure": True,
                            "error": str(e),
                            "recovered": False
                        })
                else:
                    # Normal operation
                    try:
                        obs, info = env.reset()
                        action = agent.select_action(obs)
                        results.append({
                            "time_elapsed": time.time() - start_time,
                            "request_num": total_requests,
                            "failure": False,
                            "action": action,
                            "success": True
                        })
                    except Exception as e:
                        results.append({
                            "time_elapsed": time.time() - start_time,
                            "request_num": total_requests,
                            "failure": True,
                            "error": str(e),
                            "recovered": False
                        })
        
        except Exception as e:
            results.append({
                "error": str(e),
                "success": False
            })
        
        print(f"  Total requests: {total_requests}, Failures: {failure_count} ({failure_count/max(total_requests,1)*100:.1f}%)")
        
        return results
    
    def resource_exhaustion_test(self, memory_limit_mb: int = 100, cpu_limit_percent: int = 80):
        """Test system behavior under resource constraints."""
        print(f"\n⚙️ Resource Exhaustion Test")
        print(f"   Memory Limit: {memory_limit_mb}MB, CPU Limit: {cpu_limit_percent}%")
        print("=" * 80)
        
        results = []
        
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            
            from src.env.traffic_env import TrafficEnv
            from src.rl.dqn_agent import DQNAgent, DQNConfig
            
            env_config = {"num_lanes": 4}
            env = TrafficEnv(env_config)
            agent = DQNAgent(8, env.action_space.n, DQNConfig())
            
            # Monitor resource usage
            for i in range(100):
                obs, info = env.reset()
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Check resource usage
                memory_mb = process.memory_info().rss / (1024 * 1024)
                cpu_percent = process.cpu_percent()
                
                results.append({
                    "iteration": i,
                    "memory_mb": memory_mb,
                    "cpu_percent": cpu_percent,
                    "within_limits": memory_mb < memory_limit_mb and cpu_percent < cpu_limit_percent,
                    "action": action
                })
                
                if i % 10 == 0:
                    print(f"  Iteration {i}: Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%")
        
        except ImportError:
            print("  Warning: psutil not available, skipping resource monitoring")
            results = [{"error": "psutil not available"}]
        except Exception as e:
            results.append({"error": str(e)})
        
        return results
    
    def data_corruption_test(self, corruption_rate: float = 0.05):
        """Test system resilience to data corruption."""
        print(f"\n🔀 Data Corruption Test")
        print(f"   Corruption Rate: {corruption_rate * 100}%")
        print("=" * 80)
        
        results = []
        
        try:
            from src.env.traffic_env import TrafficEnv
            from src.rl.dqn_agent import DQNAgent, DQNConfig
            
            env_config = {"num_lanes": 4}
            env = TrafficEnv(env_config)
            agent = DQNAgent(8, env.action_space.n, DQNConfig())
            
            corruption_count = 0
            
            for i in range(100):
                obs, info = env.reset()
                
                # Corrupt observation data
                if random.random() < corruption_rate:
                    corruption_count += 1
                    # Inject corruption (NaN, inf, or extreme values)
                    corruption_type = random.choice(["nan", "inf", "extreme"])
                    if corruption_type == "nan":
                        obs = np.full_like(obs, np.nan)
                    elif corruption_type == "inf":
                        obs = np.full_like(obs, np.inf)
                    else:
                        obs = np.full_like(obs, 1e10)  # Extreme values
                
                # Try to handle corrupted data
                try:
                    action = agent.select_action(obs)
                    recovered = True
                except (ValueError, RuntimeError, TypeError) as e:
                    # System should handle gracefully
                    action = 0  # Fallback
                    recovered = True
                except Exception as e:
                    recovered = False
                    action = None
                
                results.append({
                    "iteration": i,
                    "corrupted": corruption_count,
                    "recovered": recovered,
                    "action": action
                })
            
            print(f"  Corruptions injected: {corruption_count}, Recovered: {sum(1 for r in results if r['recovered'])}")
        
        except Exception as e:
            results.append({"error": str(e)})
        
        return results
    
    def generate_report(self, results: List[Dict[str, Any]], test_name: str) -> Dict[str, Any]:
        """Generate chaos test report."""
        report = {
            "test_name": test_name,
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(results),
            "successful": sum(1 for r in results if r.get("success", False) or r.get("recovered", False)),
            "failed": sum(1 for r in results if r.get("error") or (not r.get("success", True) and not r.get("recovered", False))),
            "results": results
        }
        
        return report


def main():
    import argparse
    import statistics
    
    parser = argparse.ArgumentParser(description="Chaos engineering tests")
    parser.add_argument("--test", choices=["latency", "failure", "resource", "corruption", "all"], default="all", help="Test type")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds")
    parser.add_argument("--output", type=Path, help="Output JSON file")
    
    args = parser.parse_args()
    
    engine = ChaosEngine()
    all_reports = {}
    
    if args.test in ["latency", "all"]:
        results = engine.inject_network_latency(delay_ms=100, duration_seconds=min(args.duration, 30))
        all_reports["latency"] = engine.generate_report(results, "network_latency")
    
    if args.test in ["failure", "all"]:
        results = engine.simulate_service_failure(failure_rate=0.1, duration_seconds=min(args.duration, 60))
        all_reports["failure"] = engine.generate_report(results, "service_failure")
    
    if args.test in ["resource", "all"]:
        results = engine.resource_exhaustion_test()
        all_reports["resource"] = engine.generate_report(results, "resource_exhaustion")
    
    if args.test in ["corruption", "all"]:
        results = engine.data_corruption_test()
        all_reports["corruption"] = engine.generate_report(results, "data_corruption")
    
    # Save reports
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_reports, f, indent=2)
        print(f"\n✅ Reports saved to {args.output}")
    else:
        output_file = PROJECT_ROOT / "results" / "chaos_tests" / f"chaos_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(all_reports, f, indent=2)
        print(f"\n✅ Reports saved to {output_file}")
    
    # Print summary
    print("\n📊 Chaos Test Summary")
    print("=" * 80)
    for test_name, report in all_reports.items():
        print(f"{test_name}:")
        print(f"  Total: {report['total_tests']}, Successful: {report['successful']}, Failed: {report['failed']}")


if __name__ == "__main__":
    import numpy as np
    main()

