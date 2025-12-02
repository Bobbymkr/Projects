"""
Week 11: Multi-Scenario Validation Library.

Comprehensive scenario library with 10 scenarios for validation.
"""

from typing import Dict, List, Any
from scenarios.scenario_library import SCENARIOS, get_scenario


class ScenarioValidator:
    """Validates agents across multiple scenarios."""
    
    def __init__(self):
        self.scenarios = SCENARIOS
        self.results = []
    
    def validate_agent(self, agent: Any, scenario_name: str, num_episodes: int = 30) -> Dict[str, Any]:
        """Validate agent on a specific scenario."""
        scenario = get_scenario(scenario_name)
        if not scenario:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        # Run validation
        episode_results = []
        for episode in range(num_episodes):
            # Run episode and collect metrics
            result = self._run_episode(agent, scenario)
            episode_results.append(result)
        
        # Calculate statistics
        wait_times = [r["wait_time"] for r in episode_results]
        throughputs = [r["throughput"] for r in episode_results]
        
        return {
            "scenario": scenario_name,
            "episodes": num_episodes,
            "avg_wait_time": sum(wait_times) / len(wait_times),
            "avg_throughput": sum(throughputs) / len(throughputs),
            "std_wait_time": self._std_dev(wait_times),
            "episode_results": episode_results
        }
    
    def validate_all_scenarios(self, agent: Any, num_episodes: int = 30) -> List[Dict[str, Any]]:
        """Validate agent on all scenarios."""
        results = []
        for scenario_name in self.scenarios.keys():
            result = self.validate_agent(agent, scenario_name, num_episodes)
            results.append(result)
        return results
    
    def _run_episode(self, agent: Any, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single episode."""
        # Implementation would run actual episode
        # This is a placeholder
        return {
            "wait_time": 12.0,
            "throughput": 500.0,
            "episode_length": 1000
        }
    
    def _std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

