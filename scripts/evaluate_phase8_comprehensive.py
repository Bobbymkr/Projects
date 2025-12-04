"""
Phase 8: Comprehensive Evaluation Framework

World-class evaluation as a top evaluator would perform:
- Multi-objective performance analysis
- Pareto front evaluation
- Constraint satisfaction verification
- Statistical significance testing
- Comparison with baselines
- Comprehensive reporting
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any
from datetime import datetime
import statistics

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research.multi_objective.phase8_multi_objective import (
    MOPPOAgent, MOPPOConfig,
    CPOAgent, ConstraintConfig,
    MultiObjectiveReward, MultiObjectiveWeights,
    NSGA2Solver,
    ConstraintOptimizer,
)
from src.env.traffic_env import TrafficEnv
from src.research.novel_algorithms.phase6_advanced_rl import PPOAgent, PPOConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """
    World-class comprehensive evaluator for Phase 8.
    
    Performs:
    - Multi-objective performance analysis
    - Pareto front evaluation
    - Constraint satisfaction verification
    - Statistical significance testing
    - Baseline comparison
    - Comprehensive reporting
    """
    
    def __init__(self, output_dir: Path):
        """Initialize evaluator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'evaluations': {},
        }
    
    def evaluate_multi_objective_performance(
        self,
        agent,
        env: TrafficEnv,
        reward_fn: MultiObjectiveReward,
        num_episodes: int = 100,
    ) -> Dict[str, Any]:
        """
        Evaluate multi-objective performance.
        
        Returns comprehensive performance metrics.
        """
        logger.info(f"Evaluating multi-objective performance ({num_episodes} episodes)...")
        
        episode_rewards = []
        episode_objectives = []
        constraint_violations = []
        
        constraint_optimizer = ConstraintOptimizer(ConstraintConfig())
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            obs = np.array(obs, dtype=np.float32).flatten()
            done = False
            
            wait_times = []
            queue_lengths = []
            vehicles_served = 0
            phase_changes = 0
            emergency_stops = 0
            violations = 0
            
            while not done:
                action, log_prob, values = agent.select_action(obs)
                next_obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_obs = np.array(next_obs, dtype=np.float32).flatten()
                
                # Collect statistics
                wait_times.extend(env.wait_times.tolist())
                queue_lengths.extend(env.queues.tolist())
                vehicles_served = info.get('total_vehicles_processed', 0)
                phase_changes += 1
                emergency_stops = 0  # Not tracked in current env
                
                # Check constraints
                green_time = env.green_values[action] if hasattr(env, 'green_values') else 10.0
                satisfied, penalty = constraint_optimizer.check_constraints(
                    action, green_time, np.array(wait_times), np.array(queue_lengths)
                )
                if not satisfied:
                    violations += 1
                
                obs = next_obs
            
            # Compute rewards
            obj_rewards = reward_fn.compute_rewards(
                wait_times=np.array(wait_times),
                queue_lengths=np.array(queue_lengths),
                vehicles_served=vehicles_served,
                phase_changes=phase_changes,
                emergency_stops=emergency_stops,
            )
            
            episode_rewards.append(obj_rewards['total'])
            episode_objectives.append([
                obj_rewards['wait_time'],
                obj_rewards['fuel_consumption'],
                obj_rewards['emissions'],
                obj_rewards['throughput'],
                obj_rewards['accidents'],
                obj_rewards['infrastructure_wear'],
            ])
            constraint_violations.append(violations / max(1, phase_changes))  # Normalize by steps
        
        # Statistical analysis
        metrics = {
            'mean_total_reward': float(np.mean(episode_rewards)),
            'std_total_reward': float(np.std(episode_rewards)),
            'min_total_reward': float(np.min(episode_rewards)),
            'max_total_reward': float(np.max(episode_rewards)),
            'median_total_reward': float(np.median(episode_rewards)),
            
            'mean_objectives': np.mean(episode_objectives, axis=0).tolist(),
            'std_objectives': np.std(episode_objectives, axis=0).tolist(),
            'min_objectives': np.min(episode_objectives, axis=0).tolist(),
            'max_objectives': np.max(episode_objectives, axis=0).tolist(),
            
            'constraint_violation_rate': float(np.mean(constraint_violations)),
            'constraint_satisfaction_rate': 1.0 - float(np.mean(constraint_violations)),
            
            'episode_rewards': episode_rewards,
            'episode_objectives': episode_objectives,
        }
        
        logger.info(f"Mean Total Reward: {metrics['mean_total_reward']:.2f} ± {metrics['std_total_reward']:.2f}")
        logger.info(f"Constraint Satisfaction: {metrics['constraint_satisfaction_rate']*100:.1f}%")
        
        return metrics
    
    def evaluate_pareto_front(
        self,
        objectives: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate Pareto front.
        
        Args:
            objectives: Array of objective values (n_solutions, n_objectives)
            
        Returns:
            Pareto front analysis
        """
        logger.info("Evaluating Pareto front...")
        
        solver = NSGA2Solver(num_objectives=objectives.shape[1])
        fronts = solver.non_dominated_sort(objectives)
        
        pareto_front = fronts[0] if fronts else []
        pareto_solutions = objectives[pareto_front] if pareto_front else objectives
        
        metrics = {
            'num_pareto_solutions': len(pareto_front),
            'pareto_front_size': len(pareto_front) / len(objectives),
            'pareto_objectives': pareto_solutions.tolist(),
            'num_fronts': len(fronts),
            'front_sizes': [len(f) for f in fronts],
        }
        
        logger.info(f"Pareto Front: {metrics['num_pareto_solutions']} solutions")
        logger.info(f"Number of Fronts: {metrics['num_fronts']}")
        
        return metrics
    
    def evaluate_constraint_satisfaction(
        self,
        agent,
        env: TrafficEnv,
        num_episodes: int = 100,
    ) -> Dict[str, Any]:
        """
        Evaluate constraint satisfaction.
        
        Returns:
            Constraint satisfaction metrics
        """
        logger.info(f"Evaluating constraint satisfaction ({num_episodes} episodes)...")
        
        constraint_optimizer = ConstraintOptimizer(ConstraintConfig())
        
        violations = {
            'min_green_time': 0,
            'max_wait_time': 0,
            'max_queue_length': 0,
            'total': 0,
        }
        
        total_checks = 0
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            obs = np.array(obs, dtype=np.float32).flatten()
            done = False
            
            while not done:
                action, _, _ = agent.select_action(obs)
                next_obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_obs = np.array(next_obs, dtype=np.float32).flatten()
                
                # Check constraints
                green_time = env.green_values[action] if hasattr(env, 'green_values') else 10.0
                wait_times = env.wait_times
                queue_lengths = env.queues
                
                satisfied, penalty = constraint_optimizer.check_constraints(
                    action, green_time, wait_times, queue_lengths
                )
                
                total_checks += 1
                if not satisfied:
                    violations['total'] += 1
                    if green_time < constraint_optimizer.config.min_green_time:
                        violations['min_green_time'] += 1
                    if len(wait_times) > 0 and np.max(wait_times) > constraint_optimizer.config.max_wait_time:
                        violations['max_wait_time'] += 1
                    if len(queue_lengths) > 0 and np.max(queue_lengths) > constraint_optimizer.config.max_queue_length:
                        violations['max_queue_length'] += 1
                
                obs = next_obs
        
        metrics = {
            'total_checks': total_checks,
            'violations': violations,
            'violation_rate': violations['total'] / total_checks if total_checks > 0 else 0.0,
            'satisfaction_rate': 1.0 - (violations['total'] / total_checks if total_checks > 0 else 0.0),
            'violation_breakdown': {
                'min_green_time': violations['min_green_time'] / total_checks if total_checks > 0 else 0.0,
                'max_wait_time': violations['max_wait_time'] / total_checks if total_checks > 0 else 0.0,
                'max_queue_length': violations['max_queue_length'] / total_checks if total_checks > 0 else 0.0,
            },
        }
        
        logger.info(f"Constraint Satisfaction Rate: {metrics['satisfaction_rate']*100:.2f}%")
        logger.info(f"Violation Rate: {metrics['violation_rate']*100:.2f}%")
        
        return metrics
    
    def compare_with_baseline(
        self,
        mo_agent,
        baseline_agent,
        env: TrafficEnv,
        reward_fn: MultiObjectiveReward,
        num_episodes: int = 100,
    ) -> Dict[str, Any]:
        """
        Compare multi-objective agent with baseline.
        
        Returns:
            Comparison metrics
        """
        logger.info("Comparing with baseline...")
        
        # Evaluate multi-objective agent
        mo_metrics = self.evaluate_multi_objective_performance(
            mo_agent, env, reward_fn, num_episodes
        )
        
        # Evaluate baseline (single-objective PPO)
        baseline_metrics = self.evaluate_multi_objective_performance(
            baseline_agent, env, reward_fn, num_episodes
        )
        
        # Compute improvements
        improvements = {}
        for key in ['mean_total_reward', 'mean_objectives']:
            if key in mo_metrics and key in baseline_metrics:
                if isinstance(mo_metrics[key], list):
                    improvements[key] = [
                        (mo - base) / abs(base) * 100 if base != 0 else 0
                        for mo, base in zip(mo_metrics[key], baseline_metrics[key])
                    ]
                else:
                    base = baseline_metrics[key]
                    mo = mo_metrics[key]
                    improvements[key] = (mo - base) / abs(base) * 100 if base != 0 else 0
        
        comparison = {
            'multi_objective': mo_metrics,
            'baseline': baseline_metrics,
            'improvements': improvements,
        }
        
        logger.info(f"Improvement in Total Reward: {improvements.get('mean_total_reward', 0):.2f}%")
        
        return comparison
    
    def statistical_significance_test(
        self,
        results1: List[float],
        results2: List[float],
    ) -> Dict[str, Any]:
        """
        Perform statistical significance test (t-test).
        
        Returns:
            Statistical test results
        """
        from scipy import stats
        
        try:
            t_stat, p_value = stats.ttest_ind(results1, results2)
            
            return {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'mean1': float(np.mean(results1)),
                'mean2': float(np.mean(results2)),
                'std1': float(np.std(results1)),
                'std2': float(np.std(results2)),
            }
        except ImportError:
            # Fallback if scipy not available
            mean1, mean2 = np.mean(results1), np.mean(results2)
            std1, std2 = np.std(results1), np.std(results2)
            
            # Simple z-test approximation
            z_score = (mean1 - mean2) / np.sqrt(std1**2 / len(results1) + std2**2 / len(results2))
            p_value = 2 * (1 - 0.5 * (1 + np.sign(z_score) * (1 - np.exp(-2 * z_score**2 / np.pi))))
            
            return {
                'z_score': float(z_score),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'mean1': float(mean1),
                'mean2': float(mean2),
                'std1': float(std1),
                'std2': float(std2),
            }
    
    def generate_comprehensive_report(
        self,
        agent_name: str,
        metrics: Dict[str, Any],
    ) -> str:
        """
        Generate comprehensive evaluation report.
        
        Returns:
            Report as string
        """
        report = f"""
================================================================================
Phase 8: Comprehensive Evaluation Report
================================================================================

Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Agent: {agent_name}

================================================================================
1. MULTI-OBJECTIVE PERFORMANCE
================================================================================

Total Reward:
  Mean: {metrics.get('mean_total_reward', 0):.4f}
  Std:  {metrics.get('std_total_reward', 0):.4f}
  Min:  {metrics.get('min_total_reward', 0):.4f}
  Max:  {metrics.get('max_total_reward', 0):.4f}
  Median: {metrics.get('median_total_reward', 0):.4f}

Objective Performance:
  1. Wait Time:        {metrics.get('mean_objectives', [0]*6)[0]:.4f} ± {metrics.get('std_objectives', [0]*6)[0]:.4f}
  2. Fuel Consumption: {metrics.get('mean_objectives', [0]*6)[1]:.4f} ± {metrics.get('std_objectives', [0]*6)[1]:.4f}
  3. Emissions:        {metrics.get('mean_objectives', [0]*6)[2]:.4f} ± {metrics.get('std_objectives', [0]*6)[2]:.4f}
  4. Throughput:       {metrics.get('mean_objectives', [0]*6)[3]:.4f} ± {metrics.get('std_objectives', [0]*6)[3]:.4f}
  5. Accidents:         {metrics.get('mean_objectives', [0]*6)[4]:.4f} ± {metrics.get('std_objectives', [0]*6)[4]:.4f}
  6. Infrastructure:    {metrics.get('mean_objectives', [0]*6)[5]:.4f} ± {metrics.get('std_objectives', [0]*6)[5]:.4f}

================================================================================
2. CONSTRAINT SATISFACTION
================================================================================

Constraint Satisfaction Rate: {metrics.get('constraint_satisfaction_rate', 0)*100:.2f}%
Constraint Violation Rate:    {metrics.get('constraint_violation_rate', 0)*100:.2f}%

================================================================================
3. STATISTICAL ANALYSIS
================================================================================

Sample Size: {len(metrics.get('episode_rewards', []))}
Confidence Level: 95%

================================================================================
4. RECOMMENDATIONS
================================================================================

"""
        
        # Add recommendations based on metrics
        if metrics.get('constraint_satisfaction_rate', 0) < 0.95:
            report += "WARNING: Constraint satisfaction rate below 95%. Consider adjusting constraints.\n"
        
        if metrics.get('mean_total_reward', 0) < 0:
            report += "WARNING: Negative average reward. Consider reward function tuning.\n"
        
        report += "\n================================================================================\n"
        
        return report
    
    def save_results(self, filename: str = "evaluation_results.json"):
        """Save evaluation results."""
        results_path = self.output_dir / filename
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 8: Comprehensive Evaluation")
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["MO-PPO", "CPO"],
        help="Algorithm to evaluate"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/intersection.json",
        help="Environment configuration"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./evaluations/phase8",
        help="Output directory"
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare with baseline PPO"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        env_config = json.load(f)
    
    # Create environment
    env = TrafficEnv(config=env_config)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create reward function
    reward_fn = MultiObjectiveReward()
    
    # Create agent
    if args.algorithm == "MO-PPO":
        agent = MOPPOAgent(state_dim, action_dim, MOPPOConfig())
    elif args.algorithm == "CPO":
        agent = CPOAgent(
            state_dim, action_dim,
            ConstraintConfig(),
            MOPPOConfig(),
        )
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(args.output)
    
    # Evaluate
    logger.info("="*80)
    logger.info("Starting Comprehensive Evaluation")
    logger.info("="*80)
    
    # 1. Multi-objective performance
    metrics = evaluator.evaluate_multi_objective_performance(
        agent, env, reward_fn, args.episodes
    )
    evaluator.results['evaluations']['multi_objective'] = metrics
    
    # 2. Constraint satisfaction
    constraint_metrics = evaluator.evaluate_constraint_satisfaction(
        agent, env, args.episodes
    )
    evaluator.results['evaluations']['constraints'] = constraint_metrics
    
    # 3. Pareto front (if applicable)
    if 'episode_objectives' in metrics:
        pareto_metrics = evaluator.evaluate_pareto_front(
            np.array(metrics['episode_objectives'])
        )
        evaluator.results['evaluations']['pareto_front'] = pareto_metrics
    
    # 4. Baseline comparison
    if args.compare_baseline:
        baseline_agent = PPOAgent(state_dim, action_dim, PPOConfig())
        comparison = evaluator.compare_with_baseline(
            agent, baseline_agent, env, reward_fn, args.episodes
        )
        evaluator.results['evaluations']['baseline_comparison'] = comparison
    
    # Generate report
    report = evaluator.generate_comprehensive_report(args.algorithm, metrics)
    
    # Save results
    evaluator.save_results()
    
    # Save report
    report_path = Path(args.output) / "evaluation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info("\n" + report)
    logger.info(f"\nReport saved to: {report_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

