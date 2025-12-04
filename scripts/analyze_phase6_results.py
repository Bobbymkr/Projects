"""
Comprehensive Analysis of Phase 6 Training Results.

Analyzes training results for PPO, SAC, and Rainbow DQN algorithms,
provides detailed comparisons, visualizations, and recommendations.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visualizations will be skipped.")


class Phase6ResultsAnalyzer:
    """Analyze Phase 6 training results."""
    
    def __init__(self, results_path: Path, output_dir: Path):
        """
        Initialize analyzer.
        
        Args:
            results_path: Path to training_results.json
            output_dir: Output directory for analysis results
        """
        self.results_path = results_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        
        # Baseline from OPTIMIZATION_ROADMAP.md
        self.baseline_reward = -107.81
        self.target_reward = -70.0  # Conservative target
    
    def analyze_algorithm(self, alg_name: str, data: Dict) -> Dict[str, Any]:
        """Analyze individual algorithm results."""
        if data.get("status") == "error":
            return {
                "status": "error",
                "error": data.get("error", "Unknown error")
            }
        
        analysis = {
            "algorithm": alg_name,
            "episodes": data.get("episodes", 0),
            "metrics": {
                "avg_reward": data.get("avg_reward", 0),
                "std_reward": data.get("std_reward", 0),
                "final_reward": data.get("final_reward", 0),
                "best_reward": data.get("best_reward", 0),
                "final_eval_reward": data.get("final_eval_reward", 0),
                "avg_length": data.get("avg_length", 0),
                "training_time": data.get("training_time", 0),
            },
            "improvements": {},
            "convergence": {},
        }
        
        # Calculate improvements vs baseline
        avg_reward = analysis["metrics"]["avg_reward"]
        final_reward = analysis["metrics"]["final_reward"]
        
        analysis["improvements"] = {
            "vs_baseline_avg": ((self.baseline_reward - avg_reward) / abs(self.baseline_reward)) * 100,
            "vs_baseline_final": ((self.baseline_reward - final_reward) / abs(self.baseline_reward)) * 100,
            "vs_target_avg": ((self.target_reward - avg_reward) / abs(self.target_reward)) * 100,
            "vs_target_final": ((self.target_reward - final_reward) / abs(self.target_reward)) * 100,
        }
        
        # Convergence analysis
        episode_rewards = data.get("episode_rewards", [])
        if len(episode_rewards) > 100:
            # Check if converged (last 20% of episodes have stable reward)
            last_20_percent = int(len(episode_rewards) * 0.2)
            recent_rewards = episode_rewards[-last_20_percent:]
            early_rewards = episode_rewards[:last_20_percent] if len(episode_rewards) > last_20_percent else episode_rewards[:10]
            
            recent_std = np.std(recent_rewards)
            recent_mean = np.mean(recent_rewards)
            early_mean = np.mean(early_rewards)
            
            improvement = early_mean - recent_mean  # Negative rewards, so improvement is positive
            convergence_threshold = 0.01 * abs(recent_mean)  # 1% of mean value
            
            analysis["convergence"] = {
                "converged": recent_std < convergence_threshold,
                "recent_std": float(recent_std),
                "recent_mean": float(recent_mean),
                "early_mean": float(early_mean),
                "improvement": float(improvement),
                "convergence_episode": self._find_convergence_episode(episode_rewards),
            }
        
        return analysis
    
    def _find_convergence_episode(self, rewards: List[float], window: int = 100) -> int:
        """Find episode where convergence occurred."""
        if len(rewards) < window * 2:
            return len(rewards)
        
        for i in range(window, len(rewards) - window):
            recent = rewards[i:i+window]
            prev = rewards[i-window:i]
            
            recent_std = np.std(recent)
            recent_mean = np.mean(recent)
            
            # Check if variance is low and mean is stable
            if recent_std < 0.05 * abs(recent_mean):
                return i
        
        return len(rewards)
    
    def compare_algorithms(self) -> Dict[str, Any]:
        """Compare all algorithms."""
        comparisons = {
            "rankings": {},
            "best_by_metric": {},
            "overall_best": None,
            "comparison_table": [],
        }
        
        valid_results = {k: v for k, v in self.results.items() if v.get("status") != "error"}
        
        if not valid_results:
            return comparisons
        
        algorithms = list(valid_results.keys())
        
        # 1. Average Reward Ranking (higher/less negative is better)
        avg_rewards = {alg: valid_results[alg]["avg_reward"] for alg in algorithms}
        avg_ranking = sorted(avg_rewards.items(), key=lambda x: x[1], reverse=True)
        comparisons["rankings"]["avg_reward"] = avg_ranking
        comparisons["best_by_metric"]["avg_reward"] = avg_ranking[0][0]
        
        # 2. Final Reward Ranking
        final_rewards = {alg: valid_results[alg]["final_reward"] for alg in algorithms}
        final_ranking = sorted(final_rewards.items(), key=lambda x: x[1], reverse=True)
        comparisons["rankings"]["final_reward"] = final_ranking
        comparisons["best_by_metric"]["final_reward"] = final_ranking[0][0]
        
        # 3. Stability Ranking (lower std is better)
        std_devs = {alg: valid_results[alg]["std_reward"] for alg in algorithms}
        stability_ranking = sorted(std_devs.items(), key=lambda x: x[1])
        comparisons["rankings"]["stability"] = stability_ranking
        comparisons["best_by_metric"]["stability"] = stability_ranking[0][0]
        
        # 4. Final Eval Reward Ranking
        eval_rewards = {alg: valid_results[alg].get("final_eval_reward", -1000) for alg in algorithms}
        eval_ranking = sorted(eval_rewards.items(), key=lambda x: x[1], reverse=True)
        comparisons["rankings"]["final_eval"] = eval_ranking
        comparisons["best_by_metric"]["final_eval"] = eval_ranking[0][0]
        
        # 5. Training Efficiency (reward per time)
        training_times = {alg: valid_results[alg].get("training_time", 1) for alg in algorithms}
        efficiency = {alg: final_rewards[alg] / max(training_times[alg], 1) for alg in algorithms}
        efficiency_ranking = sorted(efficiency.items(), key=lambda x: x[1], reverse=True)
        comparisons["rankings"]["efficiency"] = efficiency_ranking
        comparisons["best_by_metric"]["efficiency"] = efficiency_ranking[0][0]
        
        # 6. Overall Score (weighted combination)
        normalized_scores = {}
        
        # Normalize metrics
        min_avg = min(avg_rewards.values())
        max_avg = max(avg_rewards.values())
        range_avg = max_avg - min_avg if max_avg != min_avg else 1
        
        min_final = min(final_rewards.values())
        max_final = max(final_rewards.values())
        range_final = max_final - min_final if max_final != min_final else 1
        
        min_std = min(std_devs.values())
        max_std = max(std_devs.values())
        range_std = max_std - min_std if max_std != min_std else 1
        
        for alg in algorithms:
            avg_score = (avg_rewards[alg] - min_avg) / range_avg
            final_score = (final_rewards[alg] - min_final) / range_final
            stability_score = 1 - ((std_devs[alg] - min_std) / range_std)
            eval_score = (eval_rewards[alg] - min(eval_rewards.values())) / (max(eval_rewards.values()) - min(eval_rewards.values()) + 1e-8)
            
            # Weighted: 30% avg, 30% final, 20% stability, 20% eval
            composite = (0.3 * avg_score) + (0.3 * final_score) + (0.2 * stability_score) + (0.2 * eval_score)
            normalized_scores[alg] = composite
        
        overall_ranking = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        comparisons["rankings"]["overall"] = overall_ranking
        comparisons["overall_best"] = overall_ranking[0][0]
        
        # Create comparison table
        for alg in algorithms:
            comparisons["comparison_table"].append({
                "algorithm": alg,
                "avg_reward": avg_rewards[alg],
                "final_reward": final_rewards[alg],
                "std_reward": std_devs[alg],
                "final_eval": eval_rewards[alg],
                "training_time": training_times[alg],
                "overall_score": normalized_scores[alg],
            })
        
        return comparisons
    
    def generate_visualizations(self):
        """Generate visualization plots."""
        if not HAS_MATPLOTLIB:
            return
        
        valid_results = {k: v for k, v in self.results.items() if v.get("status") != "error"}
        
        # 1. Learning Curves
        fig, ax = plt.subplots(figsize=(12, 6))
        for alg_name, data in valid_results.items():
            episode_rewards = data.get("episode_rewards", [])
            if episode_rewards:
                # Smooth with moving average
                window = max(1, len(episode_rewards) // 50)
                smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
                episodes = np.arange(len(smoothed))
                ax.plot(episodes, smoothed, label=alg_name, alpha=0.7, linewidth=2)
        
        ax.axhline(y=self.baseline_reward, color='r', linestyle='--', label='Baseline', linewidth=2)
        ax.axhline(y=self.target_reward, color='g', linestyle='--', label='Target', linewidth=2)
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title('Phase 6 Learning Curves', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "learning_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Performance Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        algorithms = list(valid_results.keys())
        avg_rewards = [valid_results[alg]["avg_reward"] for alg in algorithms]
        final_rewards = [valid_results[alg]["final_reward"] for alg in algorithms]
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        ax.bar(x - width/2, avg_rewards, width, label='Average Reward', alpha=0.8)
        ax.bar(x + width/2, final_rewards, width, label='Final Reward', alpha=0.8)
        ax.axhline(y=self.baseline_reward, color='r', linestyle='--', label='Baseline')
        ax.axhline(y=self.target_reward, color='g', linestyle='--', label='Target')
        
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title('Phase 6 Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Improvement vs Baseline
        fig, ax = plt.subplots(figsize=(10, 6))
        improvements = []
        for alg in algorithms:
            avg_reward = valid_results[alg]["avg_reward"]
            improvement = ((self.baseline_reward - avg_reward) / abs(self.baseline_reward)) * 100
            improvements.append(improvement)
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax.bar(algorithms, improvements, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Improvement (%)', fontsize=12)
        ax.set_title('Improvement vs Baseline', fontsize=14, fontweight='bold')
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{imp:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "improvement_vs_baseline.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        report = {
            "analysis_date": datetime.now().isoformat(),
            "baseline_reward": self.baseline_reward,
            "target_reward": self.target_reward,
            "algorithms": {},
            "comparisons": {},
            "recommendations": {},
        }
        
        # Analyze each algorithm
        for alg_name, data in self.results.items():
            report["algorithms"][alg_name] = self.analyze_algorithm(alg_name, data)
        
        # Compare algorithms
        report["comparisons"] = self.compare_algorithms()
        
        # Generate recommendations
        report["recommendations"] = self.generate_recommendations(report)
        
        return report
    
    def generate_recommendations(self, report: Dict) -> Dict[str, Any]:
        """Generate recommendations based on results."""
        recommendations = {
            "best_algorithm": None,
            "deployment_ready": [],
            "needs_improvement": [],
            "next_steps": [],
        }
        
        comparisons = report["comparisons"]
        if comparisons.get("overall_best"):
            recommendations["best_algorithm"] = comparisons["overall_best"]
        
        # Check which algorithms meet target
        for alg_name, analysis in report["algorithms"].items():
            if analysis.get("status") == "error":
                recommendations["needs_improvement"].append({
                    "algorithm": alg_name,
                    "reason": f"Training error: {analysis.get('error', 'Unknown')}"
                })
                continue
            
            metrics = analysis.get("metrics", {})
            final_reward = metrics.get("final_reward", -1000)
            final_eval = metrics.get("final_eval_reward", -1000)
            
            # Check if meets target
            if final_reward <= self.target_reward or final_eval <= self.target_reward:
                recommendations["deployment_ready"].append({
                    "algorithm": alg_name,
                    "final_reward": final_reward,
                    "final_eval": final_eval,
                })
            else:
                recommendations["needs_improvement"].append({
                    "algorithm": alg_name,
                    "reason": f"Final reward ({final_reward:.2f}) above target ({self.target_reward:.2f})",
                    "suggestion": "Consider more training episodes or hyperparameter tuning"
                })
        
        # Next steps
        if recommendations["deployment_ready"]:
            recommendations["next_steps"].append(
                f"Deploy {recommendations['best_algorithm']} for production testing"
            )
        else:
            recommendations["next_steps"].append(
                "Continue training or optimize hyperparameters to reach target"
            )
        
        if recommendations["needs_improvement"]:
            recommendations["next_steps"].append(
                "Run hyperparameter optimization for underperforming algorithms"
            )
        
        recommendations["next_steps"].append(
            "Compare Phase 6 results with Phase 5 ensemble methods"
        )
        recommendations["next_steps"].append(
            "Consider combining best Phase 6 algorithms in ensemble"
        )
        
        return recommendations
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        return obj
    
    def save_report(self, report: Dict[str, Any]):
        """Save analysis report."""
        # Save JSON report (convert numpy types first)
        json_path = self.output_dir / "analysis_report.json"
        serializable_report = self._make_json_serializable(report)
        with open(json_path, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        # Save human-readable summary
        summary_path = self.output_dir / "analysis_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Phase 6 Training Results Analysis\n")
            f.write("="*80 + "\n\n")
            f.write(f"Analysis Date: {report['analysis_date']}\n")
            f.write(f"Baseline Reward: {report['baseline_reward']:.2f}\n")
            f.write(f"Target Reward: {report['target_reward']:.2f}\n\n")
            
            f.write("="*80 + "\n")
            f.write("Algorithm Performance\n")
            f.write("="*80 + "\n\n")
            
            for alg_name, analysis in report["algorithms"].items():
                if analysis.get("status") == "error":
                    f.write(f"{alg_name}: ERROR - {analysis.get('error', 'Unknown')}\n\n")
                    continue
                
                metrics = analysis["metrics"]
                improvements = analysis["improvements"]
                
                f.write(f"{alg_name}:\n")
                f.write(f"  Average Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}\n")
                f.write(f"  Final Reward: {metrics['final_reward']:.2f}\n")
                f.write(f"  Best Reward: {metrics['best_reward']:.2f}\n")
                f.write(f"  Final Eval Reward: {metrics['final_eval_reward']:.2f}\n")
                f.write(f"  Training Time: {metrics['training_time']:.2f}s\n")
                f.write(f"  Improvement vs Baseline: {improvements['vs_baseline_avg']:.1f}%\n")
                f.write(f"  Improvement vs Target: {improvements['vs_target_avg']:.1f}%\n")
                
                if analysis.get("convergence"):
                    conv = analysis["convergence"]
                    f.write(f"  Converged: {conv['converged']}\n")
                    if conv.get("convergence_episode"):
                        f.write(f"  Convergence Episode: {conv['convergence_episode']}\n")
                f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("Algorithm Rankings\n")
            f.write("="*80 + "\n\n")
            
            comparisons = report["comparisons"]
            f.write("Best by Average Reward: {}\n".format(
                comparisons["best_by_metric"].get("avg_reward", "N/A")
            ))
            f.write("Best by Final Reward: {}\n".format(
                comparisons["best_by_metric"].get("final_reward", "N/A")
            ))
            f.write("Best by Stability: {}\n".format(
                comparisons["best_by_metric"].get("stability", "N/A")
            ))
            f.write("Best by Final Eval: {}\n".format(
                comparisons["best_by_metric"].get("final_eval", "N/A")
            ))
            f.write(f"\nOverall Best: {comparisons.get('overall_best', 'N/A')}\n\n")
            
            f.write("="*80 + "\n")
            f.write("Recommendations\n")
            f.write("="*80 + "\n\n")
            
            recs = report["recommendations"]
            if recs.get("best_algorithm"):
                f.write(f"Best Algorithm: {recs['best_algorithm']}\n\n")
            
            if recs.get("deployment_ready"):
                f.write("Deployment Ready:\n")
                for item in recs["deployment_ready"]:
                    f.write(f"  - {item['algorithm']} (Final: {item['final_reward']:.2f}, Eval: {item['final_eval']:.2f})\n")
                f.write("\n")
            
            if recs.get("needs_improvement"):
                f.write("Needs Improvement:\n")
                for item in recs["needs_improvement"]:
                    f.write(f"  - {item['algorithm']}: {item.get('reason', 'N/A')}\n")
                f.write("\n")
            
            if recs.get("next_steps"):
                f.write("Next Steps:\n")
                for step in recs["next_steps"]:
                    f.write(f"  - {step}\n")
        
        print(f"Analysis report saved to: {json_path}")
        print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 6 training results")
    parser.add_argument(
        "--results",
        type=str,
        default="./runs/phase6_full_scale/training_results.json",
        help="Path to training_results.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./runs/phase6_full_scale/analysis",
        help="Output directory for analysis"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots"
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return 1
    
    analyzer = Phase6ResultsAnalyzer(results_path, Path(args.output))
    
    print("Analyzing Phase 6 training results...")
    report = analyzer.generate_report()
    
    analyzer.save_report(report)
    
    if not args.no_plots and HAS_MATPLOTLIB:
        print("Generating visualizations...")
        analyzer.generate_visualizations()
        print(f"Visualizations saved to: {Path(args.output)}")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nBest Algorithm: {report['comparisons'].get('overall_best', 'N/A')}")
    print(f"\nSee {Path(args.output)}/analysis_summary.txt for detailed results")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

