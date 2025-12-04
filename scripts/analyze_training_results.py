"""
Analyze Training Results and Identify Best Algorithm.

Compares all algorithms based on multiple metrics and provides
a comprehensive ranking and analysis.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(results_path: Path) -> Dict:
    """Load training results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def analyze_results(results: Dict) -> Dict:
    """Analyze results and rank algorithms."""
    analysis = {
        "rankings": {},
        "best_overall": None,
        "best_by_metric": {},
        "summary": {}
    }
    
    # Extract metrics
    algorithms = list(results.keys())
    
    # 1. Average Reward Ranking (higher is better, but these are negative, so less negative is better)
    avg_rewards = {alg: results[alg]["avg_reward"] for alg in algorithms}
    avg_reward_ranking = sorted(avg_rewards.items(), key=lambda x: x[1], reverse=True)
    analysis["rankings"]["avg_reward"] = avg_reward_ranking
    analysis["best_by_metric"]["avg_reward"] = avg_reward_ranking[0][0]
    
    # 2. Final Reward Ranking (last 10 episodes - higher is better)
    final_rewards = {alg: results[alg]["final_reward"] for alg in algorithms}
    final_reward_ranking = sorted(final_rewards.items(), key=lambda x: x[1], reverse=True)
    analysis["rankings"]["final_reward"] = final_reward_ranking
    analysis["best_by_metric"]["final_reward"] = final_reward_ranking[0][0]
    
    # 3. Stability Ranking (lower std dev is better)
    std_devs = {alg: results[alg]["std_reward"] for alg in algorithms}
    stability_ranking = sorted(std_devs.items(), key=lambda x: x[1])
    analysis["rankings"]["stability"] = stability_ranking
    analysis["best_by_metric"]["stability"] = stability_ranking[0][0]
    
    # 4. Episode Length (shorter episodes might indicate better efficiency)
    avg_lengths = {alg: results[alg]["avg_length"] for alg in algorithms}
    length_ranking = sorted(avg_lengths.items(), key=lambda x: x[1])
    analysis["rankings"]["efficiency"] = length_ranking
    analysis["best_by_metric"]["efficiency"] = length_ranking[0][0]
    
    # 5. Overall Score (weighted combination)
    # Normalize each metric to 0-1 scale, then combine
    normalized_scores = {}
    
    # Normalize average reward (less negative = better)
    min_avg = min(avg_rewards.values())
    max_avg = max(avg_rewards.values())
    range_avg = max_avg - min_avg if max_avg != min_avg else 1
    
    # Normalize final reward (less negative = better)
    min_final = min(final_rewards.values())
    max_final = max(final_rewards.values())
    range_final = max_final - min_final if max_final != min_final else 1
    
    # Normalize stability (lower std = better)
    min_std = min(std_devs.values())
    max_std = max(std_devs.values())
    range_std = max_std - min_std if max_std != min_std else 1
    
    # Calculate composite score for each algorithm
    for alg in algorithms:
        # Average reward score (higher is better, normalized)
        avg_score = (avg_rewards[alg] - min_avg) / range_avg
        
        # Final reward score (higher is better, normalized)
        final_score = (final_rewards[alg] - min_final) / range_final
        
        # Stability score (lower std is better, inverted)
        stability_score = 1 - ((std_devs[alg] - min_std) / range_std)
        
        # Weighted combination (40% avg, 40% final, 20% stability)
        composite_score = (0.4 * avg_score) + (0.4 * final_score) + (0.2 * stability_score)
        normalized_scores[alg] = composite_score
    
    overall_ranking = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
    analysis["rankings"]["overall"] = overall_ranking
    analysis["best_overall"] = overall_ranking[0][0]
    
    # Summary statistics
    analysis["summary"] = {
        "total_algorithms": len(algorithms),
        "best_avg_reward": {
            "algorithm": avg_reward_ranking[0][0],
            "value": avg_reward_ranking[0][1]
        },
        "best_final_reward": {
            "algorithm": final_reward_ranking[0][0],
            "value": final_reward_ranking[0][1]
        },
        "most_stable": {
            "algorithm": stability_ranking[0][0],
            "value": stability_ranking[0][1]
        },
        "most_efficient": {
            "algorithm": length_ranking[0][0],
            "value": length_ranking[0][1]
        },
        "overall_best": {
            "algorithm": overall_ranking[0][0],
            "score": overall_ranking[0][1]
        }
    }
    
    return analysis


def generate_report(results: Dict, analysis: Dict, output_path: Path = None) -> str:
    """Generate a comprehensive markdown report."""
    report = []
    report.append("# Algorithm Performance Analysis Report\n")
    report.append(f"**Analysis Date:** {Path(__file__).stat().st_mtime}\n")
    report.append("---\n")
    
    # Executive Summary
    report.append("## 🏆 Executive Summary\n")
    best = analysis["best_overall"]
    best_data = results[best]
    report.append(f"**Best Overall Algorithm:** `{best}`\n")
    report.append(f"- Average Reward: `{best_data['avg_reward']:.2f}`")
    report.append(f"- Final Reward: `{best_data['final_reward']:.2f}`")
    report.append(f"- Stability (Std Dev): `{best_data['std_reward']:.2f}`")
    report.append(f"- Average Episode Length: `{best_data['avg_length']:.2f}`\n")
    
    # Overall Ranking
    report.append("## 📊 Overall Ranking (Composite Score)\n")
    report.append("| Rank | Algorithm | Composite Score | Avg Reward | Final Reward | Stability |")
    report.append("|------|-----------|----------------|------------|--------------|-----------|")
    
    for rank, (alg, score) in enumerate(analysis["rankings"]["overall"], 1):
        alg_data = results[alg]
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else ""
        report.append(
            f"| {rank} {medal} | **{alg}** | {score:.4f} | "
            f"{alg_data['avg_reward']:.2f} | {alg_data['final_reward']:.2f} | "
            f"{alg_data['std_reward']:.2f} |"
        )
    report.append("")
    
    # Best by Metric
    report.append("## 🎯 Best Algorithm by Metric\n")
    report.append("| Metric | Best Algorithm | Value |")
    report.append("|--------|----------------|-------|")
    
    metrics_info = {
        "avg_reward": ("Average Reward", "Higher is better"),
        "final_reward": ("Final Reward (Last 10 Episodes)", "Higher is better"),
        "stability": ("Stability (Lower Std Dev)", "Lower is better"),
        "efficiency": ("Efficiency (Episode Length)", "Shorter is better")
    }
    
    for metric_key, (metric_name, note) in metrics_info.items():
        best_alg = analysis["best_by_metric"][metric_key]
        value = results[best_alg][metric_key.replace("_reward", "_reward").replace("stability", "std_reward").replace("efficiency", "avg_length")]
        report.append(f"| {metric_name} | **{best_alg}** | {value:.2f} |")
    report.append("")
    
    # Detailed Rankings
    report.append("## 📈 Detailed Rankings\n")
    
    # Average Reward
    report.append("### Average Reward Ranking\n")
    report.append("| Rank | Algorithm | Average Reward |")
    report.append("|------|-----------|----------------|")
    for rank, (alg, value) in enumerate(analysis["rankings"]["avg_reward"], 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else ""
        report.append(f"| {rank} {medal} | {alg} | {value:.2f} |")
    report.append("")
    
    # Final Reward
    report.append("### Final Reward Ranking (Last 10 Episodes)\n")
    report.append("| Rank | Algorithm | Final Reward |")
    report.append("|------|-----------|--------------|")
    for rank, (alg, value) in enumerate(analysis["rankings"]["final_reward"], 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else ""
        report.append(f"| {rank} {medal} | {alg} | {value:.2f} |")
    report.append("")
    
    # Stability
    report.append("### Stability Ranking (Lower Standard Deviation = Better)\n")
    report.append("| Rank | Algorithm | Std Deviation |")
    report.append("|------|-----------|---------------|")
    for rank, (alg, value) in enumerate(analysis["rankings"]["stability"], 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else ""
        report.append(f"| {rank} {medal} | {alg} | {value:.2f} |")
    report.append("")
    
    # All Results Table
    report.append("## 📋 Complete Results Table\n")
    report.append("| Algorithm | Episodes | Avg Reward | Final Reward | Std Dev | Avg Length | Status |")
    report.append("|-----------|----------|------------|-------------|---------|------------|--------|")
    
    for alg in sorted(results.keys()):
        data = results[alg]
        status_icon = "✅" if data["status"] == "success" else "❌"
        report.append(
            f"| {alg} | {data['episodes']} | {data['avg_reward']:.2f} | "
            f"{data['final_reward']:.2f} | {data['std_reward']:.2f} | "
            f"{data['avg_length']:.2f} | {status_icon} {data['status']} |"
        )
    report.append("")
    
    # Recommendations
    report.append("## 💡 Recommendations\n")
    report.append("### For Production Deployment:\n")
    report.append(f"1. **Primary Choice:** `{analysis['best_overall']}` - Best overall performance")
    report.append(f"2. **For Stability:** `{analysis['best_by_metric']['stability']}` - Most consistent results")
    report.append(f"3. **For Learning Speed:** `{analysis['best_by_metric']['final_reward']}` - Best final performance")
    report.append("")
    report.append("### For Further Investigation:\n")
    report.append("- Consider ensemble methods combining top 3 algorithms\n")
    report.append("- Investigate hyperparameter tuning for top performers\n")
    report.append("- Test on different traffic scenarios to validate robustness\n")
    
    report_text = "\n".join(report)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"Report saved to: {output_path}")
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description="Analyze training results")
    parser.add_argument(
        "--results",
        type=str,
        default="runs/all_technologies/training_results.json",
        help="Path to training results JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/all_technologies/algorithm_analysis_report.md",
        help="Path to output report file"
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    output_path = Path(args.output)
    
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return 1
    
    # Load and analyze
    print(f"Loading results from: {results_path}")
    results = load_results(results_path)
    
    print("Analyzing results...")
    analysis = analyze_results(results)
    
    print("Generating report...")
    report = generate_report(results, analysis, output_path)
    
    # Print summary to console
    print("\n" + "="*80)
    print("QUICK SUMMARY")
    print("="*80)
    print(f"Best Overall Algorithm: {analysis['best_overall']}")
    print(f"  - Average Reward: {results[analysis['best_overall']]['avg_reward']:.2f}")
    print(f"  - Final Reward: {results[analysis['best_overall']]['final_reward']:.2f}")
    print(f"  - Stability: {results[analysis['best_overall']]['std_reward']:.2f}")
    print("\nTop 3 Algorithms:")
    for rank, (alg, score) in enumerate(analysis["rankings"]["overall"][:3], 1):
        print(f"  {rank}. {alg} (score: {score:.4f})")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

