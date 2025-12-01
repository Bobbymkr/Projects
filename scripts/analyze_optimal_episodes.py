"""
Optimal Episode Count Analysis Script

Analyzes convergence patterns for each technology to determine optimal training episode counts.
Tracks reward/performance metrics over episodes and identifies optimal stopping points.
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from collections import deque

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.traffic_env import TrafficEnv

# Import all technologies
try:
    from src.research.novel_algorithms.hierarchical_rl_complete import HierarchicalRLAgent
except ImportError:
    HierarchicalRLAgent = None

try:
    from src.research.novel_algorithms.model_based_rl_complete import ModelBasedRLAgent
except ImportError:
    ModelBasedRLAgent = None

try:
    from src.research.novel_algorithms.imitation_learning_complete import (
        BehavioralCloningAgent,
        DAggerAgent,
        HybridILRLAgent,
    )
except ImportError:
    BehavioralCloningAgent = None
    DAggerAgent = None
    HybridILRLAgent = None

try:
    from src.research.novel_algorithms.transformer_control import TransformerAgent
except ImportError:
    TransformerAgent = None

try:
    from src.research.novel_algorithms.bayesian_methods import BayesianAgent
except ImportError:
    BayesianAgent = None

try:
    from src.research.novel_algorithms.causal_inference import CausalAgent
except ImportError:
    CausalAgent = None

try:
    from src.research.novel_algorithms.neuro_symbolic import NeuroSymbolicAgent
except ImportError:
    NeuroSymbolicAgent = None

try:
    from src.research.novel_algorithms.meta_learning import MAMLAgent, ReptileAgent
except ImportError:
    MAMLAgent = None
    ReptileAgent = None

try:
    from src.research.novel_algorithms.llm_traffic import LLMTrafficAgent
except ImportError:
    LLMTrafficAgent = None

try:
    from src.research.novel_algorithms.diffusion_models import DiffusionTrafficAgent
except ImportError:
    DiffusionTrafficAgent = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConvergenceAnalyzer:
    """Analyzes convergence patterns for RL agents."""
    
    def __init__(self, window_size: int = 50, convergence_threshold: float = 0.01):
        """
        Initialize convergence analyzer.
        
        Args:
            window_size: Number of episodes to consider for convergence
            convergence_threshold: Relative change threshold for convergence
        """
        self.window_size = window_size
        self.convergence_threshold = convergence_threshold
    
    def analyze_convergence(
        self, 
        rewards: List[float], 
        episode_numbers: List[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze convergence pattern from reward history.
        
        Args:
            rewards: List of episode rewards
            episode_numbers: Optional list of episode numbers
            
        Returns:
            Dictionary with convergence analysis results
        """
        if len(rewards) < self.window_size:
            return {
                "converged": False,
                "convergence_episode": None,
                "reason": "Insufficient data"
            }
        
        rewards = np.array(rewards)
        episode_numbers = episode_numbers or list(range(len(rewards)))
        
        # Calculate moving average
        moving_avg = []
        moving_std = []
        for i in range(self.window_size, len(rewards)):
            window_rewards = rewards[i - self.window_size:i]
            moving_avg.append(np.mean(window_rewards))
            moving_std.append(np.std(window_rewards))
        
        moving_avg = np.array(moving_avg)
        moving_std = np.array(moving_std)
        
        # Find convergence point (when moving average stabilizes)
        convergence_episode = None
        for i in range(1, len(moving_avg)):
            # Check if relative change is below threshold
            if moving_std[i] / (abs(moving_avg[i]) + 1e-6) < self.convergence_threshold:
                # Check if mean is stable
                relative_change = abs(moving_avg[i] - moving_avg[i-1]) / (abs(moving_avg[i-1]) + 1e-6)
                if relative_change < self.convergence_threshold:
                    convergence_episode = episode_numbers[i + self.window_size - 1]
                    break
        
        # Calculate improvement metrics
        initial_performance = np.mean(rewards[:self.window_size])
        final_performance = np.mean(rewards[-self.window_size:])
        improvement = final_performance - initial_performance
        improvement_pct = (improvement / (abs(initial_performance) + 1e-6)) * 100
        
        # Find best performance
        best_episode = episode_numbers[np.argmax(rewards)]
        best_reward = np.max(rewards)
        
        return {
            "converged": convergence_episode is not None,
            "convergence_episode": convergence_episode,
            "initial_performance": float(initial_performance),
            "final_performance": float(final_performance),
            "improvement": float(improvement),
            "improvement_pct": float(improvement_pct),
            "best_episode": int(best_episode),
            "best_reward": float(best_reward),
            "final_std": float(moving_std[-1]) if len(moving_std) > 0 else None,
            "total_episodes": len(rewards)
        }


def train_and_analyze(
    tech_name: str,
    agent_factory,
    env: TrafficEnv,
    max_episodes: int = 1000,
    eval_frequency: int = 50,
    output_dir: Path = None
) -> Dict[str, Any]:
    """
    Train an agent and analyze convergence.
    
    Args:
        tech_name: Name of technology
        agent_factory: Function that creates agent instance
        env: Environment
        max_episodes: Maximum episodes to train
        eval_frequency: Frequency of evaluation logging
        output_dir: Output directory for results
        
    Returns:
        Dictionary with training and convergence results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Analyzing: {tech_name}")
    logger.info(f"{'='*80}")
    
    agent = agent_factory()
    analyzer = ConvergenceAnalyzer()
    
    episode_rewards = []
    episode_lengths = []
    
    try:
        for episode in range(max_episodes):
            obs, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                # Select action
                action = None
                
                if hasattr(agent, 'select_action'):
                    try:
                        import inspect
                        sig = inspect.signature(agent.select_action)
                        if 'epsilon' in sig.parameters:
                            result = agent.select_action(obs, epsilon=max(0.1, 1.0 - episode / max_episodes))
                        else:
                            result = agent.select_action(obs)
                        
                        if isinstance(result, tuple):
                            action = result[0]
                        else:
                            action = result
                    except Exception as e:
                        logger.warning(f"Error in select_action: {e}, using default")
                        action = 0
                        
                elif hasattr(agent, 'predict'):
                    result = agent.predict(obs)
                    if isinstance(result, tuple):
                        action = result[0]
                    else:
                        action = result
                else:
                    action = 0
                
                if action is None:
                    action = 0
                action = int(action)
                
                if action < 0 or action >= env.action_space.n:
                    action = 0
                
                next_obs, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                
                # Store experience
                if hasattr(agent, 'store_experience'):
                    agent.store_experience(obs, action, reward, next_obs, done)
                
                # Train step
                train_interval = 2 if 'Model-Based' in tech_name else 10
                should_train = (
                    episode % train_interval == 0 or 
                    ('Model-Based' in tech_name and 
                     hasattr(agent, 'world_model') and 
                     hasattr(agent, 'is_converged') and
                     not agent.is_converged and
                     hasattr(agent, 'transition_buffer') and
                     len(agent.transition_buffer) >= getattr(agent, 'min_transitions_for_training', 30))
                )
                
                if hasattr(agent, 'train_step') and should_train:
                    if hasattr(agent, 'replay_buffer'):
                        buffer = agent.replay_buffer
                        if len(buffer) > 32:
                            if hasattr(buffer, 'sample'):
                                batch = buffer.sample(32)
                                agent.train_step(batch)
                            else:
                                agent.train_step()
                    elif hasattr(agent, 'transition_buffer'):
                        if hasattr(agent, 'is_converged') and agent.is_converged:
                            pass
                        elif len(agent.transition_buffer) >= getattr(agent, 'min_transitions_for_training', 30):
                            agent.train_step()
                
                episode_reward += reward
                episode_length += 1
                obs = next_obs
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if (episode + 1) % eval_frequency == 0:
                avg_reward = np.mean(episode_rewards[-eval_frequency:])
                logger.info(f"Episode {episode + 1}/{max_episodes}, Avg Reward: {avg_reward:.2f}")
        
        # Analyze convergence
        convergence_analysis = analyzer.analyze_convergence(episode_rewards)
        
        # Save results
        results = {
            "technology": tech_name,
            "episodes": max_episodes,
            "episode_rewards": [float(r) for r in episode_rewards],
            "episode_lengths": [float(l) for l in episode_lengths],
            "convergence_analysis": convergence_analysis,
            "final_avg_reward": float(np.mean(episode_rewards[-50:])),
            "final_std_reward": float(np.std(episode_rewards[-50:])),
        }
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            results_path = output_dir / f"{tech_name.lower().replace(' ', '_')}_analysis.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved analysis to {results_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing {tech_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "technology": tech_name,
            "status": "error",
            "error": str(e)
        }


def plot_convergence(results: Dict[str, Any], output_dir: Path):
    """Plot convergence curves for all technologies."""
    if not results:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Reward over episodes
    ax1 = axes[0]
    for tech_name, data in results.items():
        if "episode_rewards" in data:
            rewards = data["episode_rewards"]
            episodes = list(range(len(rewards)))
            ax1.plot(episodes, rewards, alpha=0.6, label=tech_name)
            
            # Mark convergence point if available
            conv_analysis = data.get("convergence_analysis", {})
            if conv_analysis.get("convergence_episode"):
                conv_ep = conv_analysis["convergence_episode"]
                if conv_ep < len(rewards):
                    ax1.axvline(x=conv_ep, color='red', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Reward Convergence Over Episodes")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Moving average
    ax2 = axes[1]
    window = 50
    for tech_name, data in results.items():
        if "episode_rewards" in data:
            rewards = np.array(data["episode_rewards"])
            moving_avg = []
            for i in range(window, len(rewards)):
                moving_avg.append(np.mean(rewards[i-window:i]))
            
            episodes = list(range(window, len(rewards)))
            ax2.plot(episodes, moving_avg, label=tech_name, linewidth=2)
    
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Moving Average Reward (50 episodes)")
    ax2.set_title("Smoothed Reward Convergence")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "convergence_analysis.png"
        plt.savefig(plot_path, dpi=150)
        logger.info(f"Saved convergence plot to {plot_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze optimal episode counts for technologies")
    parser.add_argument("--config", type=str, default="configs/intersection.json", help="Config file")
    parser.add_argument("--max-episodes", type=int, default=1000, help="Maximum episodes per technology")
    parser.add_argument("--output", type=str, default="./runs/episode_analysis", help="Output directory")
    parser.add_argument("--technologies", type=str, nargs="+", help="Specific technologies to analyze")
    parser.add_argument("--eval-frequency", type=int, default=50, help="Evaluation logging frequency")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create environment
    env = TrafficEnv(config=config)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Define all technologies
    technologies = {}
    
    if HierarchicalRLAgent:
        technologies["Hierarchical RL"] = lambda: HierarchicalRLAgent(state_dim, action_dim)
    if ModelBasedRLAgent:
        technologies["Model-Based RL"] = lambda: ModelBasedRLAgent(state_dim, action_dim)
    if BehavioralCloningAgent:
        technologies["Imitation Learning (BC)"] = lambda: BehavioralCloningAgent(state_dim, action_dim)
    if TransformerAgent:
        technologies["Transformer"] = lambda: TransformerAgent(state_dim, action_dim)
    if BayesianAgent:
        technologies["Bayesian"] = lambda: BayesianAgent(state_dim, action_dim)
    if CausalAgent:
        technologies["Causal"] = lambda: CausalAgent(state_dim, action_dim)
    if NeuroSymbolicAgent:
        technologies["Neuro-Symbolic"] = lambda: NeuroSymbolicAgent(state_dim, action_dim)
    if MAMLAgent:
        technologies["Meta-Learning (MAML)"] = lambda: MAMLAgent(state_dim, action_dim)
    if LLMTrafficAgent:
        technologies["LLM"] = lambda: LLMTrafficAgent(state_dim, action_dim)
    if DiffusionTrafficAgent:
        technologies["Diffusion"] = lambda: DiffusionTrafficAgent(state_dim, action_dim)
    
    # Filter technologies if specified
    if args.technologies:
        technologies = {k: v for k, v in technologies.items() if k in args.technologies}
    
    # Analyze all technologies
    output_dir = Path(args.output)
    results = {}
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting convergence analysis for {len(technologies)} technologies")
    logger.info(f"{'='*80}\n")
    
    for idx, (tech_name, agent_factory) in enumerate(technologies.items(), 1):
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"Analyzing Technology {idx}/{len(technologies)}: {tech_name}")
            logger.info(f"{'='*80}")
            
            result = train_and_analyze(
                tech_name,
                agent_factory,
                env,
                max_episodes=args.max_episodes,
                eval_frequency=args.eval_frequency,
                output_dir=output_dir / tech_name.lower().replace(' ', '_')
            )
            results[tech_name] = result
            
            if "convergence_analysis" in result:
                conv = result["convergence_analysis"]
                logger.info(f"\n✓ Analysis complete for {tech_name}")
                logger.info(f"  Convergence Episode: {conv.get('convergence_episode', 'N/A')}")
                logger.info(f"  Final Performance: {result.get('final_avg_reward', 0):.2f}")
            
        except Exception as e:
            logger.error(f"\n✗ Failed to analyze {tech_name}: {e}")
            results[tech_name] = {"status": "error", "error": str(e)}
    
    # Generate plots
    plot_convergence(results, output_dir)
    
    # Save summary
    summary = {
        "technologies_analyzed": list(results.keys()),
        "max_episodes": args.max_episodes,
        "results": results
    }
    
    summary_path = output_dir / "convergence_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info("Convergence Analysis Complete!")
    logger.info(f"{'='*80}")
    logger.info(f"Results saved to: {summary_path}")
    
    # Print summary
    logger.info("\nConvergence Summary:")
    for tech_name, result in results.items():
        if "convergence_analysis" in result:
            conv = result["convergence_analysis"]
            logger.info(f"{tech_name:30s} Conv: {conv.get('convergence_episode', 'N/A'):>6}  "
                       f"Final: {result.get('final_avg_reward', 0):>8.2f}")


if __name__ == "__main__":
    main()

