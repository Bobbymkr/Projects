"""
Unified Training Script for All Technologies.

Trains all implemented technologies for comprehensive comparison.
"""

import argparse
import logging
import json
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, Optional
from collections import deque

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.traffic_env import TrafficEnv
from src.control.fuzzy_control import FuzzyController
from src.control.webster_method import WebsterMethod

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


def load_optimal_episodes_config(config_path: str = "configs/optimal_episodes.json") -> Dict[str, Any]:
    """Load optimal episode counts configuration."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Optimal episodes config not found at {config_path}, using defaults")
        return {}


def get_optimal_episodes(tech_name: str, use_case: str = "production", config: Optional[Dict] = None) -> int:
    """
    Get optimal episode count for a technology.
    
    Args:
        tech_name: Name of technology
        use_case: Use case ('quick_testing', 'development', 'production', 'research_quality')
        config: Optional config dictionary
        
    Returns:
        Optimal episode count
    """
    if config is None:
        config = load_optimal_episodes_config()
    
    optimal_configs = config.get("optimal_episode_counts", {})
    
    # Map technology names
    tech_mapping = {
        "Hierarchical RL": "Hierarchical RL",
        "Model-Based RL": "Model-Based RL",
        "Imitation Learning (BC)": "Imitation Learning",
        "Transformer": "Transformer",
        "Bayesian": "Bayesian",
        "Causal": "Causal",
        "Neuro-Symbolic": "Neuro-Symbolic",
        "Meta-Learning (MAML)": "Meta-Learning (MAML)",
        "LLM": "LLM",
        "Diffusion": "Diffusion"
    }
    
    mapped_name = tech_mapping.get(tech_name, tech_name)
    
    if mapped_name in optimal_configs:
        tech_config = optimal_configs[mapped_name]
        
        if use_case == "quick_testing":
            return tech_config.get("quick_testing", 100)
        elif use_case == "development":
            return tech_config.get("development", 200)
        elif use_case == "production":
            # Use middle of production range
            prod_range = tech_config.get("production", [500, 1000])
            return int(np.mean(prod_range))
        elif use_case == "research_quality":
            # Use middle of research range
            research_range = tech_config.get("research_quality", [2000, 3000])
            return int(np.mean(research_range))
        else:
            # Default to production
            prod_range = tech_config.get("production", [500, 1000])
            return int(np.mean(prod_range))
    
    # Default values if not found
    defaults = {
        "quick_testing": 100,
        "development": 200,
        "production": 500,
        "research_quality": 2000
    }
    return defaults.get(use_case, 500)


class EarlyStopping:
    """Early stopping based on convergence detection."""
    
    def __init__(self, patience: int = 100, min_delta: float = 0.01, monitor: str = "reward"):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of episodes to wait without improvement
            min_delta: Minimum change to qualify as improvement
            monitor: Metric to monitor ('reward' or 'loss')
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_value = None
        self.wait_count = 0
        self.stopped_epoch = 0
        self.best_episode = 0
        
    def check(self, current_value: float, episode: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_value: Current metric value
            episode: Current episode number
            
        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = current_value
            self.best_episode = episode
            return False
        
        # For reward, higher is better; for loss, lower is better
        is_better = False
        if self.monitor == "reward":
            is_better = current_value > self.best_value + self.min_delta
        else:
            is_better = current_value < self.best_value - self.min_delta
        
        if is_better:
            self.best_value = current_value
            self.best_episode = episode
            self.wait_count = 0
        else:
            self.wait_count += 1
        
        if self.wait_count >= self.patience:
            self.stopped_episode = episode
            return True
        
        return False


def train_technology(
    tech_name: str,
    agent: Any,
    env: TrafficEnv,
    episodes: int = 200,
    output_dir: Path = None,
    early_stopping: Optional[EarlyStopping] = None,
    track_performance: bool = True,
) -> Dict[str, Any]:
    """
    Train a technology.
    
    Args:
        tech_name: Name of technology
        agent: Agent instance
        env: Environment
        episodes: Number of episodes
        output_dir: Output directory
        
    Returns:
        Training results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Training: {tech_name}")
    logger.info(f"{'='*80}")
    
    episode_rewards = []
    episode_lengths = []
    performance_history = {
        "rewards": [],
        "wait_times": [],
        "queue_lengths": []
    }
    
    # Initialize early stopping if provided
    if early_stopping is None:
        early_stopping = EarlyStopping(patience=100, min_delta=0.01, monitor="reward")
    
    try:
        for episode in range(episodes):
            obs, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                # Select action based on agent type
                action = None
                
                if hasattr(agent, 'select_action'):
                    try:
                        import inspect
                        sig = inspect.signature(agent.select_action)
                        if 'epsilon' in sig.parameters:
                            result = agent.select_action(obs, epsilon=max(0.1, 1.0 - episode / episodes))
                        else:
                            result = agent.select_action(obs)
                        
                        # Handle tuple returns (action, explanation/reasoning)
                        if isinstance(result, tuple):
                            action = result[0]
                        else:
                            action = result
                    except Exception as e:
                        logger.warning(f"Error in select_action: {e}, using default")
                        action = 0
                        
                elif hasattr(agent, 'predict'):
                    result = agent.predict(obs)
                    # Handle tuple returns
                    if isinstance(result, tuple):
                        action = result[0]
                    else:
                        action = result
                        
                elif hasattr(agent, 'compute_timing'):
                    # For Fuzzy Controller
                    green_time = agent.compute_timing(obs)
                    green_values = env.green_values
                    action = np.argmin(np.abs(green_values - green_time))
                else:
                    action = 0
                
                # Ensure action is valid integer
                if action is None:
                    action = 0
                action = int(action)
                
                # Validate action is in valid range
                if action < 0 or action >= env.action_space.n:
                    logger.warning(f"Invalid action {action}, using 0")
                    action = 0
                
                next_obs, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                
                # Store experience (if agent supports it)
                if hasattr(agent, 'store_experience'):
                    agent.store_experience(obs, action, reward, next_obs, done)
                
                # Train step (if agent supports it)
                # Train more frequently for Model-Based RL (needs world model training)
                train_interval = 2 if 'Model-Based' in tech_name else 10
                # Also train every step for Model-Based RL until world model is trained
                should_train = (
                    episode % train_interval == 0 or 
                    ('Model-Based' in tech_name and 
                     hasattr(agent, 'world_model') and 
                     not agent.world_model.is_trained and
                     len(agent.transition_buffer) >= agent.min_transitions_for_training)
                )
                if hasattr(agent, 'train_step') and should_train:
                    # Collect batch and train
                    if hasattr(agent, 'replay_buffer'):
                        buffer = agent.replay_buffer
                        if len(buffer) > 32:
                            # Handle different buffer types
                            if hasattr(buffer, 'sample'):
                                # Replay buffer with sample method
                                batch = buffer.sample(32)
                                agent.train_step(batch)
                            else:
                                # deque or list - train without batch
                                agent.train_step()
                    elif hasattr(agent, 'transition_buffer'):
                        # For Model-Based RL with transition buffer
                        # Skip training if converged
                        if hasattr(agent, 'is_converged') and agent.is_converged:
                            # Already converged, skip training
                            pass
                        elif len(agent.transition_buffer) >= agent.min_transitions_for_training:
                            agent.train_step()
                
                episode_reward += reward
                episode_length += 1
                obs = next_obs
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Track performance metrics
            if track_performance:
                performance_history["rewards"].append(episode_reward)
                # Extract wait times and queue lengths from step_info if available
                # (This would need to be collected during episode if available)
            
            # Check for early stopping
            if len(episode_rewards) >= 50:
                recent_avg = np.mean(episode_rewards[-50:])
                if early_stopping.check(recent_avg, episode):
                    logger.info(f"Early stopping triggered at episode {episode + 1}")
                    logger.info(f"Best performance at episode {early_stopping.best_episode + 1}: {early_stopping.best_value:.2f}")
                    break
            
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                conv_status = " (converged)" if (hasattr(agent, 'is_converged') and agent.is_converged) else ""
                early_stop_info = f" [Early stop: {early_stopping.wait_count}/{early_stopping.patience}]" if early_stopping.wait_count > 0 else ""
                logger.info(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}{conv_status}{early_stop_info}")
        
        # Save model
        if output_dir and hasattr(agent, 'save'):
            output_dir.mkdir(parents=True, exist_ok=True)
            model_path = output_dir / f"{tech_name.lower().replace(' ', '_')}_model.pt"
            try:
                agent.save(str(model_path))
                logger.info(f"Saved model to {model_path}")
            except Exception as e:
                logger.warning(f"Could not save model: {e}")
        
        final_status = "success"
        if hasattr(agent, 'is_converged') and agent.is_converged:
            final_status = "success (converged)"
        
        actual_episodes = len(episode_rewards)
        logger.info(f"Completed {actual_episodes} episodes for {tech_name}")
        
        result = {
            "episodes": actual_episodes,
            "requested_episodes": episodes,
            "avg_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "final_reward": float(np.mean(episode_rewards[-10:])),
            "best_reward": float(np.max(episode_rewards)),
            "best_episode": int(np.argmax(episode_rewards)),
            "avg_length": float(np.mean(episode_lengths)),
            "status": final_status,
            "converged": agent.is_converged if hasattr(agent, 'is_converged') else False,
            "early_stopped": early_stopping.wait_count >= early_stopping.patience,
            "early_stopping_best_episode": early_stopping.best_episode,
            "early_stopping_best_value": float(early_stopping.best_value) if early_stopping.best_value else None,
        }
        
        if track_performance:
            result["performance_history"] = {
                "rewards": [float(r) for r in performance_history["rewards"]]
            }
        
        return result
    
    except Exception as e:
        logger.error(f"Error training {tech_name}: {e}")
        return {
            "episodes": episodes,
            "status": "error",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Train all technologies")
    parser.add_argument("--config", type=str, default="configs/intersection.json", help="Config file")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes per technology (overrides use-case)")
    parser.add_argument("--use-case", type=str, default="production", 
                       choices=["quick_testing", "development", "production", "research_quality"],
                       help="Use case for determining optimal episode counts")
    parser.add_argument("--output", type=str, default="./runs/all_technologies", help="Output directory")
    parser.add_argument("--technologies", type=str, nargs="+", help="Specific technologies to train")
    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping based on convergence")
    parser.add_argument("--early-stopping-patience", type=int, default=100, help="Early stopping patience")
    parser.add_argument("--early-stopping-delta", type=float, default=0.01, help="Early stopping minimum delta")
    parser.add_argument("--optimal-episodes-config", type=str, default="configs/optimal_episodes.json",
                       help="Path to optimal episodes configuration file")
    
    args = parser.parse_args()
    
    # Load optimal episodes configuration
    optimal_config = load_optimal_episodes_config(args.optimal_episodes_config)
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create environment
    env = TrafficEnv(
        config=config,
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Define all technologies (only include available ones)
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
    
    # Production episode counts for each technology (explicit assignments)
    PRODUCTION_EPISODES = {
        "Hierarchical RL": 1000,           # Best performer: 800-1200 range, using 1000
        "Model-Based RL": 400,              # Most sample-efficient: 300-500 range, using 400
        "Imitation Learning (BC)": 300,    # 200-400 range, using 300
        "Transformer": 1150,                # 800-1500 range, using 1150
        "Bayesian": 650,                    # 500-800 range, using 650
        "Causal": 800,                      # 600-1000 range, using 800
        "Neuro-Symbolic": 650,              # 500-800 range, using 650
        "Meta-Learning (MAML)": 2250,      # 1500-3000 range, using 2250
        "LLM": 1150,                        # 800-1500 range, using 1150
        "Diffusion": 2250,                  # 1500-3000 range, using 2250
    }
    
    # Train all technologies
    output_dir = Path(args.output)
    results = {}
    
    total_techs = len(technologies)
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting training for {total_techs} technologies")
    logger.info(f"{'='*80}\n")
    
    for idx, (tech_name, agent_factory) in enumerate(technologies.items(), 1):
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"Training Technology {idx}/{total_techs}: {tech_name}")
            logger.info(f"{'='*80}")
            
            # Determine episode count
            if args.episodes is not None:
                episodes = args.episodes
                logger.info(f"Using specified episode count: {episodes}")
            elif args.use_case == "production" and tech_name in PRODUCTION_EPISODES:
                # Use explicit production episode count
                episodes = PRODUCTION_EPISODES[tech_name]
                logger.info(f"Using explicit production episode count: {episodes}")
            else:
                # Fall back to config-based lookup for other use cases
                episodes = get_optimal_episodes(tech_name, args.use_case, optimal_config)
                logger.info(f"Using optimal episode count for {args.use_case}: {episodes}")
            
            # Setup early stopping
            early_stopping = None
            if args.early_stopping:
                early_stopping_config = optimal_config.get("early_stopping", {})
                patience = args.early_stopping_patience or early_stopping_config.get("patience", 100)
                min_delta = args.early_stopping_delta or early_stopping_config.get("min_delta", 0.01)
                monitor = early_stopping_config.get("monitor", "reward")
                early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, monitor=monitor)
                logger.info(f"Early stopping enabled: patience={patience}, min_delta={min_delta}")
            
            agent = agent_factory()
            result = train_technology(
                tech_name,
                agent,
                env,
                episodes=episodes,
                output_dir=output_dir / tech_name.lower().replace(' ', '_'),
                early_stopping=early_stopping,
                track_performance=True,
            )
            results[tech_name] = result
            
            logger.info(f"\n✓ Completed training for {tech_name}")
            if result.get("status") == "success":
                logger.info(f"  Average Reward: {result.get('avg_reward', 0):.2f}")
            else:
                logger.warning(f"  Status: {result.get('status', 'unknown')}")
                
        except Exception as e:
            logger.error(f"\n✗ Failed to train {tech_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results[tech_name] = {"status": "error", "error": str(e)}
    
    # Save results
    results_path = output_dir / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info("Training Complete!")
    logger.info(f"{'='*80}")
    logger.info(f"Results saved to: {results_path}")
    
    # Print summary
    logger.info("\nTraining Summary:")
    logger.info(f"{'Technology':<30} {'Episodes':<10} {'Avg Reward':<12} {'Best Reward':<12} {'Status':<15}")
    logger.info("-" * 85)
    for tech_name, result in results.items():
        if result.get("status") == "success":
            episodes = result.get("episodes", 0)
            avg_reward = result.get("avg_reward", 0)
            best_reward = result.get("best_reward", 0)
            status = result.get("status", "unknown")
            if result.get("early_stopped"):
                status += " (early stop)"
            logger.info(f"{tech_name:<30} {episodes:<10} {avg_reward:<12.2f} {best_reward:<12.2f} {status:<15}")
        else:
            logger.info(f"{tech_name:<30} {'N/A':<10} {'N/A':<12} {'N/A':<12} {result.get('status', 'unknown'):<15}")


if __name__ == "__main__":
    main()

