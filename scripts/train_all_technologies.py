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
from typing import Dict, Any

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


def train_technology(
    tech_name: str,
    agent: Any,
    env: TrafficEnv,
    episodes: int = 200,
    output_dir: Path = None,
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
            
            # Early stop for Model-Based RL when converged
            if 'Model-Based' in tech_name and hasattr(agent, 'is_converged') and agent.is_converged:
                logger.info(f"Convergence achieved for Model-Based RL at episode {episode + 1}. Ending training early.")
                break
            
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                conv_status = " (converged)" if (hasattr(agent, 'is_converged') and agent.is_converged) else ""
                logger.info(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}{conv_status}")
        
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
        
        logger.info(f"Completed {len(episode_rewards)} episodes for {tech_name}")
        return {
            "episodes": len(episode_rewards),
            "avg_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "final_reward": float(np.mean(episode_rewards[-10:])),
            "avg_length": float(np.mean(episode_lengths)),
            "status": final_status,
            "converged": agent.is_converged if hasattr(agent, 'is_converged') else False,
        }
    
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
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes per technology")
    parser.add_argument("--output", type=str, default="./runs/all_technologies", help="Output directory")
    parser.add_argument("--technologies", type=str, nargs="+", help="Specific technologies to train")
    
    args = parser.parse_args()
    
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
            
            agent = agent_factory()
            result = train_technology(
                tech_name,
                agent,
                env,
                episodes=args.episodes,
                output_dir=output_dir / tech_name.lower().replace(' ', '_'),
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
    for tech_name, result in results.items():
        if result.get("status") == "success":
            logger.info(f"{tech_name:30s} Avg Reward: {result.get('avg_reward', 0):.2f}")
        else:
            logger.info(f"{tech_name:30s} Status: {result.get('status', 'unknown')}")


if __name__ == "__main__":
    main()

