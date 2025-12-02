#!/usr/bin/env python3
"""
Hyperparameter Optimization Framework.

Optimizes hyperparameters for RL agents using Optuna.

Usage:
    python scripts/optimize_hyperparameters.py --agent model_based_rl --trials 200
    python scripts/optimize_hyperparameters.py --agent dqn --quick
    python scripts/optimize_hyperparameters.py --agent all --trials 100
"""

import json
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.research.hyperparameter_optimization import (
        HyperparameterOptimizer,
        create_dqn_optimization_objective
    )
    OPTUNA_AVAILABLE = True
except ImportError:
    try:
        import optuna
        OPTUNA_AVAILABLE = True
    except ImportError:
        OPTUNA_AVAILABLE = False
        print("Warning: Optuna not available. Install with: pip install optuna", file=sys.stderr)

try:
    from src.env.traffic_env import TrafficEnv
    from src.rl.dqn_agent import DQNAgent, DQNConfig
except ImportError as e:
    print(f"Warning: Could not import agents: {e}", file=sys.stderr)

# Try to import advanced agents
try:
    from src.research.novel_algorithms.model_based_rl import ModelBasedRLAgent
    from src.research.novel_algorithms.hierarchical_rl import HierarchicalRLAgent
    from src.research.novel_algorithms.transformer_control import TransformerAgent
except ImportError:
    ModelBasedRLAgent = None
    HierarchicalRLAgent = None
    TransformerAgent = None


def create_model_based_rl_objective(env_config: Dict[str, Any]) -> Callable:
    """Create optimization objective for Model-Based RL."""
    if not OPTUNA_AVAILABLE:
        return None
    
    def objective(trial):
        """Optuna objective function for Model-Based RL."""
        horizon = trial.suggest_int("horizon", 3, 10)
        candidates = trial.suggest_int("candidates", 10, 50)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        world_model_lr = trial.suggest_loguniform("world_model_lr", 1e-5, 1e-3)
        
        try:
            # Create environment
            env = TrafficEnv(env_config)
            
            # Create agent with suggested hyperparameters
            if ModelBasedRLAgent:
                agent = ModelBasedRLAgent(
                    horizon=horizon,
                    num_candidates=candidates,
                    learning_rate=learning_rate,
                    world_model_lr=world_model_lr
                )
            else:
                # Fallback: return poor score if agent not available
                return float('-inf')
            
            # Train for a short period
            obs, info = env.reset()
            total_reward = 0
            episodes = 10  # Short training for optimization
            
            for episode in range(episodes):
                obs, info = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action = agent.decide(obs) if hasattr(agent, 'decide') else env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    
                    if hasattr(agent, 'update'):
                        agent.update(obs, action, reward, obs, done)
                
                total_reward += episode_reward
            
            # Return average reward (maximize)
            return total_reward / episodes
        
        except Exception as e:
            print(f"Trial failed: {e}", file=sys.stderr)
            return float('-inf')
    
    return objective


def create_hierarchical_rl_objective(env_config: Dict[str, Any]) -> Callable:
    """Create optimization objective for Hierarchical RL."""
    if not OPTUNA_AVAILABLE:
        return None
    
    def objective(trial):
        """Optuna objective function for Hierarchical RL."""
        num_options = trial.suggest_int("num_options", 3, 8)
        option_horizon = trial.suggest_int("option_horizon", 5, 20)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        termination_beta = trial.suggest_float("termination_beta", 0.01, 0.1)
        
        try:
            env = TrafficEnv(env_config)
            
            if HierarchicalRLAgent:
                agent = HierarchicalRLAgent(
                    num_options=num_options,
                    option_horizon=option_horizon,
                    learning_rate=learning_rate,
                    termination_beta=termination_beta
                )
            else:
                return float('-inf')
            
            # Short training
            total_reward = 0
            episodes = 10
            
            for episode in range(episodes):
                obs, info = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action = agent.decide(obs) if hasattr(agent, 'decide') else env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                
                total_reward += episode_reward
            
            return total_reward / episodes
        
        except Exception as e:
            print(f"Trial failed: {e}", file=sys.stderr)
            return float('-inf')
    
    return objective


def create_transformer_objective(env_config: Dict[str, Any]) -> Callable:
    """Create optimization objective for Transformer Agent."""
    if not OPTUNA_AVAILABLE:
        return None
    
    def objective(trial):
        """Optuna objective function for Transformer."""
        num_heads = trial.suggest_int("num_heads", 2, 8)
        num_layers = trial.suggest_int("num_layers", 2, 6)
        d_model = trial.suggest_int("d_model", 64, 256, step=32)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        
        try:
            env = TrafficEnv(env_config)
            
            if TransformerAgent:
                agent = TransformerAgent(
                    num_heads=num_heads,
                    num_layers=num_layers,
                    d_model=d_model,
                    learning_rate=learning_rate,
                    dropout=dropout
                )
            else:
                return float('-inf')
            
            # Short training
            total_reward = 0
            episodes = 10
            
            for episode in range(episodes):
                obs, info = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action = agent.decide(obs) if hasattr(agent, 'decide') else env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                
                total_reward += episode_reward
            
            return total_reward / episodes
        
        except Exception as e:
            print(f"Trial failed: {e}", file=sys.stderr)
            return float('-inf')
    
    return objective


def create_dqn_objective(env_config: Dict[str, Any]) -> Callable:
    """Create optimization objective for DQN."""
    if not OPTUNA_AVAILABLE:
        return None
    
    def objective(trial):
        """Optuna objective function for DQN."""
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        batch_size = trial.suggest_int("batch_size", 16, 256)
        gamma = trial.suggest_float("gamma", 0.9, 0.999)
        epsilon_start = trial.suggest_float("epsilon_start", 0.9, 1.0)
        epsilon_end = trial.suggest_float("epsilon_end", 0.01, 0.1)
        replay_buffer_size = trial.suggest_int("replay_buffer_size", 10000, 100000, log=True)
        
        try:
            env = TrafficEnv(env_config)
            obs_dim = env_config.get("num_lanes", 4) * 2
            action_dim = env.action_space.n
            
            config = DQNConfig(
                learning_rate=learning_rate,
                batch_size=batch_size,
                gamma=gamma,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                replay_buffer_size=replay_buffer_size
            )
            
            agent = DQNAgent(obs_dim, action_dim, config)
            
            # Short training
            total_reward = 0
            episodes = 10
            
            for episode in range(episodes):
                obs, info = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action = agent.select_action(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    
                    # Update agent
                    if hasattr(agent, 'update'):
                        agent.update(obs, action, reward, obs, done)
                
                total_reward += episode_reward
            
            return total_reward / episodes
        
        except Exception as e:
            print(f"Trial failed: {e}", file=sys.stderr)
            return float('-inf')
    
    return objective


def optimize_agent(agent_name: str, n_trials: int = 200, quick: bool = False) -> Dict[str, Any]:
    """Optimize hyperparameters for a specific agent."""
    if not OPTUNA_AVAILABLE:
        print("Error: Optuna not available. Install with: pip install optuna", file=sys.stderr)
        return {}
    
    if quick:
        n_trials = 20  # Quick optimization
    
    print(f"\n🔧 Optimizing {agent_name} hyperparameters...")
    print(f"   Trials: {n_trials}")
    print("=" * 80)
    
    # Default environment config
    env_config = {
        "num_lanes": 4,
        "min_green": 5,
        "max_green": 60,
        "green_step": 5,
        "arrival_rates": [0.3, 0.25, 0.35, 0.2],
        "queue_capacity": 40
    }
    
    # Get objective function
    if agent_name == "model_based_rl":
        objective = create_model_based_rl_objective(env_config)
    elif agent_name == "hierarchical_rl":
        objective = create_hierarchical_rl_objective(env_config)
    elif agent_name == "transformer":
        objective = create_transformer_objective(env_config)
    elif agent_name == "dqn":
        objective = create_dqn_objective(env_config)
    else:
        print(f"Error: Unknown agent {agent_name}", file=sys.stderr)
        return {}
    
    if objective is None:
        print(f"Error: Could not create objective for {agent_name}", file=sys.stderr)
        return {}
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        study_name=f"adaptive-traffic-{agent_name}",
        direction="maximize",
        sampler="tpe"
    )
    
    if not optimizer.enabled:
        print("Error: HyperparameterOptimizer not enabled", file=sys.stderr)
        return {}
    
    # Run optimization
    start_time = time.time()
    result = optimizer.optimize(objective, n_trials=n_trials, show_progress=True)
    optimization_time = time.time() - start_time
    
    if result:
        print(f"\n✅ Optimization complete!")
        print(f"   Best value: {result.get('best_value', 'N/A')}")
        print(f"   Best parameters:")
        for key, value in result.get('best_params', {}).items():
            print(f"     {key}: {value}")
        print(f"   Trials: {result.get('n_trials', 0)}")
        print(f"   Time: {optimization_time/60:.2f} minutes")
        
        # Save results
        output_dir = PROJECT_ROOT / "results" / "optimization"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{agent_name}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "optimization_time_seconds": optimization_time,
            "best_params": result.get("best_params", {}),
            "best_value": result.get("best_value"),
            "n_trials": result.get("n_trials", 0)
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"   Results saved to: {output_file}")
        
        return report
    
    return {}


def main():
    parser = argparse.ArgumentParser(description="Optimize hyperparameters for RL agents")
    parser.add_argument("--agent", type=str, required=True, help="Agent name (model_based_rl, hierarchical_rl, transformer, dqn, or 'all')")
    parser.add_argument("--trials", type=int, default=200, help="Number of optimization trials")
    parser.add_argument("--quick", action="store_true", help="Quick optimization (20 trials)")
    parser.add_argument("--output", type=Path, help="Output directory")
    
    args = parser.parse_args()
    
    if not OPTUNA_AVAILABLE:
        print("Error: Optuna not available. Install with: pip install optuna", file=sys.stderr)
        sys.exit(1)
    
    agents_to_optimize = []
    
    if args.agent == "all":
        agents_to_optimize = ["model_based_rl", "hierarchical_rl", "transformer", "dqn"]
    else:
        agents_to_optimize = [args.agent]
    
    results = {}
    
    for agent in agents_to_optimize:
        result = optimize_agent(agent, args.trials, args.quick)
        if result:
            results[agent] = result
    
    if len(agents_to_optimize) > 1:
        print(f"\n📊 Summary: Optimized {len(results)}/{len(agents_to_optimize)} agents")


if __name__ == "__main__":
    main()

