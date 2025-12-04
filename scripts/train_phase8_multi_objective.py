"""
Phase 8: Multi-Objective Optimization Training Script

Trains agents with multi-objective optimization:
- MO-PPO (Multi-Objective PPO)
- Constraint optimization
- CPO (Constrained Policy Optimization)
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research.multi_objective.phase8_multi_objective import (
    MOPPOAgent, MOPPOConfig,
    CPOAgent, ConstraintConfig,
    MultiObjectiveReward, MultiObjectiveWeights,
)
from src.env.traffic_env import TrafficEnv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 8: Multi-Objective Optimization Training")
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["MO-PPO", "CPO"],
        help="Algorithm to train"
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
        default=1000,
        help="Number of training episodes"
    )
    parser.add_argument(
        "--objective-weights",
        type=str,
        default=None,
        help="Comma-separated objective weights (wait_time,fuel,emissions,throughput,accidents,infrastructure)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/phase8_multi_objective",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        env_config = json.load(f)
    
    # Create environment
    env = TrafficEnv(config=env_config)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Parse objective weights
    if args.objective_weights:
        weights = [float(w) for w in args.objective_weights.split(',')]
        if len(weights) != 6:
            raise ValueError("Must provide 6 objective weights")
        objective_weights = MultiObjectiveWeights(
            wait_time=weights[0],
            fuel_consumption=weights[1],
            emissions=weights[2],
            throughput=weights[3],
            accidents=weights[4],
            infrastructure_wear=weights[5],
        )
    else:
        objective_weights = MultiObjectiveWeights()
    
    # Create multi-objective reward function
    reward_fn = MultiObjectiveReward(weights=objective_weights)
    
    # Create agent
    if args.algorithm == "MO-PPO":
        agent_config = MOPPOConfig()
        agent = MOPPOAgent(state_dim, action_dim, agent_config)
    elif args.algorithm == "CPO":
        constraint_config = ConstraintConfig()
        agent_config = MOPPOConfig()
        agent = CPOAgent(state_dim, action_dim, constraint_config, agent_config)
    
    # Training loop
    episode_rewards = []
    episode_objectives = []
    
    logger.info(f"Starting {args.algorithm} training for {args.episodes} episodes")
    
    for episode in range(args.episodes):
        obs, _ = env.reset()
        obs = np.array(obs, dtype=np.float32).flatten()
        done = False
        episode_reward = 0.0
        episode_objective_rewards = np.zeros(6)
        
        wait_times = []
        queue_lengths = []
        vehicles_served = 0
        phase_changes = 0
        emergency_stops = 0
        
        while not done:
            # Select action
            action, log_prob, values = agent.select_action(obs)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_obs = np.array(next_obs, dtype=np.float32).flatten()
            
            # Collect statistics
            wait_times.extend(env.wait_times.tolist())
            queue_lengths.extend(env.queues.tolist())
            vehicles_served += info.get('vehicles_served', 0)
            phase_changes += 1
            emergency_stops += info.get('emergency_stops', 0)
            
            # Compute multi-objective rewards
            obj_rewards = reward_fn.compute_rewards(
                wait_times=np.array(wait_times),
                queue_lengths=np.array(queue_lengths),
                vehicles_served=vehicles_served,
                phase_changes=phase_changes,
                emergency_stops=emergency_stops,
            )
            
            # Store transition
            if hasattr(agent, 'store_transition'):
                agent.store_transition(
                    obs, action, log_prob, values,
                    np.array([
                        obj_rewards['wait_time'],
                        obj_rewards['fuel_consumption'],
                        obj_rewards['emissions'],
                        obj_rewards['throughput'],
                        obj_rewards['accidents'],
                        obj_rewards['infrastructure_wear'],
                    ]),
                    next_obs, done,
                )
            
            episode_reward += obj_rewards['total']
            episode_objective_rewards += np.array([
                obj_rewards['wait_time'],
                obj_rewards['fuel_consumption'],
                obj_rewards['emissions'],
                obj_rewards['throughput'],
                obj_rewards['accidents'],
                obj_rewards['infrastructure_wear'],
            ])
            
            obs = next_obs
        
        episode_rewards.append(episode_reward)
        episode_objectives.append(episode_objective_rewards)
        
        # Train
        if hasattr(agent, 'train_step'):
            metrics = agent.train_step()
        
        # Logging
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_objectives = np.mean(episode_objectives[-100:], axis=0)
            logger.info(
                f"Episode {episode + 1}/{args.episodes} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Objectives: {avg_objectives}"
            )
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'episode_rewards': episode_rewards,
        'episode_objectives': [obj.tolist() for obj in episode_objectives],
        'final_avg_reward': float(np.mean(episode_rewards[-100:])),
        'final_avg_objectives': np.mean(episode_objectives[-100:], axis=0).tolist(),
    }
    
    with open(output_dir / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Training complete! Results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

