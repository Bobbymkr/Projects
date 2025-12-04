"""
Complete Test for Phase 2.1 and 2.2.

Tests Curriculum Learning and PER together.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.traffic_env import TrafficEnv
from src.rl.dqn_with_per import DQNAgentWithPER, PERConfig
from src.rl.pytorch_dqn import DQNConfig
from src.rl.curriculum_learning import TrafficCurriculum
from src.rl.training_stability import TrainingStabilityFramework
from src.rl.convergence_monitor import ConvergenceMonitor

def test_phase2_complete():
    """Test Phase 2.1 and 2.2 together."""
    print("="*80)
    print("Testing Phase 2: Curriculum Learning + PER")
    print("="*80)
    
    # Create environment
    config = {
        "num_lanes": 4,
        "phase_lanes": [[0, 1], [2, 3]],
        "min_green": 5,
        "max_green": 60,
        "green_step": 5,
        "cycle_yellow": 3,
        "cycle_all_red": 1,
        "arrival_rates": [0.3, 0.25, 0.35, 0.2],
        "queue_capacity": 40,
        "episode_horizon": 3600
    }
    env = TrafficEnv(config=config)
    
    # Create agent with PER
    dqn_cfg = DQNConfig()
    per_cfg = PERConfig(enabled=True, alpha=0.6, beta=0.4)
    agent = DQNAgentWithPER(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        cfg=dqn_cfg,
        per_config=per_cfg
    )
    
    # Create curriculum
    base_arrival_rates = config["arrival_rates"]
    curriculum = TrafficCurriculum(
        base_arrival_rates=base_arrival_rates,
        performance_threshold=0.7,
        min_episodes_per_level=20,  # Lower for testing
        performance_window=50
    )
    
    # Create stability framework
    stability_framework = TrainingStabilityFramework(
        optimizer=agent.optimizer,
        policy_net=agent.policy_net,
        target_net=agent.target_net,
        config={"grad_clip_norm": 10.0}
    )
    
    # Create convergence monitor
    convergence_monitor = ConvergenceMonitor(
        window=50,
        threshold=0.01,
        patience=200,
        min_episodes=50
    )
    
    print(f"\n✅ All components initialized")
    print(f"   Agent: DQN with PER")
    print(f"   Curriculum: {len(curriculum.levels)} levels")
    print(f"   PER: Alpha={per_cfg.alpha}, Beta={per_cfg.beta}")
    
    # Training loop
    print(f"\n🎓 Training with Phase 2 optimizations...")
    episode_rewards = []
    
    for episode in range(200):
        # Update environment with curriculum
        config_update = curriculum.get_config_update()
        env.arrival_rates = np.array(config_update["arrival_rates"])
        env.cfg["arrival_rates"] = config_update["arrival_rates"]
        
        obs, _ = env.reset()
        episode_reward = 0.0
        steps = 0
        
        done = False
        while not done:
            # Select action
            epsilon = stability_framework.get_exploration_rate() if stability_framework else 0.0
            if epsilon and np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            
            # Store experience
            agent.push(obs, action, reward, next_obs, done)
            
            # Train
            if len(agent.memory) >= agent.cfg.batch_size:
                loss = agent.train_step()
                if loss is not None:
                    stability_framework.clip_gradients()
                    stability_framework.update_target_network()
            
            obs = next_obs
        
        # Update components
        episode_rewards.append(episode_reward)
        convergence_monitor.update(episode_reward, episode)
        curriculum.update_performance(episode_reward, episode)
        stability_framework.step_scheduler(metric=episode_reward)
        stability_framework.step_exploration()
        
        # Logging
        if (episode + 1) % 50 == 0:
            stats = convergence_monitor.get_statistics()
            curriculum_stats = curriculum.get_statistics()
            print(
                f"Episode {episode + 1}/200 | "
                f"Reward: {episode_reward:.2f} | "
                f"Avg: {np.mean(episode_rewards[-50:]):.2f} | "
                f"Best: {stats['best_reward']:.2f} | "
                f"Curriculum Level: {curriculum_stats['current_level']}/{curriculum_stats['total_levels']-1} | "
                f"PER Beta: {agent.memory.beta:.3f}"
            )
    
    # Final statistics
    print(f"\n📊 Final Statistics:")
    print(f"   Episodes: {len(episode_rewards)}")
    print(f"   Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"   Best Reward: {convergence_monitor.get_statistics()['best_reward']:.2f}")
    
    curriculum_stats = curriculum.get_statistics()
    print(f"   Final Curriculum Level: {curriculum_stats['current_level']}/{curriculum_stats['total_levels']-1}")
    print(f"   Episodes at Level: {curriculum_stats['episodes_at_level']}")
    
    print(f"   PER Buffer Size: {len(agent.memory)}")
    print(f"   Final PER Beta: {agent.memory.beta:.4f}")
    print(f"   Final PER Alpha: {agent.memory.alpha:.4f}")
    
    print(f"\n✅ Phase 2 Test Complete!")
    return True

if __name__ == "__main__":
    test_phase2_complete()

