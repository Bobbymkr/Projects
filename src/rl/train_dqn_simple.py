import json
import os
import argparse
import numpy as np
from tqdm import trange
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.env.traffic_env import TrafficEnv
from src.rl.dqn_agent import DQNAgent, DQNConfig
from src.rl.convergence_monitor import ConvergenceMonitor
from src.rl.curriculum_learning import TrafficCurriculum

def load_config(path: str):
    """Load configuration from JSON file.
    
    Args:
        path: Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)

def train(cfg_path: str, episodes: int, out_dir: str):
    """Train DQN agent using NumPy implementation with optional PyTorch backend.
    
    Args:
        cfg_path: Path to environment configuration file
        episodes: Number of training episodes
        out_dir: Output directory for saving models and checkpoints
    """
    os.makedirs(out_dir, exist_ok=True)
    env_cfg = load_config(cfg_path)
    env = TrafficEnv(env_cfg)

    # Build DQNConfig from environment variables set by CLI (--use_torch, --device, --use_per)
    cfg = DQNConfig()
    cfg.use_torch = os.environ.get('ADAPTIVE_TRAFFIC_USE_TORCH', '0') == '1'
    dev = os.environ.get('ADAPTIVE_TRAFFIC_TORCH_DEVICE', None)
    cfg.device = dev if dev else None
    if cfg.use_torch:
        print(f"Using PyTorch-backed agent; device={cfg.device or 'auto'}")
    
    # Enable PER if requested (Phase 2.2: Advanced Training Techniques)
    cfg.use_per = os.environ.get('ADAPTIVE_TRAFFIC_USE_PER', '0') == '1'
    if cfg.use_per:
        print("Using Prioritized Experience Replay (PER)")
        cfg.per_alpha = float(os.environ.get('ADAPTIVE_TRAFFIC_PER_ALPHA', '0.6'))
        cfg.per_beta = float(os.environ.get('ADAPTIVE_TRAFFIC_PER_BETA', '0.4'))
        cfg.per_beta_increment = float(os.environ.get('ADAPTIVE_TRAFFIC_PER_BETA_INC', '0.001'))

    agent = DQNAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, cfg=cfg)

    # Initialize convergence monitor (Phase 0.3: Convergence Detection)
    convergence_monitor = ConvergenceMonitor(
        window=100,
        threshold=0.01,
        patience=500,
        min_episodes=200,
        mode="maximize"  # Maximize reward
    )
    
    # Initialize curriculum learning (Phase 2.1: Advanced Training Techniques)
    # Get base arrival rates from environment config
    base_arrival_rates = env_cfg.get("arrival_rates", [0.3] * env.num_lanes)
    curriculum = TrafficCurriculum(
        base_arrival_rates=base_arrival_rates,
        performance_threshold=0.7,
        min_episodes_per_level=50,
        performance_window=100
    )

    rewards = []
    start_episode = 0
    checkpoint_path = os.path.join(out_dir, 'checkpoint')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        agent.load(os.path.join(checkpoint_path, 'dqn_traffic.npz'))
        rewards = np.load(os.path.join(checkpoint_path, 'rewards.npy')).tolist()
        start_episode = len(rewards)
        # Restore convergence monitor state if available
        if os.path.exists(os.path.join(checkpoint_path, 'convergence_state.npz')):
            conv_state = np.load(os.path.join(checkpoint_path, 'convergence_state.npz'), allow_pickle=True)
            convergence_monitor.best_reward = float(conv_state['best_reward'])
            convergence_monitor.best_episode = int(conv_state['best_episode'])
            convergence_monitor.no_improvement_count = int(conv_state['no_improvement_count'])

    for ep in trange(start_episode, episodes, desc="Training"):
        # Update environment with curriculum level (Phase 2.1)
        current_level = curriculum.get_current_level()
        env.arrival_rates = curriculum.get_arrival_rates()
        
        s, _ = env.reset()
        ep_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            a = agent.select_action(s.astype(np.float32))
            ns, r, terminated, truncated, _ = env.step(a)
            agent.push(s.astype(np.float32), a, r, ns.astype(np.float32), terminated or truncated)
            loss = agent.train_step()
            ep_reward += r
            s = ns
        rewards.append(ep_reward)
        
        # Update curriculum learning (Phase 2.1)
        curriculum.update_performance(ep_reward, ep)

        # Update convergence monitor (Phase 0.3)
        conv_status = convergence_monitor.update(ep_reward, ep)
        
        # Log convergence status periodically
        if (ep + 1) % 50 == 0:
            stats = convergence_monitor.get_statistics()
            print(f"\nEpisode {ep+1}: Reward={ep_reward:.2f}, "
                  f"Best={stats['best_reward']:.2f} (ep {stats['best_episode']}), "
                  f"No improvement={stats['no_improvement_count']}/{convergence_monitor.patience}, "
                  f"Recent avg={stats['recent_avg']:.2f}±{stats['recent_std']:.2f}")

        # Check for early stopping
        if convergence_monitor.should_stop():
            print(f"\nEarly stopping triggered at episode {ep+1}")
            print(f"Best reward: {convergence_monitor.best_reward:.2f} at episode {convergence_monitor.best_episode}")
            print(f"No improvement for {convergence_monitor.no_improvement_count} episodes")
            break

        # Save checkpoint after each episode
        os.makedirs(checkpoint_path, exist_ok=True)
        agent.save(os.path.join(checkpoint_path, 'dqn_traffic.npz'))
        np.save(os.path.join(checkpoint_path, 'rewards.npy'), np.array(rewards))
        # Save convergence monitor state
        stats = convergence_monitor.get_statistics()
        np.savez(
            os.path.join(checkpoint_path, 'convergence_state.npz'),
            best_reward=stats['best_reward'],
            best_episode=stats['best_episode'],
            no_improvement_count=stats['no_improvement_count']
        )

    # Save model
    model_path = os.path.join(out_dir, 'dqn_traffic.npz')
    agent.save(model_path)
    print(f"Saved model to {model_path}")

    # Save training stats
    np.save(os.path.join(out_dir, 'rewards.npy'), np.array(rewards))
    print(f"Average reward over {episodes} episodes: {np.mean(rewards):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/intersection.json')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--out', default='runs')
    parser.add_argument('--use_torch', action='store_true', help='Use PyTorch-backed agent (enable GPU when available)')
    parser.add_argument('--device', default=None, help="Optional device string for PyTorch (e.g. 'cuda:0' or 'cpu')")
    args = parser.parse_args()

    # Configure env vars for DQNAgent setup
    os.environ['ADAPTIVE_TRAFFIC_USE_TORCH'] = '1' if args.use_torch else '0'
    if args.device:
        os.environ['ADAPTIVE_TRAFFIC_TORCH_DEVICE'] = args.device

    train(args.config, args.episodes, args.out)
