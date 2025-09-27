import json
import os
import argparse
import numpy as np
from tqdm import trange
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.env.traffic_env import TrafficEnv
from src.rl.dqn_agent import DQNAgent, DQNConfig

def load_config(path: str):
    with open(path, 'r') as f:
        return json.load(f)

def train(cfg_path: str, episodes: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    cfg = load_config(cfg_path)
    env = TrafficEnv(cfg)

    # Build DQNConfig from environment variables set by CLI (--use_torch, --device)
    cfg = DQNConfig()
    cfg.use_torch = os.environ.get('ADAPTIVE_TRAFFIC_USE_TORCH', '0') == '1'
    dev = os.environ.get('ADAPTIVE_TRAFFIC_TORCH_DEVICE', None)
    cfg.device = dev if dev else None
    if cfg.use_torch:
        print(f"Using PyTorch-backed agent; device={cfg.device or 'auto'}")

    agent = DQNAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, cfg=cfg)

    rewards = []
    start_episode = 0
    checkpoint_path = os.path.join(out_dir, 'checkpoint')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        agent.load(os.path.join(checkpoint_path, 'dqn_traffic.npz'))
        rewards = np.load(os.path.join(checkpoint_path, 'rewards.npy')).tolist()
        start_episode = len(rewards)

    for ep in trange(start_episode, episodes, desc="Training"):
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

        # Save checkpoint after each episode
        os.makedirs(checkpoint_path, exist_ok=True)
        agent.save(os.path.join(checkpoint_path, 'dqn_traffic.npz'))
        np.save(os.path.join(checkpoint_path, 'rewards.npy'), np.array(rewards))

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
