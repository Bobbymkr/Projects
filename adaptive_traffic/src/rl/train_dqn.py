import json
import os
import argparse
import numpy as np
from tqdm import trange

from src.env.traffic_env import TrafficEnv
from src.env.sumo_env import SumoEnv
from src.rl.dqn_agent import DQNAgent, DQNConfig


def load_config(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def make_env(cfg_path: str, use_sumo: bool) -> TrafficEnv:
    cfg = load_config(cfg_path)
    if use_sumo:
        return SumoEnv(cfg)
    return TrafficEnv(cfg)


def train(cfg_path: str, episodes: int, out_dir: str, use_sumo: bool):
    os.makedirs(out_dir, exist_ok=True)
    env = make_env(cfg_path, use_sumo)
    agent = DQNAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, cfg=DQNConfig())

    rewards = []

    for ep in trange(episodes, desc="Training"):
        s, _ = env.reset()
        ep_reward = 0.0
        truncated = terminated = False
        while not (terminated or truncated):
            a = agent.select_action(s.astype(np.float32))
            ns, r, terminated, truncated, _ = env.step(a)
            agent.push(s.astype(np.float32), a, r, ns.astype(np.float32), terminated or truncated)
            loss = agent.train_step()
            ep_reward += r
            s = ns
        rewards.append(ep_reward)

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
    parser.add_argument('--use_sumo', action='store_true', help='Use SUMO-based environment instead of custom simulator')
    args = parser.parse_args()
    train(args.config, args.episodes, args.out, args.use_sumo)
