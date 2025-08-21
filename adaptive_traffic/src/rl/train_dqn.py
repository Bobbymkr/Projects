import json
import os
import argparse
import numpy as np
from tqdm import trange
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.env.traffic_env import TrafficEnv
from src.env.sumo_env import SumoEnv
from src.env.marl_env import MarlEnv
from src.rl.dqn_agent import DQNAgent, DQNConfig


def load_config(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def make_env(cfg_path: str, use_sumo: bool, use_marl: bool):
    if use_marl:
        return MarlEnv(config_path=cfg_path)
    cfg = load_config(cfg_path)
    if use_sumo:
        return SumoEnv(cfg)
    return TrafficEnv(cfg)


def train(cfg_path: str, episodes: int, out_dir: str, use_sumo: bool, use_marl: bool):
    os.makedirs(out_dir, exist_ok=True)
    env = make_env(cfg_path, use_sumo, use_marl)
    if use_marl:
        # Pre-train forecasters
        num_collection_episodes = 10
        input_timesteps = 10
        forecast_steps = 5
        for _ in range(num_collection_episodes):
            states = env.reset()
            done = False
            episode_data = [[] for _ in range(env.num_agents)]
            while not done:
                actions = [np.random.randint(0, 2) for _ in range(env.num_agents)]
                next_states, _, dones, _ = env.step(actions)
                for i in range(env.num_agents):
                    base_state = next_states[i][:8]
                    episode_data[i].append(base_state)
                states = next_states
                done = any(dones)
            for i in range(env.num_agents):
                data = np.array(episode_data[i])
                if len(data) > input_timesteps + forecast_steps:
                    X = []
                    y = []
                    for t in range(len(data) - input_timesteps - forecast_steps + 1):
                        X.append(data[t:t+input_timesteps])
                        y.append(data[t+input_timesteps:t+input_timesteps+forecast_steps])
                    X = np.array(X)
                    y = np.array(y)
                    env.forecaster[env.intersections[i]].train(X, y, epochs=20, batch_size=32)
        # Save forecasters
        for i, tl in enumerate(env.intersections):
            env.forecaster[tl].save(os.path.join(out_dir, f'forecaster_{tl}.h5'))

    if use_marl:
        num_agents = env.num_agents
        agents = [DQNAgent(state_dim=env.observation_space[i].shape[0], action_dim=env.action_space[i].n, cfg=DQNConfig()) for i in range(num_agents)]
    else:
        agent = DQNAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, cfg=DQNConfig())

    rewards = []
    start_episode = 0
    checkpoint_path = os.path.join(out_dir, 'checkpoint')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        if use_marl:
            for i, ag in enumerate(agents):
                ag.load(os.path.join(checkpoint_path, f'dqn_traffic_agent_{i}.npz'))
        else:
            agent.load(os.path.join(checkpoint_path, 'dqn_traffic.npz'))
        rewards = np.load(os.path.join(checkpoint_path, 'rewards.npy')).tolist()
        start_episode = len(rewards)

    for ep in trange(start_episode, episodes, desc="Training"):
        if use_marl:
            states = env.reset()
            ep_rewards = [0.0] * num_agents
            done = False
            while not done:
                actions = [ag.select_action(st.astype(np.float32)) for ag, st in zip(agents, states)]
                next_states, rews, dones, _ = env.step(actions)
                done = any(dones)
                for i in range(num_agents):
                    agents[i].push(states[i].astype(np.float32), actions[i], rews[i], next_states[i].astype(np.float32), dones[i])
                    agents[i].train_step()
                    ep_rewards[i] += rews[i]
                states = next_states
            rewards.append(np.mean(ep_rewards))
        else:
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

        # Save checkpoint after each episode
        os.makedirs(checkpoint_path, exist_ok=True)
        if use_marl:
            for i, ag in enumerate(agents):
                ag.save(os.path.join(checkpoint_path, f'dqn_traffic_agent_{i}.npz'))
        else:
            agent.save(os.path.join(checkpoint_path, 'dqn_traffic.npz'))
        np.save(os.path.join(checkpoint_path, 'rewards.npy'), np.array(rewards))

    # Save model
    if use_marl:
        for i, ag in enumerate(agents):
            model_path = os.path.join(out_dir, f'dqn_traffic_agent_{i}.npz')
            ag.save(model_path)
            print(f"Saved model for agent {i} to {model_path}")
    else:
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
    parser.add_argument('--marl', action='store_true', help='Use MARL environment for network coordination')
    args = parser.parse_args()
    train(args.config, args.episodes, args.out, args.use_sumo, args.marl)
