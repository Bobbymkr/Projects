import json
import os
import argparse
import numpy as np
from tqdm import trange
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import optuna
from optuna.integration import SB3OptunaCallback

from src.env.traffic_env import TrafficEnv
from src.env.sumo_env import SumoEnv
from src.env.marl_env import MarlEnv
from src.rl.dqn_agent import DQNAgent, DQNConfig
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def load_config(path: str):
    # Detailed comment: Load configuration from a JSON file.
    # Parameters:
    # - path: Path to the configuration file.
    # Returns: Dictionary of configuration settings.
    with open(path, 'r') as f:
        return json.load(f)


def make_env(cfg_path: str, use_sumo: bool, use_marl: bool):
    # Detailed comment: Create the appropriate environment based on flags.
    # Parameters:
    # - cfg_path: Path to configuration.
    # - use_sumo: Flag to use SUMO environment.
    # - use_marl: Flag to use MARL environment.
    # Returns: Instantiated environment.
    if use_marl:
        return MarlEnv(config_path=cfg_path)
    cfg = load_config(cfg_path)
    if use_sumo:
        return SumoEnv(cfg)
    return TrafficEnv(cfg)


def train(cfg_path: str, episodes: int, out_dir: str, use_sumo: bool, use_marl: bool, tune: bool = False, n_envs: int = 1):
    # Detailed comment: Train the DQN model(s) with optional tuning and parallelization.
    # Parameters:
    # - cfg_path: Configuration path.
    # - episodes: Number of training episodes.
    # - out_dir: Output directory for saves.
    # - use_sumo: Use SUMO env.
    # - use_marl: Use MARL setup.
    # - tune: Perform hyperparameter tuning.
    # - n_envs: Number of parallel environments (single-agent only).
    os.makedirs(out_dir, exist_ok=True)
    if use_marl:
        if n_envs > 1:
            print("Warning: Parallelization not supported for MARL yet.")
        env = make_env(cfg_path, use_sumo, use_marl)
    else:
        def create_env():
            return make_env(cfg_path, use_sumo, use_marl)
        if n_envs > 1:
            env = SubprocVecEnv([create_env for _ in range(n_envs)])
        else:
            env = DummyVecEnv([create_env])
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

    checkpoint_path = os.path.join(out_dir, 'checkpoint')
    checkpoint_callback = CheckpointCallback(save_freq=1, save_path=checkpoint_path, name_prefix='dqn_model')

    if use_marl:
        num_agents = env.num_agents
        agents = []
        start_episode = 0
        rewards = []
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            for i in range(num_agents):
                agent_env = DummyVecEnv([lambda: env])  # Placeholder, needs proper env wrapper for MARL
                agent = DQN.load(os.path.join(checkpoint_path, f'dqn_traffic_agent_{i}'))
                agents.append(agent)
            rewards = np.load(os.path.join(checkpoint_path, 'rewards.npy')).tolist()
            start_episode = len(rewards)
        else:
            for i in range(num_agents):
                agent_env = DummyVecEnv([lambda: env])  # Adapt for per-agent env if needed
                agent = DQN("MlpPolicy", agent_env, verbose=1, tensorboard_log=os.path.join(out_dir, 'tensorboard_logs'))
                agents.append(agent)

        for ep in trange(start_episode, episodes, desc="Training"):
            states = env.reset()
            ep_rewards = [0.0] * num_agents
            done = False
            while not done:
                actions = [ag.predict(st)[0] for ag, st in zip(agents, states)]
                next_states, rews, dones, _ = env.step(actions)
                done = any(dones)
                # Train each agent individually - this may need a custom loop for MARL
                ep_rewards = [ep_rewards[i] + rews[i] for i in range(num_agents)]
                states = next_states
            rewards.append(np.mean(ep_rewards))
            # Save using SB3 method
            for i, ag in enumerate(agents):
                ag.save(os.path.join(checkpoint_path, f'dqn_traffic_agent_{i}'))
            np.save(os.path.join(checkpoint_path, 'rewards.npy'), np.array(rewards))
    else:
        model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join(out_dir, 'tensorboard_logs'))
        if os.path.exists(os.path.join(checkpoint_path, 'dqn_model.zip')):
            model = DQN.load(os.path.join(checkpoint_path, 'dqn_model'))
            rewards = np.load(os.path.join(checkpoint_path, 'rewards.npy')).tolist()
            start_episode = len(rewards)
        else:
            start_episode = 0
            rewards = []

        model.learn(total_timesteps=episodes * 1000, callback=checkpoint_callback)  # Assume 1000 steps per episode
        rewards.extend([0] * (episodes - start_episode))  # Placeholder, extract rewards from logger or callback
        model.save(os.path.join(out_dir, 'dqn_traffic'))

    np.save(os.path.join(out_dir, 'rewards.npy'), np.array(rewards))
    print(f"Average reward over {episodes} episodes: {np.mean(rewards):.2f}")


def objective(trial):
    # Detailed comment: Objective function for Optuna hyperparameter tuning.
    # Parameters:
    # - trial: Optuna trial object.
    # Returns: Evaluation metric (total reward).
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    buffer_size = trial.suggest_int('buffer_size', 10000, 1000000, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    gamma = trial.suggest_float('gamma', 0.9, 0.999)

    if use_marl:
        # For MARL, tune shared params and train agents
        agents = []
        ep_rewards = []
        for i in range(env.num_agents):
            agent_env = DummyVecEnv([lambda: env])
            agent = DQN('MlpPolicy', agent_env, learning_rate=learning_rate, buffer_size=buffer_size, batch_size=batch_size, gamma=gamma, verbose=0, tensorboard_log="logs/optuna/")
            agent.learn(total_timesteps=10000)  # Reduced for tuning
            agents.append(agent)
        # Simulate one episode to evaluate
        states = env.reset()
        done = False
        total_reward = 0
        while not done:
            actions = [ag.predict(st)[0] for ag, st in zip(agents, states)]
            next_states, rews, dones, _ = env.step(actions)
            total_reward += sum(rews)
            states = next_states
            done = any(dones)
        return total_reward
    else:
        model = DQN('MlpPolicy', env, learning_rate=learning_rate, buffer_size=buffer_size, batch_size=batch_size, gamma=gamma, verbose=0, tensorboard_log="logs/optuna/")
        model.learn(total_timesteps=10000)  # Reduced for tuning
        # Evaluate
        eval_env = make_env(cfg_path, use_sumo, use_marl)
        obs = eval_env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = eval_env.step(action)
            total_reward += reward
        return total_reward

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print('Best hyperparameters:', study.best_params)
# Use best params for full training
best_params = study.best_params
else:
    best_params = {}  # Default params

# Proceed with training using best_params
if use_marl:
    num_agents = env.num_agents
    agents = []
    for i in range(num_agents):
        agent_env = DummyVecEnv([lambda: env])
        agent = DQN('MlpPolicy', agent_env, **best_params, verbose=1, tensorboard_log=os.path.join(out_dir, 'tensorboard_logs'))
        agents.append(agent)
    for ep in trange(start_episode, episodes, desc="Training"):
        states = env.reset()
        ep_rewards = [0.0] * num_agents
        done = False
        while not done:
            actions = [ag.predict(st)[0] for ag, st in zip(agents, states)]
            next_states, rews, dones, _ = env.step(actions)
            done = any(dones)
            # Train each agent individually - this may need a custom loop for MARL
            ep_rewards = [ep_rewards[i] + rews[i] for i in range(num_agents)]
            states = next_states
        rewards.append(np.mean(ep_rewards))
        # Save using SB3 method
        for i, ag in enumerate(agents):
            ag.save(os.path.join(checkpoint_path, f'dqn_traffic_agent_{i}'))
        np.save(os.path.join(checkpoint_path, 'rewards.npy'), np.array(rewards))
else:
    model = DQN('MlpPolicy', env, **best_params, verbose=1, tensorboard_log=os.path.join(out_dir, 'tensorboard_logs'))
    if os.path.exists(os.path.join(checkpoint_path, 'dqn_model.zip')):
        model = DQN.load(os.path.join(checkpoint_path, 'dqn_model'))
        rewards = np.load(os.path.join(checkpoint_path, 'rewards.npy')).tolist()
        start_episode = len(rewards)
    else:
        start_episode = 0
        rewards = []

    model.learn(total_timesteps=episodes * 1000, callback=checkpoint_callback)  # Assume 1000 steps per episode
    rewards.extend([0] * (episodes - start_episode))  # Placeholder, extract rewards from logger or callback
    model.save(os.path.join(out_dir, 'dqn_traffic'))

    np.save(os.path.join(out_dir, 'rewards.npy'), np.array(rewards))
    print(f"Average reward over {episodes} episodes: {np.mean(rewards):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/intersection.json')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--out', default='runs')
    parser.add_argument('--use_sumo', action='store_true', help='Use SUMO-based environment instead of custom simulator')
    parser.add_argument('--marl', action='store_true', help='Use MARL environment for network coordination')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning with Optuna')
    parser.add_argument('--n_envs', type=int, default=1, help='Number of parallel environments (for single-agent only)')
    args = parser.parse_args()
    train(args.config, args.episodes, args.out, args.use_sumo, args.marl, args.tune)
