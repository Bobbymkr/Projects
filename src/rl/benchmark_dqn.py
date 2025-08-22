import time
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.env.sumo_env import SumoEnv  # Use single-agent env
from dqn_agent import DQNAgent, DQNConfig  # Custom DQN

# Function to train and benchmark SB3 DQN
def benchmark_sb3(env, total_steps=10000):
    start_time = time.time()
    model = DQN('MlpPolicy', env, verbose=0)
    model.learn(total_timesteps=total_steps)
    end_time = time.time()
    # Get average reward (simple evaluation)
    rewards = []
    for _ in range(10):
        obs, _ = env.reset()
        terminated = truncated = False
        ep_reward = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
    avg_reward = np.mean(rewards)
    return end_time - start_time, avg_reward

# Function to train and benchmark custom DQN
def benchmark_custom(env, total_steps=10000):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, DQNConfig())
    start_time = time.time()
    obs, _ = env.reset()
    for step in range(total_steps):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.push(obs, action, reward, next_obs, done)
        loss = agent.train_step()
        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs
    end_time = time.time()
    # Get average reward
    rewards = []
    for _ in range(10):
        obs, _ = env.reset()
        terminated = truncated = False
        ep_reward = 0
        while not (terminated or truncated):
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
    avg_reward = np.mean(rewards)
    return end_time - start_time, avg_reward

if __name__ == '__main__':
    config = {
        'sumocfg': 'c:\\Users\\Admin\\OneDrive\\Desktop\\5th_Sem\\Projects\\adaptive_traffic\\configs\\sumo.sumocfg',
        'min_green': 5,
        'max_green': 60,
        'green_step': 5,
        'cycle_yellow': 3,
        'cycle_all_red': 1,
        'queue_capacity': 40,
        'reward_weights': {'queue': -1.0, 'wait_penalty': -0.1}
    }
    env = SumoEnv(config)  # Use single env for both
    time_sb3, reward_sb3 = benchmark_sb3(env, 1000)  # Small number for quick benchmark
    time_custom, reward_custom = benchmark_custom(env, 1000)  # Note: custom uses single env
    print(f'SB3: Time={time_sb3:.2f}s, Avg Reward={reward_sb3:.2f}')
    print(f'Custom: Time={time_custom:.2f}s, Avg Reward={reward_custom:.2f}')