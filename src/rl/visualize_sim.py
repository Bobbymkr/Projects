import argparse
import json
import os
import sys
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.env.traffic_env import TrafficEnv
from src.rl.dqn_agent import DQNAgent, DQNConfig


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def simulate(cfg_path: str, model_path: str | None, steps: int, seed: int) -> Dict[str, Any]:
    cfg = load_config(cfg_path)
    env = TrafficEnv(cfg)

    agent = None
    if model_path:
        from stable_baselines3 import DQN
        agent = DQN.load(model_path)

    obs, _ = env.reset(seed=seed)

    history: List[Dict[str, Any]] = []
    for _ in range(max(1, steps)):
        if agent is not None:
            action, _ = agent.predict(obs, deterministic=True)
        else:
            action = int(np.random.randint(env.action_space.n))

        next_obs, reward, terminated, truncated, info = env.step(action)
        history.append({
            "time": info.get("time", 0),
            "phase": info.get("phase", 0),
            "queues": next_obs.copy().tolist(),
            "reward": float(reward),
            "action_index": int(action),
            "green": int(env.green_values[action]),
        })
        obs = next_obs
        if terminated or truncated:
            break

    return {"env": env, "history": history}


def plot_timeseries(env: TrafficEnv, history: List[Dict[str, Any]], out_path: str):
    if not history:
        return
    queues_over_time = np.array([h["queues"] for h in history], dtype=int)
    times = [h["time"] for h in history]
    num_lanes = queues_over_time.shape[1]

    plt.figure(figsize=(10, 6))
    for lane in range(num_lanes):
        plt.plot(times, queues_over_time[:, lane], marker='o', label=f"Lane {lane}")

    plt.title("Queue lengths over time")
    plt.xlabel("Time (s)")
    plt.ylabel("Vehicles in queue")
    plt.ylim(0, max(1, env.queue_capacity))
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/intersection.json')
    parser.add_argument('--model', default=None)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--outdir', default='runs')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    sim = simulate(args.config, args.model, args.steps, args.seed)
    env = sim["env"]
    history = sim["history"]

    img_path = os.path.join(args.outdir, 'queue_timeseries.png')
    plot_timeseries(env, history, img_path)

    # Print a brief summary and the output path for the user
    print(f"Saved queue time-series visualization to {img_path}")
    if history:
        avg_reward = float(np.mean([h["reward"] for h in history]))
        print(f"Frames: {len(history)} | Final time: {history[-1]['time']}s | Avg reward/frame: {avg_reward:.2f}")


if __name__ == "__main__":
    main()


