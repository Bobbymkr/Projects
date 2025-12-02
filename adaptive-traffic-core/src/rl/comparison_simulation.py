"""
Side-by-Side Comparison Simulation

This script shows a real-time comparison between DQN agent and random actions
in a split-screen visualization.
"""

import json
import argparse
import time
import numpy as np
import cv2
from typing import Dict, Any, List

from src.env.traffic_env import TrafficEnv
from src.rl.dqn_agent import DQNAgent, DQNConfig


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration dictionary from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def create_intersection_visualization(width: int = 400, height: int = 300):
    """Create a visual representation of the traffic intersection."""
    # Create a white background
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw intersection roads
    road_width = 50
    road_color = (100, 100, 100)  # Gray
    
    # Horizontal road (East-West)
    cv2.rectangle(canvas, (0, height//2 - road_width//2), (width, height//2 + road_width//2), road_color, -1)
    
    # Vertical road (North-South)
    cv2.rectangle(canvas, (width//2 - road_width//2, 0), (width//2 + road_width//2, height), road_color, -1)
    
    # Draw intersection center
    center_size = 10
    cv2.rectangle(canvas, (width//2 - center_size, height//2 - center_size), 
                 (width//2 + center_size, height//2 + center_size), (50, 50, 50), -1)
    
    return canvas


def draw_vehicle_queues(canvas: np.ndarray, queues: np.ndarray, phase: int, green_time: int):
    """Draw vehicle queues on the intersection visualization."""
    width, height = canvas.shape[1], canvas.shape[0]
    lane_width = 12
    vehicle_size = 4
    
    # Define lane positions (North, South, East, West)
    lane_positions = [
        # North lane (top)
        {'x': width//2 - lane_width, 'y': 25, 'dx': 0, 'dy': 1, 'color': (0, 255, 0) if phase == 0 else (255, 0, 0)},
        # South lane (bottom) 
        {'x': width//2 + lane_width, 'y': height - 25, 'dx': 0, 'dy': -1, 'color': (0, 255, 0) if phase == 0 else (255, 0, 0)},
        # East lane (right)
        {'x': width - 25, 'y': height//2 - lane_width, 'dx': -1, 'dy': 0, 'color': (0, 255, 0) if phase == 1 else (255, 0, 0)},
        # West lane (left)
        {'x': 25, 'y': height//2 + lane_width, 'dx': 1, 'dy': 0, 'color': (0, 255, 0) if phase == 1 else (255, 0, 0)}
    ]
    
    # Draw vehicles for each lane
    for i, (queue_count, lane) in enumerate(zip(queues, lane_positions)):
        # Limit queue display to prevent overcrowding
        display_count = min(queue_count, 8)
        
        for j in range(display_count):
            # Calculate vehicle position
            offset = j * (vehicle_size + 1)
            if lane['dx'] != 0:  # Horizontal lane
                x = lane['x'] + lane['dx'] * offset
                y = lane['y']
            else:  # Vertical lane
                x = lane['x']
                y = lane['y'] + lane['dy'] * offset
            
            # Draw vehicle
            cv2.circle(canvas, (int(x), int(y)), vehicle_size//2, lane['color'], -1)
            cv2.circle(canvas, (int(x), int(y)), vehicle_size//2, (0, 0, 0), 1)
    
    # Draw traffic signal
    signal_x, signal_y = width - 50, 50
    signal_color = (0, 255, 0) if green_time > 0 else (0, 0, 255)  # Green if active, red if not
    cv2.circle(canvas, (signal_x, signal_y), 8, signal_color, -1)
    cv2.circle(canvas, (signal_x, signal_y), 8, (0, 0, 0), 1)
    
    return canvas


def draw_info_panel(canvas: np.ndarray, queues: np.ndarray, phase: int, green_time: int, 
                   reward: float, step: int, agent_name: str):
    """Draw information panel on the visualization."""
    # Create info panel background
    panel_height = 80
    panel = np.ones((panel_height, canvas.shape[1], 3), dtype=np.uint8) * 240
    
    # Add text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (0, 0, 0)
    thickness = 1
    
    # Agent name
    cv2.putText(panel, f"{agent_name}", (10, 20), font, font_scale, color, thickness)
    
    # Queue information
    cv2.putText(panel, f"Queues: N={queues[0]} S={queues[1]} E={queues[2]} W={queues[3]}", 
                (10, 40), font, font_scale, color, thickness)
    
    # Phase and timing information
    phase_text = f"Phase: {phase} | Green: {green_time}s | Reward: {reward:.1f}"
    cv2.putText(panel, phase_text, (10, 60), font, font_scale, color, thickness)
    
    # Combine panel with main canvas
    combined = np.vstack([canvas, panel])
    return combined


def run_comparison_simulation(cfg_path: str, model_path: str, steps: int = 100, delay: float = 0.5):
    """Run a side-by-side comparison simulation."""
    # Load configuration and create two environments
    cfg = load_config(cfg_path)
    env_dqn = TrafficEnv(cfg)
    env_random = TrafficEnv(cfg)
    
    # Load DQN agent
    agent = DQNAgent(state_dim=env_dqn.observation_space.shape[0], 
                    action_dim=env_dqn.action_space.n, cfg=DQNConfig())
    agent.load(model_path)
    print(f"Loaded DQN model from {model_path}")
    
    # Initialize environments
    obs_dqn, _ = env_dqn.reset()
    obs_random, _ = env_random.reset()
    
    # Create visualization window
    window_name = "DQN vs Random Actions - Traffic Signal Control"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 500)
    
    print("Comparison simulation started! Press 'q' to quit, 'p' to pause/resume")
    print("Left: DQN Agent | Right: Random Actions")
    print("Green circles = vehicles, Green signal = active phase, Red signal = inactive phase")
    
    paused = False
    step_count = 0
    total_reward_dqn = 0
    total_reward_random = 0
    
    try:
        while step_count < steps:
            if not paused:
                # DQN agent action
                action_dqn = agent.select_action(obs_dqn.astype(np.float32), evaluate=True)
                next_obs_dqn, reward_dqn, terminated_dqn, truncated_dqn, info_dqn = env_dqn.step(action_dqn)
                
                # Random action
                action_random = np.random.randint(env_random.action_space.n)
                next_obs_random, reward_random, terminated_random, truncated_random, info_random = env_random.step(action_random)
                
                # Get current state information
                phase_dqn = info_dqn.get("phase", 0)
                green_time_dqn = env_dqn.green_values[action_dqn] if info_dqn.get("green_active", False) else 0
                
                phase_random = info_random.get("phase", 0)
                green_time_random = env_random.green_values[action_random] if info_random.get("green_active", False) else 0
                
                # Update total rewards
                total_reward_dqn += reward_dqn
                total_reward_random += reward_random
                
                # Create visualizations
                canvas_dqn = create_intersection_visualization()
                canvas_dqn = draw_vehicle_queues(canvas_dqn, next_obs_dqn, phase_dqn, green_time_dqn)
                canvas_dqn = draw_info_panel(canvas_dqn, next_obs_dqn, phase_dqn, green_time_dqn, 
                                           reward_dqn, step_count, "DQN Agent")
                
                canvas_random = create_intersection_visualization()
                canvas_random = draw_vehicle_queues(canvas_random, next_obs_random, phase_random, green_time_random)
                canvas_random = draw_info_panel(canvas_random, next_obs_random, phase_random, green_time_random, 
                                              reward_random, step_count, "Random Actions")
                
                # Combine side by side
                combined = np.hstack([canvas_dqn, canvas_random])
                
                # Add comparison header
                header_height = 40
                header = np.ones((header_height, combined.shape[1], 3), dtype=np.uint8) * 200
                cv2.putText(header, f"Step: {step_count} | DQN Total Reward: {total_reward_dqn:.1f} | Random Total Reward: {total_reward_random:.1f}", 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                combined = np.vstack([header, combined])
                
                # Display the visualization
                cv2.imshow(window_name, combined)
                
                # Update states
                obs_dqn = next_obs_dqn
                obs_random = next_obs_random
                step_count += 1
                
                # Check for termination
                if terminated_dqn or truncated_dqn:
                    obs_dqn, _ = env_dqn.reset()
                if terminated_random or truncated_random:
                    obs_random, _ = env_random.reset()
            
            # Handle key presses
            key = cv2.waitKey(int(delay * 1000)) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    
    finally:
        cv2.destroyAllWindows()
        print(f"Comparison completed!")
        print(f"DQN Agent Total Reward: {total_reward_dqn:.1f}")
        print(f"Random Actions Total Reward: {total_reward_random:.1f}")
        print(f"DQN Advantage: {total_reward_dqn - total_reward_random:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Side-by-Side Comparison Simulation")
    parser.add_argument('--config', default='configs/intersection.json', 
                       help='Traffic configuration file')
    parser.add_argument('--model', required=True, 
                       help='Path to trained DQN model')
    parser.add_argument('--steps', type=int, default=150, 
                       help='Number of simulation steps')
    parser.add_argument('--delay', type=float, default=0.4, 
                       help='Delay between steps in seconds')
    
    args = parser.parse_args()
    
    run_comparison_simulation(args.config, args.model, args.steps, args.delay)


if __name__ == "__main__":
    main()
