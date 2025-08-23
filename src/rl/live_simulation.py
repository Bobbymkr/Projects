"""
Live Traffic Simulation Visualization
This script provides a real-time visual simulation of the traffic intersection
showing vehicle queues, signal phases, and DQN agent decisions.
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


def create_intersection_visualization(width: int = 800, height: int = 600):
    """Create a visual representation of the traffic intersection."""
    # Create a white background
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw intersection roads
    road_width = 100
    road_color = (100, 100, 100)  # Gray
    
    # Horizontal road (East-West)
    cv2.rectangle(canvas, (0, height//2 - road_width//2), (width, height//2 + road_width//2), road_color, -1)
    
    # Vertical road (North-South)
    cv2.rectangle(canvas, (width//2 - road_width//2, 0), (width//2 + road_width//2, height), road_color, -1)
    
    # Draw intersection center
    center_size = 20
    cv2.rectangle(canvas, (width//2 - center_size, height//2 - center_size), 
                 (width//2 + center_size, height//2 + center_size), (50, 50, 50), -1)
    
    return canvas


def draw_vehicle_queues(canvas: np.ndarray, queues: np.ndarray, phase: int, green_time: int):
    """Draw vehicle queues on the intersection visualization."""
    width, height = canvas.shape[1], canvas.shape[0]
    lane_width = 25
    vehicle_size = 8
    
    # Define lane positions (North, South, East, West)
    lane_positions = [
        # North lane (top)
        {'x': width//2 - lane_width, 'y': 50, 'dx': 0, 'dy': 1, 'color': (0, 255, 0) if phase == 0 else (255, 0, 0)},
        # South lane (bottom) 
        {'x': width//2 + lane_width, 'y': height - 50, 'dx': 0, 'dy': -1, 'color': (0, 255, 0) if phase == 0 else (255, 0, 0)},
        # East lane (right)
        {'x': width - 50, 'y': height//2 - lane_width, 'dx': -1, 'dy': 0, 'color': (0, 255, 0) if phase == 1 else (255, 0, 0)},
        # West lane (left)
        {'x': 50, 'y': height//2 + lane_width, 'dx': 1, 'dy': 0, 'color': (0, 255, 0) if phase == 1 else (255, 0, 0)}
    ]
    
    # Draw vehicles for each lane
    for i, (queue_count, lane) in enumerate(zip(queues, lane_positions)):
        # Limit queue display to prevent overcrowding
        display_count = min(queue_count, 15)
        
        for j in range(display_count):
            # Calculate vehicle position
            offset = j * (vehicle_size + 2)
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
    signal_x, signal_y = width - 100, 100
    signal_color = (0, 255, 0) if green_time > 0 else (0, 0, 255)  # Green if active, red if not
    cv2.circle(canvas, (signal_x, signal_y), 15, signal_color, -1)
    cv2.circle(canvas, (signal_x, signal_y), 15, (0, 0, 0), 2)
    
    return canvas


def draw_info_panel(canvas: np.ndarray, queues: np.ndarray, phase: int, green_time: int, 
                   reward: float, step: int, total_time: float):
    """Draw information panel on the visualization."""
    # Create info panel background
    panel_height = 120
    panel = np.ones((panel_height, canvas.shape[1], 3), dtype=np.uint8) * 240
    
    # Add text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (0, 0, 0)
    thickness = 2
    
    # Queue information
    cv2.putText(panel, f"Queue Lengths: N={queues[0]} S={queues[1]} E={queues[2]} W={queues[3]}", 
                (10, 25), font, font_scale, color, thickness)
    
    # Phase and timing information
    phase_text = f"Phase: {phase} ({'North-South' if phase == 0 else 'East-West'})"
    cv2.putText(panel, phase_text, (10, 50), font, font_scale, color, thickness)
    
    # Green time information
    green_text = f"Green Time: {green_time}s" if green_time > 0 else "Red Light"
    cv2.putText(panel, green_text, (10, 75), font, font_scale, color, thickness)
    
    # Performance metrics
    cv2.putText(panel, f"Reward: {reward:.1f} | Step: {step} | Time: {total_time:.1f}s", 
                (10, 100), font, font_scale, color, thickness)
    
    # Combine panel with main canvas
    combined = np.vstack([canvas, panel])
    return combined


def run_live_simulation(cfg_path: str, model_path: str = None, steps: int = 100, delay: float = 0.5):
    """Run a live visual simulation of the traffic intersection."""
    # Load configuration and environment
    cfg = load_config(cfg_path)
    env = TrafficEnv(cfg)
    
    # Load DQN agent if model path provided
    agent = None
    if model_path:
        agent = DQNAgent(state_dim=env.observation_space.shape[0], 
                        action_dim=env.action_space.n, cfg=DQNConfig())
        agent.load(model_path)
        print(f"Loaded DQN model from {model_path}")
    else:
        print("Running with random actions (no model loaded)")
    
    # Initialize environment
    obs, _ = env.reset()
    
    # Create visualization window
    window_name = "Adaptive Traffic Signal Control - Live Simulation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 750)
    
    print("Live simulation started! Press 'q' to quit, 'p' to pause/resume")
    print("Green circles = vehicles, Green signal = active phase, Red signal = inactive phase")
    
    paused = False
    step_count = 0
    start_time = time.time()
    
    try:
        while step_count < steps:
            if not paused:
                # Select action
                if agent is not None:
                    action = agent.select_action(obs.astype(np.float32), evaluate=True)
                else:
                    action = env.action_space.sample()
                
                # Take step in environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Get current state information
                current_phase = info.get("phase", 0)
                green_time = env.green_values[action] if info.get("green_active", False) else 0
                total_time = time.time() - start_time
                
                # Create visualization
                canvas = create_intersection_visualization()
                canvas = draw_vehicle_queues(canvas, next_obs, current_phase, green_time)
                canvas = draw_info_panel(canvas, next_obs, current_phase, green_time, 
                                       reward, step_count, total_time)
                
                # Display the visualization
                cv2.imshow(window_name, canvas)
                
                # Update state
                obs = next_obs
                step_count += 1
                
                # Check for termination
                if terminated or truncated:
                    obs, _ = env.reset()
            
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
        print(f"Simulation completed! Total steps: {step_count}")


def main():
    parser = argparse.ArgumentParser(description="Live Traffic Simulation Visualization")
    parser.add_argument('--config', default='configs/intersection.json', 
                       help='Traffic configuration file')
    parser.add_argument('--model', default=None, 
                       help='Path to trained DQN model (optional)')
    parser.add_argument('--steps', type=int, default=200, 
                       help='Number of simulation steps')
    parser.add_argument('--delay', type=float, default=0.3, 
                       help='Delay between steps in seconds')
    
    args = parser.parse_args()
    
    run_live_simulation(args.config, args.model, args.steps, args.delay)


if __name__ == "__main__":
    main()

