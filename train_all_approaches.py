#!/usr/bin/env python3
"""
Comprehensive Training Script for All Four Approaches
This script trains your DQN agent on diverse traffic scenarios covering all approaches.
"""

import os
import sys
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from src.env.traffic_env import TrafficEnv
from src.utils.config import load_config

class MultiScenarioTraining:
    """Training manager for all four approaches with diverse scenarios"""
    
    def __init__(self):
        self.scenarios = {
            "balanced": "configs/intersection.json",
            "morning_rush": "configs/morning_rush.json", 
            "evening_rush": "configs/evening_rush.json"
        }
        
        # Additional scenario configs we'll create
        self.additional_scenarios = {
            "north_heavy": {
                "arrival_rates": [0.6, 0.2, 0.3, 0.25],  # North heavy
                "description": "Heavy North approach traffic"
            },
            "south_heavy": {
                "arrival_rates": [0.2, 0.6, 0.3, 0.25],  # South heavy
                "description": "Heavy South approach traffic"
            },
            "east_heavy": {
                "arrival_rates": [0.25, 0.2, 0.6, 0.3],  # East heavy
                "description": "Heavy East approach traffic"
            },
            "west_heavy": {
                "arrival_rates": [0.3, 0.25, 0.2, 0.6],  # West heavy
                "description": "Heavy West approach traffic"
            },
            "cross_flow": {
                "arrival_rates": [0.5, 0.5, 0.5, 0.5],  # Equal high traffic
                "description": "High traffic from all approaches"
            },
            "diagonal_flow": {
                "arrival_rates": [0.5, 0.2, 0.5, 0.2],  # North-East diagonal
                "description": "Diagonal traffic pattern"
            }
        }
        
        self.create_additional_configs()
    
    def create_additional_configs(self):
        """Create additional configuration files for comprehensive training"""
        print("🔧 Creating additional training scenarios...")
        
        # Load base config
        base_config = load_config("configs/intersection.json")
        
        for scenario_name, scenario_data in self.additional_scenarios.items():
            config_path = f"configs/{scenario_name}.json"
            
            # Update arrival rates
            new_config = base_config.copy()
            new_config["arrival_rates"] = scenario_data["arrival_rates"]
            
            # Save new config
            with open(config_path, 'w') as f:
                json.dump(new_config, f, indent=2)
            
            print(f"   ✅ Created {scenario_name}: {scenario_data['description']}")
            self.scenarios[scenario_name] = config_path
    
    def create_multi_scenario_env(self, scenario_rotation=True):
        """Create environment that rotates through different scenarios"""
        
        class MultiScenarioEnv:
            def __init__(self, scenarios, rotation=True):
                self.scenarios = list(scenarios.values())
                self.scenario_names = list(scenarios.keys())
                self.rotation = rotation
                self.current_scenario_idx = 0
                self.episode_count = 0
                
                # Load first scenario
                self.current_env = TrafficEnv(load_config(self.scenarios[0]))
                print(f"🚦 Starting with scenario: {self.scenario_names[0]}")
                
            def reset(self, **kwargs):
                # Rotate scenario every 50 episodes if rotation enabled
                if self.rotation and self.episode_count > 0 and self.episode_count % 50 == 0:
                    self.current_scenario_idx = (self.current_scenario_idx + 1) % len(self.scenarios)
                    scenario_name = self.scenario_names[self.current_scenario_idx]
                    config_path = self.scenarios[self.current_scenario_idx]
                    
                    print(f"🔄 Switching to scenario: {scenario_name} (Episode {self.episode_count})")
                    self.current_env = TrafficEnv(load_config(config_path))
                
                self.episode_count += 1
                return self.current_env.reset(**kwargs)
            
            def step(self, action):
                return self.current_env.step(action)
            
            def __getattr__(self, name):
                return getattr(self.current_env, name)
        
        return MultiScenarioEnv(self.scenarios, scenario_rotation)
    
    def train_comprehensive_agent(self, total_timesteps=500000, scenario_rotation=True):
        """Train agent on all approaches with comprehensive scenarios"""
        
        print("🚀 STARTING COMPREHENSIVE TRAINING ON ALL FOUR APPROACHES")
        print("=" * 70)
        print(f"📊 Training Scenarios: {len(self.scenarios)}")
        for name, desc in [(k, self.additional_scenarios.get(k, {}).get('description', 'Standard scenario')) 
                          for k in self.scenarios.keys()]:
            print(f"   • {name}: {desc}")
        
        print(f"⏱️  Total Training Steps: {total_timesteps:,}")
        print(f"🔄 Scenario Rotation: {'Enabled' if scenario_rotation else 'Disabled'}")
        print("=" * 70)
        
        # Create multi-scenario environment
        env = self.create_multi_scenario_env(scenario_rotation)
        env = Monitor(env, "logs/training_monitor.log")
        env = DummyVecEnv([lambda: env])
        
        # Model configuration
        model_config = {
            'policy': 'MlpPolicy',
            'env': env,
            'learning_rate': 0.0003,
            'buffer_size': 100000,
            'learning_starts': 10000,
            'batch_size': 32,
            'tau': 1.0,
            'gamma': 0.99,
            'train_freq': 4,
            'gradient_steps': 1,
            'target_update_interval': 1000,
            'exploration_fraction': 0.3,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.02,
            'policy_kwargs': dict(net_arch=[256, 256, 128]),
            'verbose': 1,
            'tensorboard_log': "./logs/tensorboard/"
        }
        
        # Check if we should continue from existing model
        existing_model = "runs/dqn_traffic.zip"
        if os.path.exists(existing_model):
            print(f"🔄 Continuing training from existing model: {existing_model}")
            model = DQN.load(existing_model, env=env)
            # Update learning rate for continued training
            model.learning_rate = 0.0001  # Lower learning rate for fine-tuning
        else:
            print("🆕 Creating new model for training")
            model = DQN(**model_config)
        
        # Callbacks for monitoring and checkpointing
        os.makedirs("logs", exist_ok=True)
        os.makedirs("runs/checkpoints", exist_ok=True)
        
        checkpoint_callback = CheckpointCallback(
            save_freq=25000,
            save_path="runs/checkpoints/",
            name_prefix="dqn_multi_scenario"
        )
        
        # Start training
        print(f"\n🎯 Beginning training for {total_timesteps:,} timesteps...")
        start_time = datetime.now()
        
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=[checkpoint_callback],
                reset_num_timesteps=False,  # Continue from existing training
                progress_bar=True
            )
            
            # Save the final model
            final_model_path = "runs/dqn_traffic_all_approaches.zip"
            model.save(final_model_path)
            
            # Training summary
            end_time = datetime.now()
            training_duration = end_time - start_time
            
            print("\n" + "=" * 70)
            print("🎉 COMPREHENSIVE TRAINING COMPLETED!")
            print("=" * 70)
            print(f"⏰ Training Duration: {training_duration}")
            print(f"💾 Model Saved: {final_model_path}")
            print(f"📊 Scenarios Trained: {len(self.scenarios)}")
            print(f"🎯 Total Timesteps: {total_timesteps:,}")
            
            # Save training summary
            training_summary = {
                "completion_time": end_time.isoformat(),
                "training_duration": str(training_duration),
                "total_timesteps": total_timesteps,
                "scenarios_used": list(self.scenarios.keys()),
                "model_path": final_model_path,
                "scenario_rotation": scenario_rotation
            }
            
            with open("training_summary_all_approaches.json", 'w') as f:
                json.dump(training_summary, f, indent=2)
            
            return model, final_model_path
            
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
            # Save intermediate model
            interrupted_model_path = "runs/dqn_traffic_interrupted.zip"
            model.save(interrupted_model_path)
            print(f"💾 Intermediate model saved: {interrupted_model_path}")
            return model, interrupted_model_path
        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            return None, None

def main():
    """Main training function"""
    
    print("🚦 ADAPTIVE TRAFFIC CONTROL - ALL APPROACHES TRAINING")
    print("=" * 60)
    
    # Initialize training manager
    trainer = MultiScenarioTraining()
    
    # Training options
    print("📋 Training Options:")
    print("1. Quick Training (100K steps) - ~30 minutes")
    print("2. Standard Training (500K steps) - ~2 hours") 
    print("3. Extended Training (1M steps) - ~4 hours")
    print("4. Custom Training (specify steps)")
    
    choice = input("\nSelect training option (1-4): ").strip()
    
    if choice == "1":
        timesteps = 100000
    elif choice == "2":
        timesteps = 500000
    elif choice == "3":
        timesteps = 1000000
    elif choice == "4":
        try:
            timesteps = int(input("Enter number of training steps: "))
        except ValueError:
            print("Invalid input. Using default 500K steps.")
            timesteps = 500000
    else:
        print("Invalid choice. Using default 500K steps.")
        timesteps = 500000
    
    # Ask about scenario rotation
    rotation_choice = input("\nEnable scenario rotation during training? (y/n): ").strip().lower()
    scenario_rotation = rotation_choice in ['y', 'yes', '1', 'true']
    
    print(f"\n🎯 Starting training with {timesteps:,} timesteps")
    print(f"🔄 Scenario rotation: {'Enabled' if scenario_rotation else 'Disabled'}")
    
    # Start training
    model, model_path = trainer.train_comprehensive_agent(
        total_timesteps=timesteps,
        scenario_rotation=scenario_rotation
    )
    
    if model and model_path:
        print(f"\n✅ Training completed successfully!")
        print(f"🎊 Your agent is now trained on all four approaches!")
        print(f"📁 Model saved at: {model_path}")
        print(f"\n💡 Next steps:")
        print(f"   • Run evaluation: python quick_agent_evaluation.py")
        print(f"   • Test with demos: python demo.py")
        print(f"   • Compare performance with previous model")
    else:
        print("\n❌ Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main()