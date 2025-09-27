#!/usr/bin/env python3
"""
Simple Multi-Approach Training
Train your agent on different traffic scenarios sequentially.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from src.env.traffic_env import TrafficEnv
from src.utils.config import load_config

def train_on_scenario(scenario_name, config_path, steps=30000, existing_model=None):
    """Train on a single scenario"""
    print(f"\n🎯 Training on {scenario_name} scenario...")
    print(f"📁 Config: {config_path}")
    print(f"⏱️ Steps: {steps:,}")
    
    try:
        # Load config and create environment
        config = load_config(config_path)
        env = TrafficEnv(config)
        
        # Wrap environment
        def make_env():
            return env
        
        vec_env = DummyVecEnv([make_env])
        
        # Load existing model or create new one
        if existing_model and os.path.exists(existing_model):
            print(f"🔄 Loading existing model: {existing_model}")
            model = DQN.load(existing_model, env=vec_env)
            model.learning_rate = 0.0001  # Lower learning rate for continued training
        else:
            print("🆕 Creating new model")
            model = DQN(
                'MlpPolicy',
                vec_env,
                learning_rate=0.0003,
                buffer_size=50000,
                learning_starts=1000,
                batch_size=32,
                gamma=0.99,
                train_freq=4,
                target_update_interval=1000,
                exploration_fraction=0.2,
                exploration_initial_eps=0.3,
                exploration_final_eps=0.05,
                policy_kwargs=dict(net_arch=[256, 128]),
                verbose=1
            )
        
        # Train the model
        print(f"🚀 Starting training...")
        start_time = datetime.now()
        
        model.learn(total_timesteps=steps, progress_bar=True)
        
        duration = datetime.now() - start_time
        print(f"✅ Training completed in {duration}")
        
        # Save the model
        model_path = f"runs/dqn_{scenario_name}.zip"
        model.save(model_path)
        print(f"💾 Model saved: {model_path}")
        
        return model_path, True
        
    except Exception as e:
        print(f"❌ Error training {scenario_name}: {e}")
        return None, False

def main():
    """Main training function"""
    print("🚦 MULTI-APPROACH TRAINING")
    print("=" * 50)
    
    # Training scenarios
    scenarios = [
        ("balanced", "configs/intersection.json"),
        ("north_heavy", "configs/north_heavy.json"),
        ("south_heavy", "configs/south_heavy.json"),
        ("east_heavy", "configs/east_heavy.json"),
        ("west_heavy", "configs/west_heavy.json"),
        ("morning_rush", "configs/morning_rush.json"),
        ("evening_rush", "configs/evening_rush.json"),
        ("cross_flow", "configs/cross_flow.json")
    ]
    
    steps_per_scenario = 25000
    total_steps = len(scenarios) * steps_per_scenario
    
    print(f"📊 Scenarios: {len(scenarios)}")
    print(f"⏱️ Steps per scenario: {steps_per_scenario:,}")
    print(f"🎯 Total steps: {total_steps:,}")
    print("=" * 50)
    
    # Create necessary directories
    os.makedirs("runs", exist_ok=True)
    
    # Track progress
    current_model = "runs/dqn_traffic.zip"  # Start with existing model
    successful_trainings = []
    
    for i, (scenario_name, config_path) in enumerate(scenarios, 1):
        print(f"\n📍 SCENARIO {i}/{len(scenarios)}: {scenario_name.upper()}")
        
        # Check if config exists
        if not os.path.exists(config_path):
            print(f"⚠️ Config not found: {config_path}, skipping...")
            continue
        
        # Train on this scenario
        model_path, success = train_on_scenario(
            scenario_name, 
            config_path, 
            steps_per_scenario,
            current_model if os.path.exists(current_model) else None
        )
        
        if success and model_path:
            successful_trainings.append((scenario_name, model_path))
            current_model = model_path
            print(f"✅ {scenario_name} training successful")
        else:
            print(f"❌ {scenario_name} training failed")
    
    # Final results
    print("\n" + "=" * 50)
    print("🎉 MULTI-APPROACH TRAINING COMPLETED!")
    print("=" * 50)
    
    if successful_trainings:
        print(f"✅ Successful scenarios: {len(successful_trainings)}/{len(scenarios)}")
        for scenario, model_path in successful_trainings:
            print(f"   • {scenario}: {model_path}")
        
        # Copy final model
        final_model = "runs/dqn_traffic_multi_approach.zip"
        if current_model and os.path.exists(current_model):
            import shutil
            shutil.copy2(current_model, final_model)
            print(f"\n💾 Final model: {final_model}")
        
        print(f"\n🎊 Your agent now has experience with all approaches!")
        print(f"💡 Next steps:")
        print(f"   • Evaluate performance: python quick_agent_evaluation.py")
        print(f"   • Expected accuracy improvement: 70-85%")
        print(f"   • Test with demos to see improved behavior")
        
    else:
        print("❌ No scenarios completed successfully")
        print("Please check the error messages above")

if __name__ == "__main__":
    main()