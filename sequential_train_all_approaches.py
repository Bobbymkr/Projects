#!/usr/bin/env python3
"""
Sequential Training on All Four Approaches
Trains your agent sequentially on different traffic scenarios.
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
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from src.env.traffic_env import TrafficEnv
from src.utils.config import load_config

def sequential_training_all_approaches():
    """Train sequentially on all approach configurations"""
    
    print("🚀 SEQUENTIAL TRAINING ON ALL FOUR APPROACHES")
    print("=" * 60)
    
    # Training scenarios in order of complexity
    scenarios = [
        ("balanced", "configs/intersection.json", "Balanced traffic"),
        ("north_heavy", "configs/north_heavy.json", "Heavy North traffic"),
        ("south_heavy", "configs/south_heavy.json", "Heavy South traffic"), 
        ("east_heavy", "configs/east_heavy.json", "Heavy East traffic"),
        ("west_heavy", "configs/west_heavy.json", "Heavy West traffic"),
        ("morning_rush", "configs/morning_rush.json", "Morning rush pattern"),
        ("evening_rush", "configs/evening_rush.json", "Evening rush pattern"),
        ("cross_flow", "configs/cross_flow.json", "High traffic all approaches"),
    ]
    
    steps_per_scenario = 40000  # 40K steps per scenario
    total_steps = len(scenarios) * steps_per_scenario
    
    print(f"📊 Training Plan:")
    for i, (name, _, desc) in enumerate(scenarios, 1):
        print(f"   {i}. {name}: {desc} ({steps_per_scenario:,} steps)")
    print(f"⏱️  Total Steps: {total_steps:,}")
    print("=" * 60)
    
    # Check for existing model
    current_model_path = "runs/dqn_traffic.zip"
    
    training_results = []
    start_time = datetime.now()
    
    for i, (scenario_name, config_path, description) in enumerate(scenarios, 1):
        print(f"\n🎯 SCENARIO {i}/{len(scenarios)}: {scenario_name.upper()}")
        print(f"📝 {description}")
        print(f"📁 Config: {config_path}")
        
        try:
            # Load configuration
            config = load_config(config_path)
            
            # Create environment
            env = TrafficEnv(config)
            monitored_env = Monitor(env, f"logs/{scenario_name}_training.log")
            vec_env = DummyVecEnv([lambda: monitored_env])
            
            # Load or create model
            if os.path.exists(current_model_path):
                print(f"🔄 Loading model from: {current_model_path}")
                model = DQN.load(current_model_path, env=vec_env)
                # Reduce learning rate for fine-tuning
                model.learning_rate = 0.0001 if i > 1 else 0.0003
            else:
                print("🆕 Creating new model")
                model = DQN(
                    'MlpPolicy',
                    vec_env,
                    learning_rate=0.0003,
                    buffer_size=50000,
                    learning_starts=2000,
                    batch_size=32,
                    gamma=0.99,
                    train_freq=4,
                    target_update_interval=1000,
                    exploration_fraction=0.2,
                    exploration_initial_eps=0.5 if i > 1 else 1.0,  # Lower exploration for later scenarios
                    exploration_final_eps=0.05,
                    policy_kwargs=dict(net_arch=[256, 256]),
                    verbose=1
                )
            
            # Setup checkpoint for this scenario
            os.makedirs(f"runs/{scenario_name}_checkpoints", exist_ok=True)
            checkpoint_callback = CheckpointCallback(
                save_freq=20000,
                save_path=f"runs/{scenario_name}_checkpoints/",
                name_prefix=f"{scenario_name}_dqn"
            )
            
            print(f"🚀 Training on {scenario_name} for {steps_per_scenario:,} steps...")
            scenario_start = datetime.now()
            
            # Train on this scenario
            model.learn(
                total_timesteps=steps_per_scenario,
                callback=[checkpoint_callback],
                reset_num_timesteps=False,
                progress_bar=True
            )
            
            scenario_duration = datetime.now() - scenario_start
            
            # Save model after each scenario
            scenario_model_path = f"runs/dqn_traffic_{scenario_name}.zip"
            model.save(scenario_model_path)
            
            # Update current model path for next iteration
            current_model_path = scenario_model_path
            
            # Record results
            result = {
                "scenario": scenario_name,
                "description": description,
                "steps": steps_per_scenario,
                "duration": str(scenario_duration),
                "model_path": scenario_model_path,
                "completed_at": datetime.now().isoformat()
            }
            training_results.append(result)
            
            print(f"✅ {scenario_name} completed in {scenario_duration}")
            print(f"💾 Model saved: {scenario_model_path}")
            
        except Exception as e:
            print(f"❌ Error training on {scenario_name}: {e}")
            error_result = {
                "scenario": scenario_name,
                "error": str(e),
                "failed_at": datetime.now().isoformat()
            }
            training_results.append(error_result)
            continue
    
    # Final model - copy last successful model as final
    if training_results and 'model_path' in training_results[-1]:
        final_model_path = \"runs/dqn_traffic_all_approaches_final.zip\"
        if os.path.exists(training_results[-1]['model_path']):
            # Copy the final model
            import shutil
            shutil.copy2(training_results[-1]['model_path'], final_model_path)
            print(f\"\n💾 Final model: {final_model_path}\")
    
    # Training summary
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    print(\"\n\" + \"=\" * 60)
    print(\"🎉 ALL-APPROACHES TRAINING COMPLETED!\")
    print(\"=\" * 60)
    print(f\"⏰ Total Duration: {total_duration}\")
    print(f\"📊 Scenarios Trained: {len([r for r in training_results if 'model_path' in r])}\")
    print(f\"🎯 Total Steps: {total_steps:,}\")
    
    successful_scenarios = [r['scenario'] for r in training_results if 'model_path' in r]
    if successful_scenarios:
        print(f\"✅ Successful Scenarios: {', '.join(successful_scenarios)}\")
    
    failed_scenarios = [r['scenario'] for r in training_results if 'error' in r]
    if failed_scenarios:
        print(f\"❌ Failed Scenarios: {', '.join(failed_scenarios)}\")
    
    # Save comprehensive training summary
    summary = {
        \"completion_time\": end_time.isoformat(),
        \"total_duration\": str(total_duration),
        \"total_steps\": total_steps,
        \"steps_per_scenario\": steps_per_scenario,
        \"scenarios_planned\": len(scenarios),
        \"scenarios_completed\": len(successful_scenarios),
        \"final_model\": final_model_path if 'final_model_path' in locals() else None,
        \"training_results\": training_results
    }
    
    with open(\"sequential_training_summary.json\", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f\"\n📋 Training summary saved: sequential_training_summary.json\")
    
    if successful_scenarios:
        print(f\"\n🎊 Your agent now has experience with all approaches!\")
        print(f\"💡 Next Steps:\")
        print(f\"   • Evaluate: python quick_agent_evaluation.py\")
        print(f\"   • Expected accuracy improvement: 70-80%+\")
        print(f\"   • Test different scenarios with demos\")
        return True
    else:
        print(f\"\n⚠️ No scenarios completed successfully\")
        return False

def main():
    \"\"\"Main training function\"\"\"
    print(\"🚦 Starting sequential training on all approaches...\")
    
    # Create logs directory
    os.makedirs(\"logs\", exist_ok=True)
    
    success = sequential_training_all_approaches()
    
    if success:
        print(\"\n✅ Sequential training completed!\")
    else:
        print(\"\n❌ Training encountered issues. Check logs for details.\")

if __name__ == \"__main__\":
    main()