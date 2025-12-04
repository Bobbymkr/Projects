"""
Phase 7: Large-Scale Pre-training Script

Pre-trains models on diverse synthetic scenarios (1M+ episodes)
before fine-tuning on target intersections.
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research.transfer_learning.phase7_transfer_learning import (
    PreTrainingFramework,
    PreTrainingConfig,
)
from src.env.traffic_env import TrafficEnv
from src.research.novel_algorithms.phase6_advanced_rl import (
    PPOAgent, PPOConfig,
    SACAgent, SACConfig,
    RainbowDQNAgent, RainbowDQNConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def create_env_factory(base_config: Dict[str, Any]):
    """Create environment factory function."""
    def env_factory(config: Dict[str, Any]):
        # Merge with base config
        merged_config = {**base_config, **config}
        return TrafficEnv(config=merged_config)
    return env_factory


def main():
    parser = argparse.ArgumentParser(description="Phase 7: Large-Scale Pre-training")
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["PPO", "SAC", "Rainbow DQN"],
        help="Algorithm to pre-train"
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/intersection.json",
        help="Base environment configuration"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100000,  # Reduced for testing, use 1000000 for full pre-training
        help="Total episodes for pre-training"
    )
    parser.add_argument(
        "--episodes-per-scenario",
        type=int,
        default=100,
        help="Episodes per scenario"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10000,
        help="Episodes between checkpoints"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/pretrained",
        help="Output directory for pre-trained models"
    )
    
    args = parser.parse_args()
    
    # Load base config
    with open(args.base_config, 'r') as f:
        base_config = json.load(f)
    
    # Create environment factory
    env_factory = create_env_factory(base_config)
    
    # Get agent class and config
    if args.algorithm == "PPO":
        agent_class = PPOAgent
        agent_config = PPOConfig()
    elif args.algorithm == "SAC":
        agent_class = SACAgent
        agent_config = SACConfig()
    elif args.algorithm == "Rainbow DQN":
        agent_class = RainbowDQNAgent
        agent_config = RainbowDQNConfig()
    
    # Create pre-training framework
    pretraining_config = PreTrainingConfig(
        total_episodes=args.episodes,
        episodes_per_scenario=args.episodes_per_scenario,
        checkpoint_interval=args.checkpoint_interval,
        save_dir=args.output,
    )
    
    framework = PreTrainingFramework(
        agent_class=agent_class,
        agent_config=agent_config,
        base_env_config=base_config,
        pretraining_config=pretraining_config,
    )
    
    # Run pre-training
    logger.info(f"Starting pre-training for {args.algorithm}")
    stats = framework.pretrain(env_factory)
    
    # Save statistics
    stats_path = Path(args.output) / "pretraining_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Pre-training complete! Statistics saved to: {stats_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

