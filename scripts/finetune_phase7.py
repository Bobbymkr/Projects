"""
Phase 7: Fine-tuning Script

Fine-tunes pre-trained models on target intersections (10K episodes).
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
    FineTuningFramework,
    FineTuningConfig,
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
        merged_config = {**base_config, **config}
        return TrafficEnv(config=merged_config)
    return env_factory


def main():
    parser = argparse.ArgumentParser(description="Phase 7: Fine-tuning on Target Intersection")
    parser.add_argument(
        "--pretrained-model",
        type=str,
        required=True,
        help="Path to pre-trained model"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["PPO", "SAC", "Rainbow DQN"],
        help="Algorithm type"
    )
    parser.add_argument(
        "--target-config",
        type=str,
        required=True,
        help="Target intersection configuration"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10000,
        help="Episodes for fine-tuning"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Fine-tuning learning rate"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Episodes between checkpoints"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/finetuned",
        help="Output directory for fine-tuned models"
    )
    
    args = parser.parse_args()
    
    # Load target config
    with open(args.target_config, 'r') as f:
        target_config = json.load(f)
    
    # Get agent class and config
    if args.algorithm == "PPO":
        agent_class = PPOAgent
        agent_config = PPOConfig(lr=args.learning_rate)
    elif args.algorithm == "SAC":
        agent_class = SACAgent
        agent_config = SACConfig(lr=args.learning_rate)
    elif args.algorithm == "Rainbow DQN":
        agent_class = RainbowDQNAgent
        agent_config = RainbowDQNConfig(lr=args.learning_rate)
    
    # Create fine-tuning framework
    finetuning_config = FineTuningConfig(
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        checkpoint_interval=args.checkpoint_interval,
        save_dir=args.output,
    )
    
    framework = FineTuningFramework(
        pretrained_model_path=args.pretrained_model,
        agent_class=agent_class,
        agent_config=agent_config,
        target_env_config=target_config,
        finetuning_config=finetuning_config,
    )
    
    # Create environment factory
    env_factory = create_env_factory(target_config)
    
    # Run fine-tuning
    logger.info(f"Starting fine-tuning for {args.algorithm}")
    stats = framework.finetune(env_factory)
    
    # Save statistics
    stats_path = Path(args.output) / "finetuning_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Fine-tuning complete! Statistics saved to: {stats_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

