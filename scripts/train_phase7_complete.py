"""
Phase 7: Complete Transfer Learning Pipeline

Combines pre-training and fine-tuning in a single workflow.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research.transfer_learning.phase7_transfer_learning import (
    PreTrainingFramework,
    PreTrainingConfig,
    FineTuningFramework,
    FineTuningConfig,
)
from src.env.traffic_env import TrafficEnv
from src.research.novel_algorithms.phase6_advanced_rl import (
    PPOAgent, PPOConfig,
    SACAgent, SACConfig,
    RainbowDQNAgent, RainbowDQNConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_env_factory(base_config: dict):
    """Create environment factory function."""
    def env_factory(config: dict):
        merged_config = {**base_config, **config}
        return TrafficEnv(config=merged_config)
    return env_factory


def main():
    parser = argparse.ArgumentParser(description="Phase 7: Complete Transfer Learning Pipeline")
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["PPO", "SAC", "Rainbow DQN"],
        help="Algorithm to train"
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/intersection.json",
        help="Base environment configuration"
    )
    parser.add_argument(
        "--target-config",
        type=str,
        required=True,
        help="Target intersection configuration"
    )
    parser.add_argument(
        "--pretrain-episodes",
        type=int,
        default=100000,  # Reduced for testing
        help="Episodes for pre-training"
    )
    parser.add_argument(
        "--finetune-episodes",
        type=int,
        default=10000,
        help="Episodes for fine-tuning"
    )
    parser.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Skip pre-training (use existing model)"
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default=None,
        help="Path to existing pre-trained model (if skipping pre-training)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/phase7_transfer",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Load configs
    with open(args.base_config, 'r') as f:
        base_config = json.load(f)
    
    with open(args.target_config, 'r') as f:
        target_config = json.load(f)
    
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
    
    # Step 1: Pre-training (if not skipped)
    pretrained_model_path = args.pretrained_model
    
    if not args.skip_pretrain:
        logger.info("="*80)
        logger.info("Step 1: Large-Scale Pre-training")
        logger.info("="*80)
        
        # Create pre-training framework
        pretraining_config = PreTrainingConfig(
            total_episodes=args.pretrain_episodes,
            episodes_per_scenario=100,
            checkpoint_interval=10000,
            save_dir=str(Path(args.output) / "pretrained"),
        )
        
        framework = PreTrainingFramework(
            agent_class=agent_class,
            agent_config=agent_config,
            base_env_config=base_config,
            pretraining_config=pretraining_config,
        )
        
        # Create environment factory
        env_factory = create_env_factory(base_config)
        
        # Run pre-training
        stats = framework.pretrain(env_factory)
        
        # Find the final pre-trained model
        pretrained_dir = Path(args.output) / "pretrained"
        final_model = pretrained_dir / "pretrained_model_final.pt"
        if not final_model.exists():
            # Find latest checkpoint
            checkpoints = list(pretrained_dir.glob("pretrained_model_ep*.pt"))
            if checkpoints:
                final_model = max(checkpoints, key=lambda p: int(p.stem.split('_ep')[1].split('.')[0]))
        
        pretrained_model_path = str(final_model)
        logger.info(f"Pre-trained model: {pretrained_model_path}")
    
    if not pretrained_model_path or not Path(pretrained_model_path).exists():
        logger.error("No pre-trained model available. Please run pre-training first.")
        return 1
    
    # Step 2: Fine-tuning
    logger.info("\n" + "="*80)
    logger.info("Step 2: Fine-tuning on Target Intersection")
    logger.info("="*80)
    
    # Create fine-tuning framework
    finetuning_config = FineTuningConfig(
        episodes=args.finetune_episodes,
        learning_rate=1e-4,
        checkpoint_interval=1000,
        save_dir=str(Path(args.output) / "finetuned"),
    )
    
    framework = FineTuningFramework(
        pretrained_model_path=pretrained_model_path,
        agent_class=agent_class,
        agent_config=agent_config,
        target_env_config=target_config,
        finetuning_config=finetuning_config,
    )
    
    # Create environment factory
    env_factory = create_env_factory(target_config)
    
    # Run fine-tuning
    stats = framework.finetune(env_factory)
    
    logger.info("\n" + "="*80)
    logger.info("Phase 7 Transfer Learning Complete!")
    logger.info("="*80)
    logger.info(f"Pre-trained model: {pretrained_model_path}")
    logger.info(f"Fine-tuned model: {Path(args.output) / 'finetuned' / 'finetuned_model_final.pt'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
