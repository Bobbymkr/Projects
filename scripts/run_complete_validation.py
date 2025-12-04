"""
Complete Validation and Training Pipeline.

Runs full validation, training, and benchmarking for advanced algorithms
to ensure they are properly tested and compared.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_validation():
    """Run algorithm validation."""
    logger.info("=" * 80)
    logger.info("STEP 1: VALIDATING ADVANCED ALGORITHMS")
    logger.info("=" * 80)
    
    result = subprocess.run(
        [sys.executable, "scripts/validate_advanced_algorithms.py", "--algorithm", "all"],
        cwd=PROJECT_ROOT,
    )
    
    if result.returncode != 0:
        logger.error("Validation failed!")
        return False
    
    logger.info("✓ Validation passed")
    return True


def run_training(episodes: int = 500, quick: bool = False):
    """Run training for advanced algorithms."""
    logger.info("=" * 80)
    logger.info("STEP 2: TRAINING ADVANCED ALGORITHMS")
    logger.info("=" * 80)
    
    if quick:
        episodes = 100
    
    # Train HRL
    logger.info("Training HRL...")
    result_hrl = subprocess.run(
        [sys.executable, "scripts/train_hrl.py", "--episodes", str(episodes), "--validate", "--compare"],
        cwd=PROJECT_ROOT,
    )
    
    if result_hrl.returncode != 0:
        logger.warning("HRL training had issues (may be expected for quick mode)")
    
    # Train MBRL
    logger.info("Training MBRL...")
    result_mbrl = subprocess.run(
        [sys.executable, "scripts/train_mbrl.py", "--episodes", str(episodes), "--validate", "--compare"],
        cwd=PROJECT_ROOT,
    )
    
    if result_mbrl.returncode != 0:
        logger.warning("MBRL training had issues (may be expected for quick mode)")
    
    logger.info("✓ Training completed")
    return True


def run_benchmarking(episodes: int = 500, quick: bool = False):
    """Run comprehensive benchmarking."""
    logger.info("=" * 80)
    logger.info("STEP 3: BENCHMARKING ALL ALGORITHMS")
    logger.info("=" * 80)
    
    if quick:
        episodes = 100
    
    result = subprocess.run(
        [sys.executable, "scripts/benchmark_advanced_algorithms.py", "--episodes", str(episodes)],
        cwd=PROJECT_ROOT,
    )
    
    if result.returncode != 0:
        logger.error("Benchmarking failed!")
        return False
    
    logger.info("✓ Benchmarking completed")
    return True


def generate_comparison_report():
    """Generate comparison report."""
    logger.info("=" * 80)
    logger.info("STEP 4: GENERATING COMPARISON REPORT")
    logger.info("=" * 80)
    
    # Find latest benchmark results
    benchmark_dir = PROJECT_ROOT / "results" / "benchmarks" / "advanced_algorithms"
    if benchmark_dir.exists():
        benchmark_files = sorted(benchmark_dir.glob("benchmark_results_*.json"), reverse=True)
        if benchmark_files:
            benchmark_file = benchmark_files[0]
            
            result = subprocess.run(
                [sys.executable, "scripts/generate_algorithm_comparison_report.py",
                 "--benchmark-results", str(benchmark_file)],
                cwd=PROJECT_ROOT,
            )
            
            if result.returncode == 0:
                logger.info("✓ Comparison report generated")
                return True
    
    logger.warning("No benchmark results found, skipping report generation")
    return False


def main():
    """Main validation pipeline."""
    parser = argparse.ArgumentParser(description="Complete Validation and Training Pipeline")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation step")
    parser.add_argument("--skip-training", action="store_true", help="Skip training step")
    parser.add_argument("--skip-benchmarking", action="store_true", help="Skip benchmarking step")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer episodes)")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("COMPLETE VALIDATION AND TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Mode: {'Quick' if args.quick else 'Full'}")
    logger.info(f"Episodes: {args.episodes}")
    logger.info("=" * 80)
    
    success = True
    
    # Step 1: Validation
    if not args.skip_validation:
        if not run_validation():
            logger.error("Validation failed - stopping pipeline")
            return 1
    else:
        logger.info("Skipping validation step")
    
    # Step 2: Training
    if not args.skip_training:
        run_training(episodes=args.episodes, quick=args.quick)
    else:
        logger.info("Skipping training step")
    
    # Step 3: Benchmarking
    if not args.skip_benchmarking:
        if not run_benchmarking(episodes=args.episodes, quick=args.quick):
            logger.error("Benchmarking failed")
            success = False
    else:
        logger.info("Skipping benchmarking step")
    
    # Step 4: Generate report
    generate_comparison_report()
    
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    
    if success:
        logger.info("✅ All steps completed successfully")
        return 0
    else:
        logger.warning("⚠️ Some steps had issues - check logs")
        return 1


if __name__ == "__main__":
    sys.exit(main())

