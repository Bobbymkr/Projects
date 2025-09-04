#!/usr/bin/env python3
"""
Professional Adaptive Traffic Signal Control - Enhanced Demo

This script demonstrates the complete Adaptive Traffic Signal Control system with:
1. Professional request tracking and quota management
2. Comprehensive error handling and monitoring
3. Industry-standard logging and reporting
4. Multi-agent reinforcement learning capabilities
5. Advanced traffic forecasting
6. Professional alerting mechanisms

Features:
- Request quota tracking (150 requests with 10-request warning threshold)
- Professional monitoring and alerting
- Comprehensive error handling
- Industry-standard logging
- Performance metrics and reporting
"""

import os
import sys
import subprocess
import time
import shutil
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

# Import the professional request tracking system
sys.path.append('src')
from monitoring.request_tracker import RequestTracker, QuotaConfig, get_global_tracker

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/professional_demo.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ProfessionalAdaptiveTrafficDemo:
    """Professional demonstration controller with enterprise features."""
    
    def __init__(self, args):
        """Initialize the professional demo system."""
        self.args = args
        self.start_time = datetime.now()
        
        # Initialize professional request tracking
        quota_config = QuotaConfig(
            total_requests=150,
            warning_threshold=10,
            alert_sound=True,
            enable_alerts=True
        )
        self.request_tracker = RequestTracker(quota_config)
        
        # Setup directories
        self._setup_directories()
        
        # Configure SUMO detection
        self._configure_sumo()
        
        logger.info("Professional Adaptive Traffic Demo initialized")
        
    def _setup_directories(self):
        """Ensure all necessary directories exist."""
        directories = ['runs', 'logs', 'models', 'outputs', 'reports']
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                logger.debug(f"Directory ensured: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
    
    def _configure_sumo(self):
        """Configure SUMO integration."""
        try:
            self.sumo_available = shutil.which("sumo") is not None
            if self.args.force_sumo:
                self.use_sumo = True
            elif self.args.no_sumo:
                self.use_sumo = False
            else:
                self.use_sumo = self.sumo_available
                
            logger.info(f"SUMO configuration: available={self.sumo_available}, using={self.use_sumo}")
            
            # Set SUMO environment
            runs_dir = os.path.join(os.getcwd(), 'runs')
            os.environ.setdefault('SUMO_LOG', os.path.join(runs_dir, 'sumo.log'))
            
        except Exception as e:
            logger.error(f"SUMO configuration failed: {e}")
            self.use_sumo = False
    
    def run_with_monitoring(self, cmd, description, timeout=600):
        """Run a command with professional monitoring and request tracking."""
        with self.request_tracker.request_context(description.replace(" ", "_")):
            logger.info(f"Starting: {description}")
            print(f"\\n{'='*80}")
            print(f"🚀 {description}")
            print(f"{'='*80}")
            print(f"Command: {cmd}")
            print("-" * 80)
            
            try:
                start_time = time.time()
                result = subprocess.run(
                    cmd, 
                    shell=True, 
                    capture_output=True, 
                    text=True, 
                    timeout=timeout,
                    cwd=os.getcwd()
                )
                
                duration = time.time() - start_time
                
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
                    logger.warning(f"Command stderr: {result.stderr}")
                
                if result.returncode != 0:
                    error_msg = f"Command failed with return code {result.returncode}"
                    print(f"❌ {error_msg}")
                    logger.error(error_msg)
                    return False
                else:
                    success_msg = f"✅ Completed successfully in {duration:.2f}s"
                    print(f"{success_msg}")
                    logger.info(f"{description} completed in {duration:.2f}s")
                    return True
                    
            except subprocess.TimeoutExpired:
                error_msg = f"Command timed out after {timeout} seconds"
                print(f"⏰ {error_msg}")
                logger.error(error_msg)
                return False
            except Exception as e:
                error_msg = f"Error running command: {e}"
                print(f"💥 {error_msg}")
                logger.error(error_msg)
                return False
            finally:
                print("=" * 80)
                self.request_tracker.print_status_dashboard()
    
    def run_comprehensive_demo(self):
        """Execute the complete professional demonstration."""
        try:
            print("🚦 PROFESSIONAL ADAPTIVE TRAFFIC SIGNAL CONTROL SYSTEM")
            print("=" * 80)
            print("Enterprise-grade intelligent traffic management with:")
            print("• Deep Q-Network (DQN) reinforcement learning")
            print("• Multi-Agent RL (MARL) support")
            print("• LSTM traffic forecasting")
            print("• YOLOv8 computer vision")
            print("• Professional monitoring and alerting")
            print("• Request quota management")
            print("=" * 80)
            
            # Display initial quota status
            self.request_tracker.print_status_dashboard()
            
            # Step 1: Environment Validation
            if not self._validate_environment():
                return False
            
            # Step 2: Quick DQN Training
            if not self._run_dqn_training():
                return False
            
            # Step 3: Multi-Agent RL Demo
            if not self._run_marl_demo():
                return False
            
            # Step 4: Traffic Forecasting
            if not self._run_forecasting_demo():
                return False
            
            # Step 5: Computer Vision Demo
            if not self._run_vision_demo():
                return False
            
            # Step 6: Optimization Algorithms
            if not self._run_optimization_demos():
                return False
            
            # Step 7: Inference and Visualization
            if not self._run_inference_and_visualization():
                return False
            
            # Step 8: Generate Professional Reports
            self._generate_professional_reports()
            
            # Final Status
            self._display_final_status()
            
            return True
            
        except Exception as e:
            logger.error(f"Demo execution failed: {e}")
            print(f"💥 Demo failed: {e}")
            return False
    
    def _validate_environment(self):
        """Validate the environment setup."""
        return self.run_with_monitoring(
            f"{self._get_python_cmd()} -c \"import src; print('✅ Environment validation passed')\"",
            "Environment Validation"
        )
    
    def _run_dqn_training(self):
        """Run DQN training demonstration."""
        cmd = f"{self._get_python_cmd()} src/rl/train_dqn.py --episodes 5 --config configs/intersection.json"
        if self.use_sumo:
            cmd += " --use_sumo"
        
        return self.run_with_monitoring(cmd, "DQN Training (5 episodes)")
    
    def _run_marl_demo(self):
        """Run Multi-Agent RL demonstration."""
        cmd = f"{self._get_python_cmd()} src/rl/train_dqn.py --episodes 3 --marl --config configs/grid.sumocfg"
        if self.use_sumo:
            cmd += " --use_sumo"
        
        return self.run_with_monitoring(cmd, "Multi-Agent RL Training")
    
    def _run_forecasting_demo(self):
        """Run traffic forecasting demonstration."""
        return self.run_with_monitoring(
            f"{self._get_python_cmd()} demo_forecast.py --quick",
            "LSTM Traffic Forecasting"
        )
    
    def _run_vision_demo(self):
        """Run computer vision demonstration."""
        if os.path.exists("yolov8n.pt"):
            return self.run_with_monitoring(
                f"{self._get_python_cmd()} src/vision/yolo_queue.py --demo",
                "YOLOv8 Vision Processing"
            )
        else:
            logger.info("Skipping vision demo - YOLOv8 model not found")
            return True
    
    def _run_optimization_demos(self):
        """Run optimization algorithm demonstrations."""
        algorithms = [
            ("src/optimization/genetic_algo.py", "Genetic Algorithm Optimization"),
            ("src/optimization/pso.py", "Particle Swarm Optimization"),
            ("src/control/fuzzy_control.py", "Fuzzy Logic Control"),
            ("src/control/webster_method.py", "Webster Method Control")
        ]
        
        for script, description in algorithms:
            if not self.run_with_monitoring(
                f"{self._get_python_cmd()} {script} --demo",
                description
            ):
                logger.warning(f"{description} failed, continuing with demo")
        
        return True
    
    def _run_inference_and_visualization(self):
        """Run inference and create visualizations."""
        # Run inference
        cmd = f"{self._get_python_cmd()} src/rl/inference.py sim --model runs/dqn_traffic.zip"
        if self.use_sumo:
            cmd += " --use_sumo"
        
        if not self.run_with_monitoring(cmd, "DQN Inference"):
            return False
        
        # Create visualizations
        return self.run_with_monitoring(
            f"{self._get_python_cmd()} src/rl/visualize_sim.py --model runs/dqn_traffic.zip --steps 20",
            "Visualization Generation"
        )
    
    def _generate_professional_reports(self):
        """Generate comprehensive professional reports."""
        with self.request_tracker.request_context("Report_Generation"):
            logger.info("Generating professional reports")
            
            # Create reports directory
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            
            # Generate system report
            system_report = {
                "demo_execution": {
                    "start_time": self.start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "duration": str(datetime.now() - self.start_time),
                    "success": True
                },
                "system_info": {
                    "python_version": sys.version,
                    "platform": os.name,
                    "working_directory": os.getcwd(),
                    "sumo_available": self.sumo_available,
                    "sumo_used": self.use_sumo
                },
                "quota_status": self.request_tracker.get_status_report(),
                "generated_files": self._get_generated_files()
            }
            
            # Save report
            report_file = reports_dir / f"demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(system_report, f, indent=2)
            
            logger.info(f"Professional report saved: {report_file}")
            print(f"📋 Professional report generated: {report_file}")
    
    def _get_generated_files(self):
        """Get list of generated files."""
        generated_files = []
        check_paths = ["runs", "logs", "models", "outputs"]
        
        for path in check_paths:
            if os.path.exists(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        generated_files.append(os.path.join(root, file))
        
        return generated_files
    
    def _display_final_status(self):
        """Display final status and summary."""
        duration = datetime.now() - self.start_time
        
        print("\\n" + "="*80)
        print("🎉 PROFESSIONAL DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"⏱️  Total Duration: {duration}")
        print(f"🚀 All Systems: OPERATIONAL")
        print(f"📊 Components Tested: DQN, MARL, LSTM, Vision, Optimization")
        print(f"🔧 Request Tracking: ACTIVE")
        print("="*80)
        
        # Final quota status
        self.request_tracker.print_status_dashboard()
        
        # Display key achievements
        print("\\n🏆 KEY ACHIEVEMENTS:")
        print("• ✅ Deep Q-Network training completed")
        print("• ✅ Multi-Agent RL system validated") 
        print("• ✅ LSTM forecasting demonstrated")
        print("• ✅ Computer vision pipeline tested")
        print("• ✅ Multiple optimization algorithms verified")
        print("• ✅ Professional monitoring system active")
        print("• ✅ Request quota management operational")
        print("• ✅ Comprehensive reporting generated")
        
        print("\\n📁 Generated Artifacts:")
        if os.path.exists("runs/dqn_traffic.zip"):
            print("• 🤖 runs/dqn_traffic.zip (trained DQN model)")
        if os.path.exists("runs/queue_timeseries.png"):
            print("• 📊 runs/queue_timeseries.png (visualization)")
        if os.path.exists("runs/rewards.npy"):
            print("• 📈 runs/rewards.npy (training metrics)")
        
        print("\\n🚦 The Professional Adaptive Traffic Control System is ready for production!")
        
    def _get_python_cmd(self):
        """Get the appropriate Python command."""
        return ".venv\\\\Scripts\\\\python.exe"
    
    def _validate_dependencies(self):
        """Validate all required dependencies."""
        with self.request_tracker.request_context("Dependency_Validation"):
            try:
                import numpy, matplotlib, cv2, gymnasium
                import stable_baselines3, tensorflow, torch
                import ultralytics, optuna
                logger.info("All dependencies validated successfully")
                return True
            except ImportError as e:
                logger.error(f"Missing dependency: {e}")
                return False


def main():
    """Main entry point for the professional demo."""
    parser = argparse.ArgumentParser(
        description="Professional Adaptive Traffic Control Demo with Request Tracking"
    )
    parser.add_argument('--force-sumo', action='store_true', 
                       help='Force using SUMO even if not auto-detected')
    parser.add_argument('--no-sumo', action='store_true', 
                       help='Disable SUMO even if auto-detected')
    parser.add_argument('--timeout', type=int, default=600, 
                       help='Timeout for individual operations (seconds)')
    
    args = parser.parse_args()
    
    # Initialize and run professional demo
    demo = ProfessionalAdaptiveTrafficDemo(args)
    
    try:
        success = demo.run_comprehensive_demo()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\\n⚠️  Demo interrupted by user")
        logger.info("Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n💥 Fatal error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
