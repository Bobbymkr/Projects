#!/usr/bin/env python3
"""
YOLOv11n Migration Execution Script - Week 1: Compatibility Validation and Shadow Mode Setup

Implements the first week of the YOLOv8n to YOLOv11n migration plan.
"""

import sys
import os
import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import vision components
from src.vision import (
    YOLOConfig, ModelType, YOLOModelManager, DualModelValidator, 
    ValidationConfig, ValidationStrategy, AutomatedRollbackManager,
    RollbackConfig, PerformanceMonitor, MigrationTestFramework,
    NISTComplianceManager
)

logger = logging.getLogger(__name__)


class MigrationWeek1Executor:
    """Executes Week 1 of the YOLOv8n to YOLOv11n migration plan."""
    
    def __init__(self, output_dir: str = "./migration_week1_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Migration components
        self.model_manager = None
        self.validator = None
        self.rollback_manager = None
        self.performance_monitor = None
        self.compliance_manager = None
        self.test_framework = None
        
        # Migration state
        self.migration_state = {
            "phase": "week1",
            "status": "initializing",
            "start_time": time.time(),
            "components_initialized": [],
            "validation_results": {},
            "performance_baselines": {}
        }
        
        logger.info("Migration Week 1 Executor initialized")
    
    def execute_week1_migration(self) -> Dict[str, Any]:
        """Execute complete Week 1 migration process."""
        logger.info("=== Starting YOLOv11n Migration Week 1 ===")
        
        try:
            self._setup_components()
            self._validate_model_compatibility()
            self._establish_performance_baselines()
            self._deploy_shadow_mode()
            self._run_comprehensive_tests()
            self._generate_week1_report()
            
            self.migration_state["status"] = "completed"
            logger.info("=== Week 1 Migration Completed Successfully ===")
            
            return self.migration_state
            
        except Exception as e:
            logger.error(f"Week 1 migration failed: {e}")
            self.migration_state["status"] = "failed"
            self.migration_state["error"] = str(e)
            raise
    
    def _setup_components(self):
        """Initialize all migration components."""
        logger.info("Setting up migration components...")
        
        # YOLO configuration
        yolo_config = YOLOConfig(
            model_type=ModelType.YOLOV11_NANO,
            enable_shadow_mode=True,
            shadow_model_type=ModelType.YOLOV8_NANO,
            enable_model_fallback=True,
            enable_auto_rollback=True
        )
        
        # Initialize model manager
        self.model_manager = YOLOModelManager(yolo_config)
        if not self.model_manager.initialize_models():
            raise RuntimeError("Failed to initialize model manager")
        self.migration_state["components_initialized"].append("model_manager")
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor(enable_ab_testing=True)
        self.migration_state["components_initialized"].append("performance_monitor")
        
        # Initialize rollback manager
        rollback_config = RollbackConfig(enable_auto_rollback=True)
        self.rollback_manager = AutomatedRollbackManager(
            rollback_config, self.model_manager, self.performance_monitor
        )
        self.rollback_manager.start_monitoring()
        self.migration_state["components_initialized"].append("rollback_manager")
        
        # Initialize validator
        validation_config = ValidationConfig(
            strategy=ValidationStrategy.SHADOW_MODE,
            primary_model_type=ModelType.YOLOV8_NANO,
            secondary_model_type=ModelType.YOLOV11_NANO
        )
        self.validator = DualModelValidator(validation_config, yolo_config)
        if not self.validator.initialize_models():
            raise RuntimeError("Failed to initialize validator")
        self.migration_state["components_initialized"].append("validator")
        
        # Initialize compliance manager
        self.compliance_manager = NISTComplianceManager()
        self.migration_state["components_initialized"].append("compliance_manager")
        
        # Initialize test framework
        self.test_framework = MigrationTestFramework(str(self.output_dir / "tests"))
        self.test_framework.create_default_test_suites()
        self.migration_state["components_initialized"].append("test_framework")
        
        logger.info("All components initialized successfully")
    
    def _validate_model_compatibility(self):
        """Validate YOLOv11n model compatibility."""
        logger.info("Validating model compatibility...")
        
        try:
            from ultralytics import YOLO
            import numpy as np
            
            # Test model loading
            yolo11_model = YOLO("yolo11n.pt")
            yolo8_model = YOLO("yolov8n.pt")
            
            # Test API compatibility
            test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            results11 = yolo11_model(test_frame, verbose=False)
            results8 = yolo8_model(test_frame, verbose=False)
            
            # Verify result structure
            assert hasattr(results11[0], 'boxes'), "YOLOv11n missing 'boxes' attribute"
            assert hasattr(results8[0], 'boxes'), "YOLOv8n missing 'boxes' attribute"
            
            self.migration_state["validation_results"]["model_compatibility"] = {
                "yolo11n_loading": True,
                "yolo8n_loading": True,
                "api_compatibility": True
            }
            
            logger.info("Model compatibility validation passed")
            
        except Exception as e:
            logger.error(f"Model compatibility validation failed: {e}")
            raise
    
    def _establish_performance_baselines(self):
        """Establish performance baselines."""
        logger.info("Establishing performance baselines...")
        
        import numpy as np
        
        # Generate test frames
        test_frames = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(50)]
        
        # Benchmark YOLOv8n
        yolo8_config = YOLOConfig(model_type=ModelType.YOLOV8_NANO)
        yolo8_manager = YOLOModelManager(yolo8_config)
        yolo8_manager.initialize_models()
        
        yolo8_times = []
        for frame in test_frames:
            start_time = time.perf_counter()
            detections, metrics = yolo8_manager.predict(frame)
            yolo8_times.append(time.perf_counter() - start_time)
        
        # Benchmark YOLOv11n
        yolo11_times = []
        for frame in test_frames:
            start_time = time.perf_counter()
            detections, metrics = self.model_manager.predict(frame)
            yolo11_times.append(time.perf_counter() - start_time)
        
        # Calculate baselines
        yolo8_fps = 1.0 / np.mean(yolo8_times)
        yolo11_fps = 1.0 / np.mean(yolo11_times)
        improvement = (yolo11_fps - yolo8_fps) / yolo8_fps * 100
        
        self.migration_state["performance_baselines"] = {
            "yolov8n_fps": yolo8_fps,
            "yolov11n_fps": yolo11_fps,
            "fps_improvement_percent": improvement
        }
        
        logger.info(f"Performance improvement: {improvement:.1f}%")
        yolo8_manager.cleanup()
    
    def _deploy_shadow_mode(self):
        """Deploy shadow mode for A/B testing."""
        logger.info("Deploying shadow mode...")
        
        if not self.validator.start_validation():
            raise RuntimeError("Failed to start shadow mode validation")
        
        # Run shadow validation with test frames
        import numpy as np
        import cv2
        
        test_frames = []
        for i in range(100):
            frame = np.random.randint(50, 200, (640, 640, 3), dtype=np.uint8)
            # Add vehicle-like rectangles
            for j in range(np.random.randint(1, 4)):
                x, y = np.random.randint(0, 500, 2)
                w, h = 80, 40
                cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), -1)
            test_frames.append(frame)
        
        # Process through validator
        for frame in test_frames:
            self.validator.process_frame(frame)
        
        validation_summary = self.validator.get_validation_summary()
        self.migration_state["validation_results"]["shadow_mode"] = validation_summary
        
        logger.info("Shadow mode validation completed")
    
    def _run_comprehensive_tests(self):
        """Run comprehensive test suite."""
        logger.info("Running comprehensive test suite...")
        
        test_results = self.test_framework.run_all_suites()
        report_path = self.test_framework.generate_test_report(test_results)
        
        self.migration_state["validation_results"]["tests"] = {
            "results": test_results,
            "report_path": report_path
        }
        
        logger.info(f"Tests completed: {test_results['overall_success_rate']:.1%} success rate")
    
    def _generate_week1_report(self):
        """Generate Week 1 migration report."""
        logger.info("Generating Week 1 report...")
        
        # Stop validation
        final_validation = self.validator.stop_validation()
        
        # Create report
        report_data = {
            "phase": "week1",
            "status": self.migration_state["status"],
            "execution_time": time.time() - self.migration_state["start_time"],
            "components": self.migration_state["components_initialized"],
            "performance": self.migration_state["performance_baselines"],
            "validation": self.migration_state["validation_results"],
            "final_validation": final_validation
        }
        
        # Save report
        report_path = self.output_dir / "week1_report.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Week 1 report generated: {report_path}")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.rollback_manager:
            self.rollback_manager.cleanup()
        if self.validator:
            self.validator.cleanup()
        if self.model_manager:
            self.model_manager.cleanup()
        if self.test_framework:
            self.test_framework.cleanup()


def main():
    """Main execution function."""
    logging.basicConfig(level=logging.INFO)
    
    executor = MigrationWeek1Executor()
    
    try:
        result = executor.execute_week1_migration()
        
        print(f"\n✅ Week 1 Migration Status: {result['status']}")
        print(f"Performance Improvement: {result['performance_baselines']['fps_improvement_percent']:.1f}%")
        print(f"Components Initialized: {len(result['components_initialized'])}")
        
        return 0 if result['status'] == 'completed' else 1
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        return 1
    finally:
        executor.cleanup()


if __name__ == "__main__":
    exit(main())