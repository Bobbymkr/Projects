"""
Comprehensive Testing Framework for YOLOv11n Migration Validation

Provides:
- Automated test suite for model comparison and validation
- Performance benchmarking and regression testing
- Integration tests for model manager and rollback systems
- Load testing and stress testing capabilities
- Edge case validation and robustness testing
- Continuous testing pipeline integration
- Test reporting and analytics
"""

import time
import unittest
import threading
import tempfile
import shutil
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
import cv2
import logging

from .model_manager import YOLOModelManager, ModelMetrics
from .validation import DualModelValidator, ValidationConfig, ValidationStrategy
from .rollback import AutomatedRollbackManager, RollbackConfig, RollbackTrigger
from .performance import PerformanceMonitor, PerformanceMetrics
from .config import YOLOConfig, ModelType, VisionSystemConfig
from .yolo_queue import YOLOQueueEstimator, ROIConfig

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    \"\"\"Result of a test execution.\"\"\"
    test_name: str
    passed: bool
    execution_time_seconds: float
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    artifacts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        \"\"\"Convert to dictionary for serialization.\"\"\"
        return {
            \"test_name\": self.test_name,
            \"passed\": self.passed,
            \"execution_time_seconds\": self.execution_time_seconds,
            \"error_message\": self.error_message,
            \"metrics\": self.metrics,
            \"artifacts\": self.artifacts
        }


@dataclass
class TestSuite:
    \"\"\"Collection of related tests.\"\"\"
    name: str
    description: str
    tests: List[Callable] = field(default_factory=list)
    setup_func: Optional[Callable] = None
    teardown_func: Optional[Callable] = None
    

class TestDataGenerator:
    \"\"\"Generates synthetic test data for validation.\"\"\"
    
    def __init__(self):
        self.test_frames_cache = {}
    
    def generate_test_frame(self, width: int = 640, height: int = 480, 
                          vehicle_count: int = 3, noise_level: float = 0.1) -> np.ndarray:
        \"\"\"Generate synthetic test frame with vehicles.\"\"\"
        # Create base frame
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Add simulated vehicles (rectangles)
        for i in range(vehicle_count):
            x = np.random.randint(50, width - 100)
            y = np.random.randint(50, height - 60)
            w, h = 80 + np.random.randint(-20, 20), 40 + np.random.randint(-10, 10)
            
            # Vehicle color (darker rectangles)
            color = tuple(np.random.randint(50, 150, 3).tolist())
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
            
            # Add some realistic features
            # Windshield
            cv2.rectangle(frame, (x + 10, y + 5), (x + w - 10, y + 15), (200, 200, 255), -1)
        
        # Add noise if specified
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 255, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return frame
    
    def generate_video_sequence(self, frame_count: int = 30, **kwargs) -> List[np.ndarray]:
        \"\"\"Generate sequence of test frames simulating video.\"\"\"
        frames = []
        for i in range(frame_count):
            # Vary vehicle count slightly over time
            vehicle_count = max(1, kwargs.get('vehicle_count', 3) + np.random.randint(-1, 2))
            frame = self.generate_test_frame(vehicle_count=vehicle_count, **kwargs)
            frames.append(frame)
        return frames
    
    def generate_edge_case_frames(self) -> Dict[str, np.ndarray]:
        \"\"\"Generate edge case test frames.\"\"\"
        edge_cases = {}
        
        # Very dark frame
        edge_cases['dark'] = np.full((480, 640, 3), 20, dtype=np.uint8)
        
        # Very bright frame
        edge_cases['bright'] = np.full((480, 640, 3), 240, dtype=np.uint8)
        
        # High noise frame
        edge_cases['noisy'] = self.generate_test_frame(noise_level=0.5)
        
        # Empty frame (no vehicles)
        edge_cases['empty'] = self.generate_test_frame(vehicle_count=0)
        
        # Overcrowded frame
        edge_cases['crowded'] = self.generate_test_frame(vehicle_count=15)
        
        # Blurred frame
        base_frame = self.generate_test_frame()
        edge_cases['blurred'] = cv2.GaussianBlur(base_frame, (15, 15), 0)
        
        return edge_cases


class PerformanceBenchmark:
    \"\"\"Benchmarking utilities for performance testing.\"\"\"
    
    def __init__(self):
        self.benchmark_results = {}
    
    def benchmark_model_inference(self, model_manager: YOLOModelManager, 
                                test_frames: List[np.ndarray], 
                                runs: int = 5) -> Dict[str, Any]:
        \"\"\"Benchmark model inference performance.\"\"\"
        results = {
            \"total_frames\": len(test_frames) * runs,
            \"inference_times\": [],
            \"fps_measurements\": [],
            \"memory_usage\": [],
            \"error_count\": 0
        }
        
        for run in range(runs):
            run_start = time.perf_counter()
            
            for frame in test_frames:
                frame_start = time.perf_counter()
                
                try:
                    detections, metrics = model_manager.predict(frame)
                    frame_time = time.perf_counter() - frame_start
                    
                    results[\"inference_times\"].append(frame_time * 1000)  # Convert to ms
                    results[\"fps_measurements\"].append(1.0 / frame_time if frame_time > 0 else 0)
                    
                    if metrics:
                        results[\"memory_usage\"].append(metrics.memory_usage_mb)
                    
                except Exception as e:
                    logger.error(f\"Inference error: {e}\")
                    results[\"error_count\"] += 1
            
            run_time = time.perf_counter() - run_start
            logger.info(f\"Benchmark run {run + 1}/{runs} completed in {run_time:.2f}s\")
        
        # Calculate statistics
        if results[\"inference_times\"]:
            results[\"avg_inference_time_ms\"] = np.mean(results[\"inference_times\"])
            results[\"min_inference_time_ms\"] = np.min(results[\"inference_times\"])
            results[\"max_inference_time_ms\"] = np.max(results[\"inference_times\"])
            results[\"std_inference_time_ms\"] = np.std(results[\"inference_times\"])
            
            results[\"avg_fps\"] = np.mean(results[\"fps_measurements\"])
            results[\"min_fps\"] = np.min(results[\"fps_measurements\"])
            results[\"max_fps\"] = np.max(results[\"fps_measurements\"])
            
            if results[\"memory_usage\"]:
                results[\"avg_memory_mb\"] = np.mean(results[\"memory_usage\"])
                results[\"max_memory_mb\"] = np.max(results[\"memory_usage\"])
        
        return results
    
    def benchmark_model_comparison(self, primary_manager: YOLOModelManager,
                                 secondary_manager: YOLOModelManager,
                                 test_frames: List[np.ndarray]) -> Dict[str, Any]:
        \"\"\"Benchmark comparison between two models.\"\"\"
        primary_results = self.benchmark_model_inference(primary_manager, test_frames)
        secondary_results = self.benchmark_model_inference(secondary_manager, test_frames)
        
        comparison = {
            \"primary_model\": primary_results,
            \"secondary_model\": secondary_results,
            \"performance_difference\": {
                \"fps_improvement\": (
                    secondary_results.get(\"avg_fps\", 0) - primary_results.get(\"avg_fps\", 0)
                ) / primary_results.get(\"avg_fps\", 1),
                \"inference_time_reduction\": (
                    primary_results.get(\"avg_inference_time_ms\", 0) - 
                    secondary_results.get(\"avg_inference_time_ms\", 0)
                ) / primary_results.get(\"avg_inference_time_ms\", 1),
                \"memory_difference_mb\": (
                    secondary_results.get(\"avg_memory_mb\", 0) - 
                    primary_results.get(\"avg_memory_mb\", 0)
                )
            }
        }
        
        return comparison


class MigrationTestFramework:
    \"\"\"Main testing framework for YOLOv11n migration validation.\"\"\"
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test components
        self.data_generator = TestDataGenerator()
        self.benchmark = PerformanceBenchmark()
        
        # Test results
        self.test_results: List[TestResult] = []
        self.test_suites: List[TestSuite] = []
        
        # Test configuration
        self.test_config = {
            \"frame_count\": 100,
            \"benchmark_runs\": 3,
            \"performance_threshold_fps\": 15.0,
            \"accuracy_threshold\": 0.7,
            \"timeout_seconds\": 300
        }
        
        logger.info(f\"MigrationTestFramework initialized, output: {self.output_dir}\")
    
    def add_test_suite(self, suite: TestSuite):
        \"\"\"Add a test suite to the framework.\"\"\"
        self.test_suites.append(suite)
        logger.info(f\"Added test suite: {suite.name}\")
    
    def run_test(self, test_func: Callable, test_name: str, **kwargs) -> TestResult:
        \"\"\"Run a single test and record results.\"\"\"
        logger.info(f\"Running test: {test_name}\")
        start_time = time.perf_counter()
        
        try:
            result = test_func(**kwargs)
            execution_time = time.perf_counter() - start_time
            
            test_result = TestResult(
                test_name=test_name,
                passed=True,
                execution_time_seconds=execution_time,
                metrics=result if isinstance(result, dict) else None
            )
            
            logger.info(f\"Test {test_name} PASSED in {execution_time:.2f}s\")
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            test_result = TestResult(
                test_name=test_name,
                passed=False,
                execution_time_seconds=execution_time,
                error_message=str(e)
            )
            
            logger.error(f\"Test {test_name} FAILED in {execution_time:.2f}s: {e}\")
        
        self.test_results.append(test_result)
        return test_result
    
    def run_all_suites(self) -> Dict[str, Any]:
        \"\"\"Run all registered test suites.\"\"\"
        logger.info(f\"Running {len(self.test_suites)} test suites\")
        start_time = time.perf_counter()
        
        suite_results = {}
        
        for suite in self.test_suites:
            logger.info(f\"Starting test suite: {suite.name}\")
            suite_start = time.perf_counter()
            
            # Setup
            if suite.setup_func:
                try:
                    suite.setup_func()
                    logger.info(f\"Setup completed for {suite.name}\")
                except Exception as e:
                    logger.error(f\"Setup failed for {suite.name}: {e}\")
                    continue
            
            # Run tests
            suite_test_results = []
            for test_func in suite.tests:
                test_name = f\"{suite.name}.{test_func.__name__}\"
                result = self.run_test(test_func, test_name)
                suite_test_results.append(result)
            
            # Teardown
            if suite.teardown_func:
                try:
                    suite.teardown_func()
                    logger.info(f\"Teardown completed for {suite.name}\")
                except Exception as e:
                    logger.error(f\"Teardown failed for {suite.name}: {e}\")
            
            suite_time = time.perf_counter() - suite_start
            passed_tests = sum(1 for r in suite_test_results if r.passed)
            
            suite_results[suite.name] = {
                \"total_tests\": len(suite_test_results),
                \"passed_tests\": passed_tests,
                \"failed_tests\": len(suite_test_results) - passed_tests,
                \"execution_time_seconds\": suite_time,
                \"success_rate\": passed_tests / len(suite_test_results) if suite_test_results else 0
            }
            
            logger.info(f\"Suite {suite.name} completed: {passed_tests}/{len(suite_test_results)} passed\")
        
        total_time = time.perf_counter() - start_time
        
        # Generate summary
        summary = {
            \"total_suites\": len(self.test_suites),
            \"total_tests\": len(self.test_results),
            \"passed_tests\": sum(1 for r in self.test_results if r.passed),
            \"failed_tests\": sum(1 for r in self.test_results if not r.passed),
            \"total_execution_time_seconds\": total_time,
            \"suite_results\": suite_results,
            \"overall_success_rate\": sum(1 for r in self.test_results if r.passed) / len(self.test_results) if self.test_results else 0
        }
        
        logger.info(f\"All tests completed: {summary['passed_tests']}/{summary['total_tests']} passed\")
        
        return summary
    
    def test_model_loading(self) -> Dict[str, Any]:
        \"\"\"Test model loading and initialization.\"\"\"
        config = YOLOConfig(
            model_type=ModelType.YOLOV11_NANO,
            confidence_threshold=0.5
        )
        
        model_manager = YOLOModelManager(config)
        
        # Test initialization
        assert model_manager.initialize_models(), \"Model initialization failed\"
        
        # Test model availability
        active_model = model_manager.get_active_model()
        assert active_model is not None, \"No active model available\"
        
        return {\"status\": \"success\", \"model_loaded\": True}
    
    def test_model_inference(self) -> Dict[str, Any]:
        \"\"\"Test model inference with synthetic data.\"\"\"
        config = YOLOConfig(model_type=ModelType.YOLOV11_NANO)
        model_manager = YOLOModelManager(config)
        model_manager.initialize_models()
        
        # Generate test frame
        test_frame = self.data_generator.generate_test_frame()
        
        # Test inference
        detections, metrics = model_manager.predict(test_frame)
        
        assert isinstance(detections, list), \"Detections should be a list\"
        assert metrics is not None, \"Metrics should not be None\"
        assert metrics.fps > 0, \"FPS should be positive\"
        
        return {
            \"detections_count\": len(detections),
            \"fps\": metrics.fps,
            \"inference_time_ms\": metrics.inference_time_ms
        }
    
    def test_model_comparison(self) -> Dict[str, Any]:
        \"\"\"Test comparison between YOLOv8n and YOLOv11n.\"\"\"
        # Create configurations
        yolov8_config = YOLOConfig(model_type=ModelType.YOLOV8_NANO)
        yolov11_config = YOLOConfig(model_type=ModelType.YOLOV11_NANO)
        
        # Initialize model managers
        yolov8_manager = YOLOModelManager(yolov8_config)
        yolov11_manager = YOLOModelManager(yolov11_config)
        
        yolov8_manager.initialize_models()
        yolov11_manager.initialize_models()
        
        # Generate test data
        test_frames = self.data_generator.generate_video_sequence(30)
        
        # Run benchmark comparison
        comparison = self.benchmark.benchmark_model_comparison(
            yolov8_manager, yolov11_manager, test_frames
        )
        
        # Validate performance improvement
        fps_improvement = comparison[\"performance_difference\"][\"fps_improvement\"]
        assert fps_improvement >= -0.2, f\"Significant FPS degradation: {fps_improvement:.2%}\"
        
        # Cleanup
        yolov8_manager.cleanup()
        yolov11_manager.cleanup()
        
        return comparison
    
    def test_rollback_mechanism(self) -> Dict[str, Any]:
        \"\"\"Test automated rollback mechanism.\"\"\"
        # Create test configuration
        yolo_config = YOLOConfig(model_type=ModelType.YOLOV11_NANO)
        rollback_config = RollbackConfig(
            fps_degradation_threshold=0.5,
            consecutive_failures_threshold=2
        )
        
        model_manager = YOLOModelManager(yolo_config)
        performance_monitor = PerformanceMonitor(enable_ab_testing=True)
        
        model_manager.initialize_models()
        
        rollback_manager = AutomatedRollbackManager(
            rollback_config, model_manager, performance_monitor
        )
        
        # Test manual rollback
        rollback_success = rollback_manager.manual_rollback(\"Test rollback\")
        assert rollback_success, \"Manual rollback should succeed\"
        
        # Check rollback history
        summary = rollback_manager.get_rollback_summary()
        assert summary[\"total_rollbacks\"] > 0, \"Rollback should be recorded\"
        
        rollback_manager.cleanup()
        model_manager.cleanup()
        
        return {\"rollback_success\": rollback_success, \"rollback_count\": summary[\"total_rollbacks\"]}
    
    def test_dual_model_validation(self) -> Dict[str, Any]:
        \"\"\"Test dual model validation framework.\"\"\"
        # Create validation configuration
        validation_config = ValidationConfig(
            strategy=ValidationStrategy.SHADOW_MODE,
            primary_model_type=ModelType.YOLOV8_NANO,
            secondary_model_type=ModelType.YOLOV11_NANO,
            minimum_sample_size=10
        )
        
        yolo_config = YOLOConfig()
        validator = DualModelValidator(validation_config, yolo_config)
        
        # Initialize models
        assert validator.initialize_models(), \"Model initialization failed\"
        
        # Start validation
        assert validator.start_validation(), \"Validation start failed\"
        
        # Process test frames
        test_frames = self.data_generator.generate_video_sequence(20)
        
        for frame in test_frames:
            detections, metrics = validator.process_frame(frame)
            assert isinstance(detections, list), \"Detections should be a list\"
            assert metrics is not None, \"Validation metrics should not be None\"
        
        # Get validation summary
        summary = validator.get_validation_summary()
        assert summary[\"status\"] == \"active\", \"Validation should be active\"
        assert summary[\"total_comparisons\"] >= 20, \"Should have processed all frames\"
        
        # Stop validation
        final_result = validator.stop_validation()
        assert final_result[\"status\"] == \"stopped\", \"Validation should be stopped\"
        
        validator.cleanup()
        
        return summary
    
    def test_edge_cases(self) -> Dict[str, Any]:
        \"\"\"Test edge cases and robustness.\"\"\"
        config = YOLOConfig(model_type=ModelType.YOLOV11_NANO)
        model_manager = YOLOModelManager(config)
        model_manager.initialize_models()
        
        edge_case_frames = self.data_generator.generate_edge_case_frames()
        results = {}
        
        for case_name, frame in edge_case_frames.items():
            try:
                detections, metrics = model_manager.predict(frame)
                results[case_name] = {
                    \"success\": True,
                    \"detections\": len(detections),
                    \"fps\": metrics.fps if metrics else 0
                }
            except Exception as e:
                results[case_name] = {
                    \"success\": False,
                    \"error\": str(e)
                }
        
        model_manager.cleanup()
        
        # Check that at least 80% of edge cases pass
        success_rate = sum(1 for r in results.values() if r.get(\"success\", False)) / len(results)
        assert success_rate >= 0.8, f\"Edge case success rate too low: {success_rate:.2%}\"
        
        return results
    
    def test_performance_thresholds(self) -> Dict[str, Any]:
        \"\"\"Test that performance meets required thresholds.\"\"\"
        config = YOLOConfig(model_type=ModelType.YOLOV11_NANO)
        model_manager = YOLOModelManager(config)
        model_manager.initialize_models()
        
        # Generate test data
        test_frames = self.data_generator.generate_video_sequence(50)
        
        # Run benchmark
        benchmark_results = self.benchmark.benchmark_model_inference(model_manager, test_frames)
        
        # Check performance thresholds
        avg_fps = benchmark_results.get(\"avg_fps\", 0)
        assert avg_fps >= self.test_config[\"performance_threshold_fps\"], \\n            f\"FPS below threshold: {avg_fps:.1f} < {self.test_config['performance_threshold_fps']}\"
        
        error_rate = benchmark_results.get(\"error_count\", 0) / benchmark_results.get(\"total_frames\", 1)
        assert error_rate <= 0.05, f\"Error rate too high: {error_rate:.2%}\"
        
        model_manager.cleanup()
        
        return benchmark_results
    
    def create_default_test_suites(self):
        \"\"\"Create default test suites for migration validation.\"\"\"
        
        # Core functionality tests
        core_suite = TestSuite(
            name=\"core_functionality\",
            description=\"Core model functionality tests\",
            tests=[
                self.test_model_loading,
                self.test_model_inference,
                self.test_edge_cases
            ]
        )
        
        # Performance tests
        performance_suite = TestSuite(
            name=\"performance\",
            description=\"Performance and benchmarking tests\",
            tests=[
                self.test_performance_thresholds,
                self.test_model_comparison
            ]
        )
        
        # Integration tests
        integration_suite = TestSuite(
            name=\"integration\",
            description=\"Integration and system tests\",
            tests=[
                self.test_rollback_mechanism,
                self.test_dual_model_validation
            ]
        )
        
        self.add_test_suite(core_suite)
        self.add_test_suite(performance_suite)
        self.add_test_suite(integration_suite)
    
    def generate_test_report(self, summary: Dict[str, Any]) -> str:
        \"\"\"Generate comprehensive test report.\"\"\"
        report_path = self.output_dir / \"test_report.json\"
        
        report_data = {
            \"timestamp\": time.time(),
            \"summary\": summary,
            \"test_results\": [r.to_dict() for r in self.test_results],
            \"configuration\": self.test_config
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f\"Test report generated: {report_path}\")
        return str(report_path)
    
    def cleanup(self):
        \"\"\"Clean up test resources.\"\"\"
        # Clean up test output directory if it was temporary
        if str(self.output_dir).startswith(tempfile.gettempdir()):
            shutil.rmtree(self.output_dir, ignore_errors=True)
        
        logger.info(\"Test framework cleanup completed\")


def run_migration_validation_tests(output_dir: Optional[str] = None) -> Dict[str, Any]:
    \"\"\"Convenience function to run complete migration validation test suite.\"\"\"
    framework = MigrationTestFramework(output_dir)
    
    try:
        # Create and run default test suites
        framework.create_default_test_suites()
        
        # Run all tests
        summary = framework.run_all_suites()
        
        # Generate report
        report_path = framework.generate_test_report(summary)
        summary[\"report_path\"] = report_path
        
        return summary
    
    finally:
        framework.cleanup()


if __name__ == \"__main__\":
    # Run tests if executed directly
    logging.basicConfig(level=logging.INFO)
    results = run_migration_validation_tests()
    
    print(f\"\nTest Results Summary:\")
    print(f\"Total Tests: {results['total_tests']}\")
    print(f\"Passed: {results['passed_tests']}\")
    print(f\"Failed: {results['failed_tests']}\")
    print(f\"Success Rate: {results['overall_success_rate']:.1%}\")
    print(f\"Report: {results.get('report_path', 'Not generated')}\")