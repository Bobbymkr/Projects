"""
Dual-Model Validation Architecture for YOLOv8n to YOLOv11n Migration

Provides comprehensive A/B testing framework including:
- Side-by-side model comparison and validation
- Statistical analysis of performance differences
- Automated decision making for model deployment
- Real-time monitoring and alerting systems
- Canary deployment support with traffic splitting
- Rollback automation based on performance thresholds
"""

import time
import threading
import logging
import statistics
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from pathlib import Path
import json
import numpy as np
from enum import Enum

from .model_manager import YOLOModelManager, ModelMetrics, ModelState
from .config import YOLOConfig, ModelType

logger = logging.getLogger(__name__)


class ValidationStrategy(Enum):
    """Validation strategies for model comparison."""
    SHADOW_MODE = "shadow_mode"  # Run both models, use primary results
    CANARY_DEPLOYMENT = "canary_deployment"  # Split traffic between models
    BLUE_GREEN_DEPLOYMENT = "blue_green"  # Switch between environments
    ROLLING_DEPLOYMENT = "rolling"  # Gradual replacement


class DecisionCriteria(Enum):
    """Criteria for automated decision making."""
    PERFORMANCE_THRESHOLD = "performance"
    ACCURACY_THRESHOLD = "accuracy"
    ERROR_RATE_THRESHOLD = "error_rate"
    STATISTICAL_SIGNIFICANCE = "statistical"
    MANUAL_APPROVAL = "manual"


@dataclass
class ValidationMetrics:
    """Comprehensive metrics for model validation."""
    timestamp: float
    model_a_fps: float
    model_b_fps: float
    model_a_accuracy: float
    model_b_accuracy: float
    model_a_error_rate: float
    model_b_error_rate: float
    detection_correlation: float
    statistical_significance: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    
    def calculate_performance_difference(self) -> float:
        """Calculate relative performance difference (Model B vs Model A)."""
        if self.model_a_fps == 0:
            return 0.0
        return (self.model_b_fps - self.model_a_fps) / self.model_a_fps
    
    def calculate_accuracy_difference(self) -> float:
        """Calculate accuracy difference (Model B vs Model A)."""
        return self.model_b_accuracy - self.model_a_accuracy
    
    def is_statistically_significant(self, p_value_threshold: float = 0.05) -> bool:
        """Check if the difference is statistically significant."""
        return self.statistical_significance < p_value_threshold


@dataclass
class ValidationConfig:
    """Configuration for dual-model validation."""
    strategy: ValidationStrategy = ValidationStrategy.SHADOW_MODE
    primary_model_type: ModelType = ModelType.YOLOV8_NANO
    secondary_model_type: ModelType = ModelType.YOLOV11_NANO
    
    # Validation parameters
    minimum_sample_size: int = 100
    validation_duration_seconds: int = 300  # 5 minutes
    statistical_confidence_level: float = 0.95
    
    # Decision thresholds
    min_performance_improvement: float = 0.15  # 15% improvement required
    max_acceptable_accuracy_loss: float = 0.05  # 5% accuracy loss acceptable
    max_acceptable_error_rate: float = 0.02  # 2% error rate threshold
    
    # Traffic splitting (for canary deployment)
    canary_traffic_percentage: float = 0.1  # 10% initial traffic
    canary_increment_percentage: float = 0.1  # 10% increments
    canary_increment_interval_seconds: int = 600  # 10 minutes
    
    # Rollback triggers
    enable_auto_rollback: bool = True
    rollback_performance_threshold: float = 0.8  # Rollback if performance drops below 80%
    rollback_error_threshold: float = 0.05  # Rollback if error rate exceeds 5%
    rollback_consecutive_failures: int = 3  # Rollback after 3 consecutive failures
    
    # Monitoring and alerting
    enable_real_time_monitoring: bool = True
    monitoring_interval_seconds: int = 30
    alert_on_significant_difference: bool = True
    alert_threshold_performance: float = 0.20  # Alert on >20% performance difference


class StatisticalAnalyzer:
    """Performs statistical analysis on model comparison data."""
    
    def __init__(self):
        self.performance_samples_a: List[float] = []
        self.performance_samples_b: List[float] = []
        self.accuracy_samples_a: List[float] = []
        self.accuracy_samples_b: List[float] = []
    
    def add_samples(self, metrics_a: ModelMetrics, metrics_b: ModelMetrics):
        """Add sample data for statistical analysis."""
        self.performance_samples_a.append(metrics_a.fps)
        self.performance_samples_b.append(metrics_b.fps)
        self.accuracy_samples_a.append(metrics_a.accuracy_score)
        self.accuracy_samples_b.append(metrics_b.accuracy_score)
    
    def calculate_statistical_significance(self) -> Tuple[float, Tuple[float, float]]:
        """Calculate statistical significance using t-test."""
        if len(self.performance_samples_a) < 30 or len(self.performance_samples_b) < 30:
            return 1.0, (0.0, 0.0)  # Not enough samples
        
        # Simplified statistical test (in production, use scipy.stats)
        mean_a = statistics.mean(self.performance_samples_a)
        mean_b = statistics.mean(self.performance_samples_b)
        std_a = statistics.stdev(self.performance_samples_a) if len(self.performance_samples_a) > 1 else 0
        std_b = statistics.stdev(self.performance_samples_b) if len(self.performance_samples_b) > 1 else 0
        
        # Simplified p-value calculation (placeholder)
        difference = abs(mean_b - mean_a)
        pooled_std = (std_a + std_b) / 2
        
        if pooled_std == 0:
            p_value = 0.0 if difference > 0 else 1.0
        else:
            # Simplified calculation - in production use proper t-test
            p_value = max(0.0, 1.0 - (difference / pooled_std) * 0.1)
        
        # Confidence interval (simplified)
        confidence_interval = (mean_b - pooled_std, mean_b + pooled_std)
        
        return p_value, confidence_interval
    
    def reset(self):
        """Reset accumulated samples."""
        self.performance_samples_a.clear()
        self.performance_samples_b.clear()
        self.accuracy_samples_a.clear()
        self.accuracy_samples_b.clear()


class AlertManager:
    """Manages alerts and notifications for validation process."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.alert_callbacks: List[Callable] = []
        self.alert_history: List[Dict[str, Any]] = []
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def trigger_alert(self, alert_type: str, message: str, severity: str = "warning", data: Optional[Dict] = None):
        """Trigger an alert with specified details."""
        alert = {
            "timestamp": time.time(),
            "type": alert_type,
            "message": message,
            "severity": severity,
            "data": data or {}
        }
        
        self.alert_history.append(alert)
        logger.log(
            logging.ERROR if severity == "critical" else logging.WARNING,
            f"ALERT [{alert_type}]: {message}"
        )
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def check_validation_alerts(self, metrics: ValidationMetrics):
        """Check for alert conditions based on validation metrics."""
        if not self.config.alert_on_significant_difference:
            return
        
        # Performance difference alert
        perf_diff = abs(metrics.calculate_performance_difference())
        if perf_diff > self.config.alert_threshold_performance:
            self.trigger_alert(
                "performance_difference",
                f"Significant performance difference detected: {perf_diff:.2%}",
                "warning",
                {"performance_difference": perf_diff, "metrics": metrics}
            )
        
        # Error rate alert
        if metrics.model_b_error_rate > self.config.rollback_error_threshold:
            self.trigger_alert(
                "high_error_rate",
                f"High error rate in secondary model: {metrics.model_b_error_rate:.2%}",
                "critical",
                {"error_rate": metrics.model_b_error_rate}
            )


class DualModelValidator:
    """
    Main class for dual-model validation and A/B testing.
    
    Supports multiple validation strategies:
    - Shadow mode: Run both models, use primary results, compare performance
    - Canary deployment: Gradually shift traffic from primary to secondary
    - Blue-green deployment: Complete environment switching
    - Rolling deployment: Gradual replacement with monitoring
    """
    
    def __init__(self, config: ValidationConfig, yolo_config: YOLOConfig):
        self.config = config
        self.yolo_config = yolo_config
        
        # Model managers
        self.primary_manager: Optional[YOLOModelManager] = None
        self.secondary_manager: Optional[YOLOModelManager] = None
        
        # Analysis components
        self.statistical_analyzer = StatisticalAnalyzer()
        self.alert_manager = AlertManager(config)
        
        # State tracking
        self.validation_active = False
        self.validation_start_time = 0.0
        self.current_traffic_split = 0.0  # Percentage going to secondary model
        
        # Metrics and history
        self.validation_metrics_history: List[ValidationMetrics] = []
        self.comparison_results: List[Dict[str, Any]] = []
        self.consecutive_failures = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info(f\"DualModelValidator initialized with strategy: {config.strategy.value}\")
    
    def initialize_models(self) -> bool:
        \"\"\"Initialize both primary and secondary models.\"\"\"
        try:
            # Primary model configuration
            primary_config = self.yolo_config.copy() if hasattr(self.yolo_config, 'copy') else self.yolo_config
            primary_config.model_type = self.config.primary_model_type
            primary_config.enable_shadow_mode = False
            
            # Secondary model configuration  
            secondary_config = self.yolo_config.copy() if hasattr(self.yolo_config, 'copy') else self.yolo_config
            secondary_config.model_type = self.config.secondary_model_type
            secondary_config.enable_shadow_mode = False
            
            # Initialize model managers
            self.primary_manager = YOLOModelManager(primary_config)
            self.secondary_manager = YOLOModelManager(secondary_config)
            
            # Load models
            primary_success = self.primary_manager.initialize_models()
            secondary_success = self.secondary_manager.initialize_models()
            
            if not primary_success:
                logger.error(\"Failed to initialize primary model\")
                return False
            
            if not secondary_success:
                logger.error(\"Failed to initialize secondary model\")
                return False
            
            logger.info(\"Both models initialized successfully\")
            return True
            
        except Exception as e:
            logger.error(f\"Model initialization failed: {e}\")
            return False
    
    def start_validation(self) -> bool:
        \"\"\"Start the validation process.\"\"\"
        if not self.primary_manager or not self.secondary_manager:
            logger.error(\"Models not initialized\")
            return False
        
        with self.lock:
            if self.validation_active:
                logger.warning(\"Validation already active\")
                return False
            
            self.validation_active = True
            self.validation_start_time = time.time()
            self.consecutive_failures = 0
            
            # Initialize traffic split based on strategy
            if self.config.strategy == ValidationStrategy.SHADOW_MODE:
                self.current_traffic_split = 0.0  # All traffic to primary
            elif self.config.strategy == ValidationStrategy.CANARY_DEPLOYMENT:
                self.current_traffic_split = self.config.canary_traffic_percentage
            else:
                self.current_traffic_split = 0.0
        
        logger.info(f\"Validation started with strategy: {self.config.strategy.value}\")
        logger.info(f\"Initial traffic split: {self.current_traffic_split:.1%} to secondary model\")
        
        return True
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict], ValidationMetrics]:
        \"\"\"Process frame through validation pipeline.\"\"\"
        if not self.validation_active:
            # Fallback to primary model only
            if self.primary_manager:
                detections, metrics = self.primary_manager.predict(frame)
                return detections, self._create_validation_metrics(metrics, metrics)
            return [], ValidationMetrics(
                timestamp=time.time(), model_a_fps=0, model_b_fps=0,
                model_a_accuracy=0, model_b_accuracy=0, model_a_error_rate=1, model_b_error_rate=1,
                detection_correlation=0, statistical_significance=1, confidence_interval=(0, 0), sample_size=0
            )
        
        # Run predictions on both models
        primary_detections, primary_metrics = self.primary_manager.predict(frame)
        secondary_detections, secondary_metrics = self.secondary_manager.predict(frame)
        
        # Determine which results to use based on strategy
        if self.config.strategy == ValidationStrategy.SHADOW_MODE:
            # Always use primary results in shadow mode
            active_detections = primary_detections
        elif self.config.strategy == ValidationStrategy.CANARY_DEPLOYMENT:
            # Use traffic split to determine which results to use
            import random
            if random.random() < self.current_traffic_split:
                active_detections = secondary_detections
            else:
                active_detections = primary_detections
        else:
            # Default to primary
            active_detections = primary_detections
        
        # Create validation metrics
        validation_metrics = self._create_validation_metrics(primary_metrics, secondary_metrics)
        
        # Update statistical analysis
        self.statistical_analyzer.add_samples(primary_metrics, secondary_metrics)
        
        # Store comparison results
        comparison = {
            \"timestamp\": time.time(),
            \"primary_detections\": len(primary_detections),
            \"secondary_detections\": len(secondary_detections),
            \"primary_fps\": primary_metrics.fps,
            \"secondary_fps\": secondary_metrics.fps,
            \"detection_correlation\": self._calculate_detection_correlation(primary_detections, secondary_detections),
            \"traffic_split\": self.current_traffic_split
        }
        self.comparison_results.append(comparison)
        
        # Keep history manageable
        if len(self.comparison_results) > 1000:
            self.comparison_results = self.comparison_results[-500:]
        
        # Check for alerts
        self.alert_manager.check_validation_alerts(validation_metrics)
        
        # Check for rollback conditions
        if self.config.enable_auto_rollback:
            self._check_rollback_conditions(validation_metrics)
        
        # Update canary deployment if applicable
        if self.config.strategy == ValidationStrategy.CANARY_DEPLOYMENT:
            self._update_canary_traffic()
        
        return active_detections, validation_metrics
    
    def _create_validation_metrics(self, primary_metrics: ModelMetrics, secondary_metrics: ModelMetrics) -> ValidationMetrics:
        \"\"\"Create validation metrics from model metrics.\"\"\"
        # Calculate statistical significance
        p_value, confidence_interval = self.statistical_analyzer.calculate_statistical_significance()
        
        return ValidationMetrics(
            timestamp=time.time(),
            model_a_fps=primary_metrics.fps,
            model_b_fps=secondary_metrics.fps,
            model_a_accuracy=primary_metrics.accuracy_score,
            model_b_accuracy=secondary_metrics.accuracy_score,
            model_a_error_rate=primary_metrics.error_rate,
            model_b_error_rate=secondary_metrics.error_rate,
            detection_correlation=0.0,  # Calculated separately
            statistical_significance=p_value,
            confidence_interval=confidence_interval,
            sample_size=len(self.statistical_analyzer.performance_samples_a)
        )
    
    def _calculate_detection_correlation(self, detections_a: List[Dict], detections_b: List[Dict]) -> float:
        \"\"\"Calculate correlation between detection results.\"\"\"
        if not detections_a and not detections_b:
            return 1.0  # Perfect correlation when both detect nothing
        
        if not detections_a or not detections_b:
            return 0.0  # No correlation when one detects nothing
        
        # Simple correlation based on detection count similarity
        count_a = len(detections_a)
        count_b = len(detections_b)
        max_count = max(count_a, count_b)
        
        if max_count == 0:
            return 1.0
        
        return 1.0 - abs(count_a - count_b) / max_count
    
    def _check_rollback_conditions(self, metrics: ValidationMetrics):
        \"\"\"Check if rollback should be triggered.\"\"\"
        # Performance degradation check
        if metrics.model_b_fps < metrics.model_a_fps * self.config.rollback_performance_threshold:
            self.consecutive_failures += 1
            logger.warning(f\"Performance degradation detected: {self.consecutive_failures} consecutive failures\")
        
        # Error rate check
        elif metrics.model_b_error_rate > self.config.rollback_error_threshold:
            self.consecutive_failures += 1
            logger.warning(f\"High error rate detected: {self.consecutive_failures} consecutive failures\")
        
        else:
            self.consecutive_failures = 0  # Reset on success
        
        # Trigger rollback if threshold reached
        if self.consecutive_failures >= self.config.rollback_consecutive_failures:
            self._trigger_rollback(\"Consecutive failures threshold reached\")
    
    def _trigger_rollback(self, reason: str):
        \"\"\"Trigger rollback to primary model.\"\"\"
        logger.error(f\"ROLLBACK TRIGGERED: {reason}\")
        
        with self.lock:
            self.current_traffic_split = 0.0  # Route all traffic to primary
            self.consecutive_failures = 0
        
        self.alert_manager.trigger_alert(
            \"rollback_triggered\",
            f\"Automatic rollback triggered: {reason}\",
            \"critical\",
            {\"reason\": reason, \"timestamp\": time.time()}
        )
    
    def _update_canary_traffic(self):
        \"\"\"Update canary deployment traffic split.\"\"\"
        if self.config.strategy != ValidationStrategy.CANARY_DEPLOYMENT:
            return
        
        elapsed_time = time.time() - self.validation_start_time
        intervals_passed = int(elapsed_time // self.config.canary_increment_interval_seconds)
        
        # Calculate target traffic split
        target_split = min(
            self.config.canary_traffic_percentage + 
            (intervals_passed * self.config.canary_increment_percentage),
            1.0
        )
        
        if target_split > self.current_traffic_split:
            with self.lock:
                old_split = self.current_traffic_split
                self.current_traffic_split = target_split
            
            logger.info(f\"Canary traffic increased: {old_split:.1%} -> {target_split:.1%}\")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        \"\"\"Get comprehensive validation summary.\"\"\"
        if not self.validation_active:
            return {\"status\": \"inactive\"}
        
        elapsed_time = time.time() - self.validation_start_time
        recent_comparisons = self.comparison_results[-20:] if self.comparison_results else []
        
        summary = {
            \"status\": \"active\",
            \"strategy\": self.config.strategy.value,
            \"elapsed_time_seconds\": elapsed_time,
            \"current_traffic_split\": self.current_traffic_split,
            \"total_comparisons\": len(self.comparison_results),
            \"consecutive_failures\": self.consecutive_failures,
            \"primary_model\": self.config.primary_model_type.value,
            \"secondary_model\": self.config.secondary_model_type.value
        }
        
        # Add recent performance statistics
        if recent_comparisons:
            recent_primary_fps = [c[\"primary_fps\"] for c in recent_comparisons]
            recent_secondary_fps = [c[\"secondary_fps\"] for c in recent_comparisons]
            
            summary.update({
                \"recent_performance\": {
                    \"primary_avg_fps\": statistics.mean(recent_primary_fps),
                    \"secondary_avg_fps\": statistics.mean(recent_secondary_fps),
                    \"avg_detection_correlation\": statistics.mean([c[\"detection_correlation\"] for c in recent_comparisons]),
                    \"performance_improvement\": (statistics.mean(recent_secondary_fps) - statistics.mean(recent_primary_fps)) / statistics.mean(recent_primary_fps) if recent_primary_fps else 0
                }
            })
        
        # Add statistical analysis
        p_value, confidence_interval = self.statistical_analyzer.calculate_statistical_significance()
        summary[\"statistical_analysis\"] = {
            \"p_value\": p_value,
            \"confidence_interval\": confidence_interval,
            \"sample_size\": len(self.statistical_analyzer.performance_samples_a),
            \"is_significant\": p_value < 0.05
        }
        
        return summary
    
    def make_deployment_decision(self) -> Dict[str, Any]:
        \"\"\"Make automated deployment decision based on validation results.\"\"\"
        if not self.validation_active:
            return {\"decision\": \"no_decision\", \"reason\": \"Validation not active\"}
        
        summary = self.get_validation_summary()
        
        # Check minimum sample size
        if summary.get(\"total_comparisons\", 0) < self.config.minimum_sample_size:
            return {
                \"decision\": \"continue_validation\",
                \"reason\": f\"Insufficient samples: {summary.get('total_comparisons', 0)} < {self.config.minimum_sample_size}\"
            }
        
        # Check if validation duration met
        if summary.get(\"elapsed_time_seconds\", 0) < self.config.validation_duration_seconds:
            return {
                \"decision\": \"continue_validation\",
                \"reason\": f\"Validation duration not met: {summary.get('elapsed_time_seconds', 0)} < {self.config.validation_duration_seconds}\"
            }
        
        # Analyze performance improvement
        recent_perf = summary.get(\"recent_performance\", {})
        performance_improvement = recent_perf.get(\"performance_improvement\", 0)
        
        # Check statistical significance
        stats = summary.get(\"statistical_analysis\", {})
        is_statistically_significant = stats.get(\"is_significant\", False)
        
        # Decision logic
        if performance_improvement >= self.config.min_performance_improvement and is_statistically_significant:
            return {
                \"decision\": \"deploy_secondary\",
                \"reason\": f\"Performance improvement: {performance_improvement:.2%}, statistically significant\",
                \"confidence\": \"high\",
                \"metrics\": recent_perf
            }
        elif performance_improvement > 0 and performance_improvement < self.config.min_performance_improvement:
            return {
                \"decision\": \"continue_validation\",
                \"reason\": f\"Marginal improvement: {performance_improvement:.2%} < {self.config.min_performance_improvement:.2%}\",
                \"recommendation\": \"extend_validation\"
            }
        else:
            return {
                \"decision\": \"keep_primary\",
                \"reason\": f\"No significant improvement: {performance_improvement:.2%}\",
                \"confidence\": \"medium\" if is_statistically_significant else \"low\"
            }
    
    def stop_validation(self) -> Dict[str, Any]:
        \"\"\"Stop validation and return final summary.\"\"\"
        with self.lock:
            if not self.validation_active:
                return {\"status\": \"not_active\"}
            
            self.validation_active = False
        
        final_summary = self.get_validation_summary()
        decision = self.make_deployment_decision()
        
        logger.info(f\"Validation stopped. Decision: {decision['decision']}\")
        
        return {
            \"status\": \"stopped\",
            \"final_summary\": final_summary,
            \"deployment_decision\": decision
        }
    
    def cleanup(self):
        \"\"\"Clean up resources.\"\"\"
        if self.primary_manager:
            self.primary_manager.cleanup()
        
        if self.secondary_manager:
            self.secondary_manager.cleanup()
        
        logger.info(\"DualModelValidator cleanup completed\")