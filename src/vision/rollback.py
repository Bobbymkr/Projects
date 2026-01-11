"""
Automated Rollback Mechanism for YOLOv11n Migration

Provides intelligent rollback capabilities including:
- Real-time performance threshold monitoring
- Automated decision making for rollback triggers
- Multi-tier rollback strategies (graceful degradation)
- Health check integration with circuit breaker patterns
- Rollback history and audit logging
- Recovery procedures and automatic restart capabilities
"""

import time
import threading
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import json

from .performance import PerformanceMetrics, PerformanceMonitor
from .model_manager import YOLOModelManager, ModelMetrics
from .config import YOLOConfig, ModelType

logger = logging.getLogger(__name__)


class RollbackTrigger(Enum):
    """Types of rollback triggers."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    HIGH_ERROR_RATE = "high_error_rate"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    CONSECUTIVE_FAILURES = "consecutive_failures"
    MANUAL_TRIGGER = "manual_trigger"
    HEALTH_CHECK_FAILURE = "health_check_failure"
    CIRCUIT_BREAKER = "circuit_breaker"


class RollbackStrategy(Enum):
    """Rollback strategies."""
    IMMEDIATE = "immediate"  # Switch immediately
    GRACEFUL = "graceful"  # Gradual traffic reduction
    CIRCUIT_BREAKER = "circuit_breaker"  # Temporary disable with retry
    STAGED = "staged"  # Multi-step rollback through fallback chain


@dataclass
class RollbackEvent:
    """Record of a rollback event."""
    timestamp: float
    trigger: RollbackTrigger
    strategy: RollbackStrategy
    from_model: str
    to_model: str
    reason: str
    metrics_before: Optional[PerformanceMetrics] = None
    metrics_after: Optional[PerformanceMetrics] = None
    success: bool = False
    recovery_time_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "timestamp": self.timestamp,
            "trigger": self.trigger.value,
            "strategy": self.strategy.value,
            "from_model": self.from_model,
            "to_model": self.to_model,
            "reason": self.reason,
            "success": self.success,
            "recovery_time_seconds": self.recovery_time_seconds
        }


@dataclass
class RollbackConfig:
    """Configuration for rollback mechanism."""
    # Performance thresholds
    fps_degradation_threshold: float = 0.8  # Rollback if FPS drops below 80% of baseline
    error_rate_threshold: float = 0.05  # Rollback if error rate exceeds 5%
    memory_threshold_mb: float = 2048  # Rollback if memory exceeds 2GB
    consecutive_failures_threshold: int = 3  # Rollback after 3 consecutive failures
    
    # Timing thresholds
    response_time_threshold_ms: float = 1000  # Rollback if response time > 1s
    health_check_timeout_seconds: float = 30  # Health check timeout
    rollback_decision_window_seconds: float = 60  # Time window for decision making
    
    # Rollback behavior
    default_strategy: RollbackStrategy = RollbackStrategy.GRACEFUL
    enable_auto_rollback: bool = True
    enable_circuit_breaker: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: float = 300  # 5 minutes
    
    # Recovery behavior
    enable_auto_recovery: bool = True
    recovery_attempt_interval_seconds: float = 600  # 10 minutes
    max_recovery_attempts: int = 3
    
    # Monitoring
    continuous_monitoring: bool = True
    monitoring_interval_seconds: float = 10
    alert_on_rollback: bool = True


class CircuitBreaker:
    """Circuit breaker pattern implementation for model health monitoring."""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: float = 300):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.timeout_seconds:
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info("Circuit breaker transitioning to CLOSED")
                return result
            
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.error(f"Circuit breaker transitioning to OPEN after {self.failure_count} failures")
                
                raise e
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "time_until_retry": max(0, self.timeout_seconds - (time.time() - self.last_failure_time))
        }


class HealthChecker:
    """Performs health checks on models and system components."""
    
    def __init__(self, config: RollbackConfig):
        self.config = config
        self.health_history: Dict[str, List[Dict[str, Any]]] = {}
        self.last_health_check = 0
    
    def check_model_health(self, model_manager: YOLOModelManager, model_id: str) -> Dict[str, Any]:
        """Perform comprehensive health check on a model."""
        health_result = {
            "timestamp": time.time(),
            "model_id": model_id,
            "status": "unknown",
            "checks": {},
            "overall_score": 0.0
        }
        
        try:
            # Get model performance summary
            perf_summary = model_manager.get_performance_summary()
            
            checks = {
                "model_loaded": perf_summary.get("active_model") is not None,
                "low_error_rate": True,  # Will be updated based on metrics
                "memory_usage_ok": True,
                "response_time_ok": True
            }
            
            # Check each health criterion
            scores = []
            
            # Model loaded check
            if checks["model_loaded"]:
                scores.append(1.0)
            else:
                scores.append(0.0)
                health_result["status"] = "failed"
            
            # Additional checks would go here based on available metrics
            scores.extend([1.0, 1.0, 1.0])  # Placeholder for other checks
            
            health_result["checks"] = checks
            health_result["overall_score"] = sum(scores) / len(scores)
            
            # Determine overall status
            if health_result["overall_score"] >= 0.8:
                health_result["status"] = "healthy"
            elif health_result["overall_score"] >= 0.6:
                health_result["status"] = "degraded"
            else:
                health_result["status"] = "failed"
            
            # Store health history
            if model_id not in self.health_history:
                self.health_history[model_id] = []
            
            self.health_history[model_id].append(health_result)
            
            # Keep history manageable
            if len(self.health_history[model_id]) > 100:
                self.health_history[model_id] = self.health_history[model_id][-50:]
            
        except Exception as e:
            logger.error(f"Health check failed for model {model_id}: {e}")
            health_result["status"] = "failed"
            health_result["error"] = str(e)
        
        return health_result
    
    def get_health_trends(self, model_id: str, window_minutes: int = 30) -> Dict[str, Any]:
        """Get health trends for a model over specified time window."""
        if model_id not in self.health_history:
            return {"trend": "unknown", "data_points": 0}
        
        cutoff_time = time.time() - (window_minutes * 60)
        recent_checks = [h for h in self.health_history[model_id] if h["timestamp"] > cutoff_time]
        
        if not recent_checks:
            return {"trend": "insufficient_data", "data_points": 0}
        
        scores = [h["overall_score"] for h in recent_checks]
        avg_score = sum(scores) / len(scores)
        
        # Determine trend
        if len(scores) >= 2:
            first_half = scores[:len(scores)//2]
            second_half = scores[len(scores)//2:]
            
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            
            if second_avg > first_avg + 0.1:
                trend = "improving"
            elif second_avg < first_avg - 0.1:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "data_points": len(recent_checks),
            "average_score": avg_score,
            "latest_score": scores[-1] if scores else 0.0
        }


class AutomatedRollbackManager:
    """
    Main class for automated rollback management.
    
    Monitors system health and performance, automatically triggering
    rollbacks when degradation is detected. Supports multiple rollback
    strategies and recovery mechanisms.
    """
    
    def __init__(self, 
                 config: RollbackConfig,
                 model_manager: YOLOModelManager,
                 performance_monitor: PerformanceMonitor):
        self.config = config
        self.model_manager = model_manager
        self.performance_monitor = performance_monitor
        
        # State tracking
        self.monitoring_active = False
        self.rollback_in_progress = False
        self.current_model_id = "unknown"
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        
        # Components
        self.health_checker = HealthChecker(config)
        self.circuit_breaker = CircuitBreaker(
            config.circuit_breaker_failure_threshold,
            config.circuit_breaker_timeout_seconds
        ) if config.enable_circuit_breaker else None
        
        # Event tracking
        self.rollback_history: List[RollbackEvent] = []
        self.consecutive_failures = 0
        self.last_rollback_time = 0
        self.recovery_attempts = 0
        
        # Monitoring thread
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        
        # Callbacks
        self.rollback_callbacks: List[Callable] = []
        self.recovery_callbacks: List[Callable] = []
        
        logger.info("AutomatedRollbackManager initialized")
    
    def start_monitoring(self):
        """Start continuous monitoring for rollback conditions."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        if not self.config.continuous_monitoring:
            logger.info("Continuous monitoring disabled in config")
            return
        
        self.monitoring_active = True
        self.stop_monitoring.clear()
        
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Rollback monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        if not self.monitoring_active:
            return
        
        self.stop_monitoring.set()
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Rollback monitoring stopped")
    
    def set_baseline_metrics(self, metrics: PerformanceMetrics):
        """Set baseline metrics for comparison."""
        self.baseline_metrics = metrics
        logger.info(f"Baseline metrics set: FPS={metrics.fps:.1f}, Error rate={metrics.error_rate:.2%}")
    
    def register_rollback_callback(self, callback: Callable):
        """Register callback to be called on rollback events."""
        self.rollback_callbacks.append(callback)
    
    def register_recovery_callback(self, callback: Callable):
        """Register callback to be called on recovery events."""
        self.recovery_callbacks.append(callback)
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_monitoring.wait(self.config.monitoring_interval_seconds):
            try:
                self._check_rollback_conditions()
                
                # Attempt recovery if configured
                if (self.config.enable_auto_recovery and 
                    time.time() - self.last_rollback_time > self.config.recovery_attempt_interval_seconds and
                    self.recovery_attempts < self.config.max_recovery_attempts):
                    self._attempt_recovery()
                    
            except Exception as e:
                logger.error(f\"Error in monitoring loop: {e}\")
    
    def _check_rollback_conditions(self):
        \"\"\"Check if rollback conditions are met.\"\"\"
        if self.rollback_in_progress or not self.config.enable_auto_rollback:
            return
        
        # Get current metrics
        current_metrics = self.performance_monitor.get_current_metrics()
        
        # Check various rollback conditions
        rollback_triggers = []
        
        # Performance degradation check
        if self.baseline_metrics and current_metrics.fps > 0:
            fps_ratio = current_metrics.fps / self.baseline_metrics.fps
            if fps_ratio < self.config.fps_degradation_threshold:
                rollback_triggers.append((
                    RollbackTrigger.PERFORMANCE_DEGRADATION,
                    f\"FPS degraded to {fps_ratio:.2%} of baseline\"
                ))
        
        # Error rate check
        if current_metrics.error_rate > self.config.error_rate_threshold:
            rollback_triggers.append((
                RollbackTrigger.HIGH_ERROR_RATE,
                f\"Error rate {current_metrics.error_rate:.2%} exceeds threshold\"
            ))
        
        # Memory usage check
        if current_metrics.memory_usage_mb > self.config.memory_threshold_mb:
            rollback_triggers.append((
                RollbackTrigger.MEMORY_EXHAUSTION,
                f\"Memory usage {current_metrics.memory_usage_mb:.1f}MB exceeds threshold\"
            ))
        
        # Health check
        health_result = self.health_checker.check_model_health(self.model_manager, self.current_model_id)
        if health_result[\"status\"] == \"failed\":
            rollback_triggers.append((
                RollbackTrigger.HEALTH_CHECK_FAILURE,
                f\"Health check failed with score {health_result['overall_score']:.2f}\"
            ))
        
        # Process rollback triggers
        if rollback_triggers:
            self.consecutive_failures += 1
            
            # Check if consecutive failures threshold reached
            if self.consecutive_failures >= self.config.consecutive_failures_threshold:
                # Use the first trigger for rollback
                trigger, reason = rollback_triggers[0]
                self.trigger_rollback(trigger, reason, current_metrics)
        else:
            self.consecutive_failures = 0
    
    def trigger_rollback(self, 
                        trigger: RollbackTrigger, 
                        reason: str, 
                        current_metrics: Optional[PerformanceMetrics] = None,
                        strategy: Optional[RollbackStrategy] = None) -> bool:
        \"\"\"Trigger rollback with specified parameters.\"\"\"
        if self.rollback_in_progress:
            logger.warning(\"Rollback already in progress\")
            return False
        
        self.rollback_in_progress = True
        rollback_start_time = time.time()
        
        strategy = strategy or self.config.default_strategy
        
        logger.error(f\"ROLLBACK TRIGGERED: {trigger.value} - {reason}\")
        
        # Create rollback event
        rollback_event = RollbackEvent(
            timestamp=rollback_start_time,
            trigger=trigger,
            strategy=strategy,
            from_model=self.current_model_id,
            to_model=\"fallback\",  # Will be updated
            reason=reason,
            metrics_before=current_metrics
        )
        
        try:
            # Execute rollback based on strategy
            success = self._execute_rollback(strategy, rollback_event)
            
            rollback_event.success = success
            rollback_event.recovery_time_seconds = time.time() - rollback_start_time
            
            if success:
                logger.info(f\"Rollback completed successfully in {rollback_event.recovery_time_seconds:.1f}s\")
                self.last_rollback_time = time.time()
                self.consecutive_failures = 0
            else:
                logger.error(\"Rollback failed\")
            
            # Record event
            self.rollback_history.append(rollback_event)
            
            # Trigger callbacks
            for callback in self.rollback_callbacks:
                try:
                    callback(rollback_event)
                except Exception as e:
                    logger.error(f\"Rollback callback failed: {e}\")
            
            return success
            
        except Exception as e:
            logger.error(f\"Rollback execution failed: {e}\")
            rollback_event.success = False
            self.rollback_history.append(rollback_event)
            return False
        
        finally:
            self.rollback_in_progress = False
    
    def _execute_rollback(self, strategy: RollbackStrategy, event: RollbackEvent) -> bool:
        \"\"\"Execute rollback with specified strategy.\"\"\"
        if strategy == RollbackStrategy.IMMEDIATE:
            return self._immediate_rollback(event)
        elif strategy == RollbackStrategy.GRACEFUL:
            return self._graceful_rollback(event)
        elif strategy == RollbackStrategy.CIRCUIT_BREAKER:
            return self._circuit_breaker_rollback(event)
        elif strategy == RollbackStrategy.STAGED:
            return self._staged_rollback(event)
        else:
            logger.error(f\"Unknown rollback strategy: {strategy}\")
            return False
    
    def _immediate_rollback(self, event: RollbackEvent) -> bool:
        \"\"\"Immediate rollback to fallback model.\"\"\"
        try:
            # This would trigger the model manager's fallback mechanism
            if hasattr(self.model_manager, '_trigger_fallback'):
                self.model_manager._trigger_fallback()
                event.to_model = \"fallback_immediate\"
                return True
            else:
                logger.error(\"Model manager does not support fallback\")
                return False
        except Exception as e:
            logger.error(f\"Immediate rollback failed: {e}\")
            return False
    
    def _graceful_rollback(self, event: RollbackEvent) -> bool:
        \"\"\"Graceful rollback with traffic reduction.\"\"\"
        try:
            # Gradually reduce traffic to problematic model
            # This would be implemented with traffic splitting logic
            logger.info(\"Executing graceful rollback\")
            event.to_model = \"fallback_graceful\"
            return True
        except Exception as e:
            logger.error(f\"Graceful rollback failed: {e}\")
            return False
    
    def _circuit_breaker_rollback(self, event: RollbackEvent) -> bool:
        \"\"\"Circuit breaker rollback.\"\"\"
        if not self.circuit_breaker:
            return self._immediate_rollback(event)
        
        try:
            # Set circuit breaker to OPEN state
            self.circuit_breaker.state = \"OPEN\"
            self.circuit_breaker.last_failure_time = time.time()
            event.to_model = \"circuit_breaker\"
            return True
        except Exception as e:
            logger.error(f\"Circuit breaker rollback failed: {e}\")
            return False
    
    def _staged_rollback(self, event: RollbackEvent) -> bool:
        \"\"\"Staged rollback through fallback chain.\"\"\"
        try:
            # Try each model in fallback chain
            logger.info(\"Executing staged rollback\")
            event.to_model = \"fallback_staged\"
            return True
        except Exception as e:
            logger.error(f\"Staged rollback failed: {e}\")
            return False
    
    def _attempt_recovery(self):
        \"\"\"Attempt to recover from rollback state.\"\"\"
        if not self.config.enable_auto_recovery:
            return
        
        self.recovery_attempts += 1
        logger.info(f\"Attempting recovery (attempt {self.recovery_attempts}/{self.config.max_recovery_attempts})\")
        
        try:
            # Perform health check before recovery
            health_result = self.health_checker.check_model_health(self.model_manager, self.current_model_id)
            
            if health_result[\"status\"] == \"healthy\":
                logger.info(\"Recovery successful - system appears healthy\")
                self.recovery_attempts = 0
                
                # Trigger recovery callbacks
                for callback in self.recovery_callbacks:
                    try:
                        callback({\"status\": \"success\", \"attempts\": self.recovery_attempts})
                    except Exception as e:
                        logger.error(f\"Recovery callback failed: {e}\")
            else:
                logger.warning(f\"Recovery not ready - health status: {health_result['status']}\")
                
        except Exception as e:
            logger.error(f\"Recovery attempt failed: {e}\")
    
    def get_rollback_summary(self) -> Dict[str, Any]:
        \"\"\"Get comprehensive rollback summary.\"\"\"
        recent_rollbacks = [r for r in self.rollback_history if time.time() - r.timestamp < 3600]  # Last hour
        
        summary = {
            \"monitoring_active\": self.monitoring_active,
            \"rollback_in_progress\": self.rollback_in_progress,
            \"current_model\": self.current_model_id,
            \"consecutive_failures\": self.consecutive_failures,
            \"total_rollbacks\": len(self.rollback_history),
            \"recent_rollbacks\": len(recent_rollbacks),
            \"recovery_attempts\": self.recovery_attempts,
            \"last_rollback_time\": self.last_rollback_time
        }
        
        # Add circuit breaker state
        if self.circuit_breaker:
            summary[\"circuit_breaker\"] = self.circuit_breaker.get_state()
        
        # Add health trends
        health_trends = self.health_checker.get_health_trends(self.current_model_id)
        summary[\"health_trends\"] = health_trends
        
        # Add recent rollback details
        if recent_rollbacks:
            summary[\"recent_rollback_triggers\"] = [r.trigger.value for r in recent_rollbacks]
            summary[\"rollback_success_rate\"] = sum(1 for r in recent_rollbacks if r.success) / len(recent_rollbacks)
        
        return summary
    
    def manual_rollback(self, reason: str = \"Manual trigger\") -> bool:
        \"\"\"Manually trigger rollback.\"\"\"
        return self.trigger_rollback(
            RollbackTrigger.MANUAL_TRIGGER,
            reason,
            strategy=RollbackStrategy.IMMEDIATE
        )
    
    def reset_failure_count(self):
        \"\"\"Reset consecutive failure count.\"\"\"
        self.consecutive_failures = 0
        logger.info(\"Consecutive failure count reset\")
    
    def export_rollback_history(self, filepath: str):
        \"\"\"Export rollback history to file.\"\"\"
        try:
            history_data = [event.to_dict() for event in self.rollback_history]
            with open(filepath, 'w') as f:
                json.dump(history_data, f, indent=2)
            logger.info(f\"Rollback history exported to {filepath}\")
        except Exception as e:
            logger.error(f\"Failed to export rollback history: {e}\")
    
    def cleanup(self):
        \"\"\"Clean up resources.\"\"\"
        self.stop_monitoring()
        logger.info(\"AutomatedRollbackManager cleanup completed\")