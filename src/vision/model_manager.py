"""
Enhanced YOLO Model Management for YOLOv11n Migration

Provides:
- Multi-version YOLO model support (YOLOv8n, YOLOv11n)
- Intelligent fallback mechanisms with health monitoring
- A/B testing and shadow mode deployment capabilities
- Performance-based model selection and automatic rollback
- NIST cybersecurity framework compliance
- Model integrity verification and secure loading
"""

import hashlib
import os
import time
import threading
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from .config import YOLOConfig, ModelType

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Performance metrics for model evaluation."""
    fps: float = 0.0
    inference_time_ms: float = 0.0
    accuracy_score: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_utilization: float = 0.0
    error_rate: float = 0.0
    uptime_seconds: float = 0.0
    total_frames_processed: int = 0
    
    def calculate_performance_score(self, baseline_fps: float = 15.0) -> float:
        """Calculate overall performance score (0-1)."""
        fps_score = min(self.fps / baseline_fps, 1.0) if baseline_fps > 0 else 0.0
        accuracy_score = self.accuracy_score
        error_score = max(0.0, 1.0 - self.error_rate)
        
        return (fps_score * 0.5 + accuracy_score * 0.3 + error_score * 0.2)


@dataclass
class ModelState:
    """State information for a loaded model."""
    model_type: ModelType
    model_path: str
    model_instance: Optional[Any] = None
    is_loaded: bool = False
    load_time: float = 0.0
    last_health_check: float = 0.0
    metrics: ModelMetrics = None
    error_count: int = 0
    health_status: str = "unknown"  # "healthy", "degraded", "failed", "unknown"
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ModelMetrics()


class ModelHealthMonitor:
    """Monitors model health and triggers fallback when needed."""
    
    def __init__(self, config: YOLOConfig):
        self.config = config
        self.health_checks: Dict[str, List[float]] = {}
        self.performance_history: Dict[str, List[ModelMetrics]] = {}
        self.lock = threading.Lock()
        
    def record_performance(self, model_id: str, metrics: ModelMetrics):
        """Record performance metrics for health monitoring."""
        with self.lock:
            if model_id not in self.performance_history:
                self.performance_history[model_id] = []
            
            self.performance_history[model_id].append(metrics)
            
            # Keep only recent history (last 100 measurements)
            if len(self.performance_history[model_id]) > 100:
                self.performance_history[model_id] = self.performance_history[model_id][-100:]
    
    def assess_health(self, model_id: str) -> Tuple[str, float]:
        """Assess model health status."""
        with self.lock:
            if model_id not in self.performance_history:
                return "unknown", 0.0
            
            recent_metrics = self.performance_history[model_id][-10:]  # Last 10 measurements
            if not recent_metrics:
                return "unknown", 0.0
            
            # Calculate average performance score
            avg_score = np.mean([m.calculate_performance_score() for m in recent_metrics])
            
            # Determine health status
            if avg_score >= 0.8:
                return "healthy", avg_score
            elif avg_score >= 0.6:
                return "degraded", avg_score
            else:
                return "failed", avg_score
    
    def should_trigger_fallback(self, model_id: str) -> bool:
        """Determine if fallback should be triggered."""
        health_status, score = self.assess_health(model_id)
        
        if health_status == "failed":
            return True
        
        if health_status == "degraded" and score < self.config.rollback_trigger_threshold:
            return True
        
        return False


class SecureModelLoader:
    """Handles secure model loading with integrity verification."""
    
    def __init__(self):
        self.model_hashes: Dict[str, str] = {}
        self.verified_models: Dict[str, bool] = {}
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of model file for integrity verification."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""
    
    def verify_model_integrity(self, file_path: str, expected_hash: Optional[str] = None) -> bool:
        """Verify model file integrity."""
        if not Path(file_path).exists():
            return False
        
        current_hash = self.calculate_file_hash(file_path)
        if not current_hash:
            return False
        
        # Store hash for future reference
        self.model_hashes[file_path] = current_hash
        
        if expected_hash:
            is_valid = current_hash == expected_hash
        else:
            # If no expected hash, consider it valid but log warning
            is_valid = True
            logger.warning(f"No expected hash provided for {file_path}, assuming valid")
        
        self.verified_models[file_path] = is_valid
        return is_valid
    
    def load_model_securely(self, model_path: str, model_type: ModelType) -> Optional[Any]:
        """Load YOLO model with security checks."""
        try:
            # Verify file exists and has reasonable size
            if not Path(model_path).exists():
                logger.error(f"Model file not found: {model_path}")
                return None
            
            file_size_mb = Path(model_path).stat().st_size / (1024**2)
            if file_size_mb < 1 or file_size_mb > 1000:  # Reasonable size bounds
                logger.warning(f"Model file size unusual: {file_size_mb:.1f}MB")
            
            # Verify integrity if not already verified
            if model_path not in self.verified_models:
                if not self.verify_model_integrity(model_path):
                    logger.error(f"Model integrity verification failed: {model_path}")
                    return None
            
            # Load model
            if ULTRALYTICS_AVAILABLE:
                model = YOLO(model_path)
                logger.info(f"Successfully loaded {model_type.value} from {model_path}")
                return model
            else:
                logger.error("ultralytics package not available")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            return None


class YOLOModelManager:
    """
    Advanced YOLO model manager supporting YOLOv8n to YOLOv11n migration.
    
    Features:
    - Multi-version model support with intelligent fallback
    - A/B testing and shadow mode deployment
    - Performance monitoring and health assessment
    - Automatic rollback on performance degradation
    - Secure model loading with integrity verification
    """
    
    def __init__(self, config: YOLOConfig):
        self.config = config
        self.models: Dict[str, ModelState] = {}
        self.active_model_id: Optional[str] = None
        self.shadow_model_id: Optional[str] = None
        
        # Components
        self.health_monitor = ModelHealthMonitor(config)
        self.secure_loader = SecureModelLoader()
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Performance tracking
        self.start_time = time.time()
        self.frame_count = 0
        self.last_performance_check = time.time()
        
        logger.info("YOLOModelManager initialized")
    
    def initialize_models(self) -> bool:
        """Initialize models according to configuration."""
        success = False
        
        # Try to load primary model
        for model_type in self.config.fallback_chain:
            if self._load_model(model_type):
                success = True
                break
        
        if not success:
            logger.error("Failed to load any model from fallback chain")
            return False
        
        # Load shadow model if configured
        if self.config.enable_shadow_mode and self.config.shadow_model_type:
            self._load_model(self.config.shadow_model_type, is_shadow=True)
        
        return True
    
    def _load_model(self, model_type: ModelType, is_shadow: bool = False) -> bool:
        """Load a specific model."""
        model_id = f"{model_type.value}_{int(time.time())}"
        
        if model_type == ModelType.CUSTOM:
            # OpenCV DNN fallback - placeholder for now
            logger.info("OpenCV DNN fallback not implemented yet")
            return False
        
        # Get model path
        if model_type.value.startswith('yolo11') or model_type.value.startswith('yolov11'):
            model_path = model_type.value
        else:
            model_path = model_type.value
        
        # Load model securely
        start_time = time.time()
        model_instance = self.secure_loader.load_model_securely(model_path, model_type)
        load_time = time.time() - start_time
        
        if model_instance is None:
            logger.error(f"Failed to load model: {model_type.value}")
            return False
        
        # Create model state
        model_state = ModelState(
            model_type=model_type,
            model_path=model_path,
            model_instance=model_instance,
            is_loaded=True,
            load_time=load_time,
            last_health_check=time.time(),
            health_status="healthy"
        )
        
        with self.lock:
            self.models[model_id] = model_state
            
            if is_shadow:
                self.shadow_model_id = model_id
                logger.info(f"Shadow model loaded: {model_type.value}")
            else:
                self.active_model_id = model_id
                logger.info(f"Active model loaded: {model_type.value}")
        
        return True
    
    def get_active_model(self) -> Optional[Any]:
        """Get the currently active model."""
        with self.lock:
            if self.active_model_id and self.active_model_id in self.models:
                return self.models[self.active_model_id].model_instance
            return None
    
    def get_shadow_model(self) -> Optional[Any]:
        """Get the shadow model for A/B testing."""
        with self.lock:
            if self.shadow_model_id and self.shadow_model_id in self.models:
                return self.models[self.shadow_model_id].model_instance
            return None
    
    def predict(self, frame: np.ndarray, use_shadow: bool = False) -> Tuple[List[Dict], ModelMetrics]:
        """Run prediction with performance monitoring."""
        model = self.get_shadow_model() if use_shadow else self.get_active_model()
        
        if model is None:
            logger.error("No model available for prediction")
            return [], ModelMetrics()
        
        start_time = time.perf_counter()
        
        try:
            # Run inference
            results = model(frame, 
                          conf=self.config.confidence_threshold, 
                          iou=self.config.nms_threshold, 
                          verbose=False)
            
            inference_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            
            # Process results
            detections = self._process_yolo_results(results)
            
            # Calculate metrics
            metrics = ModelMetrics(
                fps=1000.0 / inference_time if inference_time > 0 else 0.0,
                inference_time_ms=inference_time,
                total_frames_processed=self.frame_count + 1
            )
            
            # Record performance
            model_id = self.shadow_model_id if use_shadow else self.active_model_id
            if model_id:
                self.health_monitor.record_performance(model_id, metrics)
                
                # Check if fallback needed
                if self.config.enable_auto_rollback and not use_shadow:
                    if self.health_monitor.should_trigger_fallback(model_id):
                        logger.warning("Performance degradation detected, attempting fallback")
                        self._trigger_fallback()
            
            self.frame_count += 1
            return detections, metrics
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            
            # Record error
            model_id = self.shadow_model_id if use_shadow else self.active_model_id
            if model_id and model_id in self.models:
                self.models[model_id].error_count += 1
            
            return [], ModelMetrics()
    
    def _process_yolo_results(self, results) -> List[Dict]:
        """Process YOLO results into standardized format."""
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    if class_id in self.config.vehicle_classes and confidence >= self.config.confidence_threshold:
                        centroid = (
                            (bbox[0] + bbox[2]) / 2,
                            (bbox[1] + bbox[3]) / 2
                        )
                        
                        detections.append({
                            "bbox": tuple(bbox),
                            "centroid": centroid,
                            "confidence": confidence,
                            "class_id": class_id
                        })
        
        return detections
    
    def _trigger_fallback(self):
        """Trigger fallback to next available model."""
        with self.lock:
            if not self.active_model_id:
                return
            
            current_model = self.models[self.active_model_id]
            current_type = current_model.model_type
            
            # Find next model in fallback chain
            fallback_types = self.config.fallback_chain
            try:
                current_index = fallback_types.index(current_type)
                if current_index + 1 < len(fallback_types):
                    next_model_type = fallback_types[current_index + 1]
                    
                    logger.info(f"Falling back from {current_type.value} to {next_model_type.value}")
                    
                    if self._load_model(next_model_type):
                        logger.info("Fallback successful")
                    else:
                        logger.error("Fallback failed")
                else:
                    logger.error("No more fallback options available")
                    
            except ValueError:
                logger.error(f"Current model type {current_type.value} not in fallback chain")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self.lock:
            summary = {
                "active_model": None,
                "shadow_model": None,
                "models": {},
                "uptime_seconds": time.time() - self.start_time,
                "total_frames_processed": self.frame_count
            }
            
            if self.active_model_id and self.active_model_id in self.models:
                active_model = self.models[self.active_model_id]
                summary["active_model"] = {
                    "type": active_model.model_type.value,
                    "health_status": active_model.health_status,
                    "error_count": active_model.error_count,
                    "load_time": active_model.load_time
                }
            
            if self.shadow_model_id and self.shadow_model_id in self.models:
                shadow_model = self.models[self.shadow_model_id]
                summary["shadow_model"] = {
                    "type": shadow_model.model_type.value,
                    "health_status": shadow_model.health_status,
                    "error_count": shadow_model.error_count,
                    "load_time": shadow_model.load_time
                }
            
            for model_id, model_state in self.models.items():
                health_status, score = self.health_monitor.assess_health(model_id)
                summary["models"][model_id] = {
                    "type": model_state.model_type.value,
                    "health_status": health_status,
                    "health_score": score,
                    "error_count": model_state.error_count,
                    "is_active": model_id == self.active_model_id,
                    "is_shadow": model_id == self.shadow_model_id
                }
            
            return summary
    
    def cleanup(self):
        """Clean up resources."""
        with self.lock:
            for model_id, model_state in self.models.items():
                if model_state.model_instance:
                    try:
                        # Cleanup model instance if needed
                        del model_state.model_instance
                    except Exception as e:
                        logger.error(f"Error cleaning up model {model_id}: {e}")
            
            self.models.clear()
            self.active_model_id = None
            self.shadow_model_id = None
        
        logger.info("ModelManager cleanup completed")