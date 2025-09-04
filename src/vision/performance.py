"""
Performance Monitoring and Logging for Adaptive Traffic Vision System

Provides comprehensive performance tracking including:
- FPS monitoring and statistics
- Processing time measurements  
- Memory and GPU utilization tracking
- Queue estimation accuracy metrics
- System resource monitoring
- Real-time performance logging and visualization
"""

import time
import threading
import logging
from typing import Dict, Any, List, Optional, Deque
from collections import deque, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import json
import psutil
import gc
from contextlib import contextmanager

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
    nvml.nvmlInit()
except ImportError:
    NVML_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics snapshot."""
    timestamp: float
    fps: float
    frame_processing_time: float
    detection_time: float
    tracking_time: float
    queue_estimation_time: float
    total_vehicles_detected: int
    active_tracks: int
    dropped_frames: int
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None


@dataclass
class SystemResourceMetrics:
    """System resource utilization metrics."""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    gpu_utilization: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_temperature: Optional[float] = None


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str, monitor: 'PerformanceMonitor'):
        self.name = name
        self.monitor = monitor
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time
            self.monitor.record_timing(self.name, elapsed)


class FPSCounter:
    """Thread-safe FPS counter with rolling average."""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times: Deque[float] = deque(maxlen=window_size)
        self.lock = threading.Lock()
        self.last_frame_time = time.perf_counter()
    
    def tick(self):
        """Record a frame timestamp."""
        current_time = time.perf_counter()
        with self.lock:
            if len(self.frame_times) > 0:
                frame_interval = current_time - self.last_frame_time
                self.frame_times.append(frame_interval)
            self.last_frame_time = current_time
    
    def get_fps(self) -> float:
        """Get current FPS based on rolling average."""
        with self.lock:
            if len(self.frame_times) < 2:
                return 0.0
            
            avg_interval = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_interval if avg_interval > 0 else 0.0
    
    def get_frame_time_stats(self) -> Dict[str, float]:
        """Get frame timing statistics."""
        with self.lock:
            if len(self.frame_times) < 2:
                return {"avg": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
            
            times = list(self.frame_times)
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            # Calculate standard deviation
            variance = sum((t - avg_time) ** 2 for t in times) / len(times)
            std_time = variance ** 0.5
            
            return {
                "avg": avg_time * 1000,  # Convert to ms
                "min": min_time * 1000,
                "max": max_time * 1000,
                "std": std_time * 1000
            }


class ResourceMonitor:
    """Monitor system resource utilization."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.gpu_available = NVML_AVAILABLE
        if self.gpu_available:
            try:
                self.gpu_handle = nvml.nvmlDeviceGetHandleByIndex(0)
            except:
                self.gpu_available = False
                logger.warning("GPU monitoring unavailable")
    
    def get_system_metrics(self) -> SystemResourceMetrics:
        """Get current system resource metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = SystemResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_gb=memory.available / (1024**3),
            disk_usage_percent=disk.percent
        )
        
        # GPU metrics if available
        if self.gpu_available:
            try:
                gpu_util = nvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_memory = nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_temp = nvml.nvmlDeviceGetTemperature(self.gpu_handle, nvml.NVML_TEMPERATURE_GPU)
                
                metrics.gpu_utilization = gpu_util.gpu
                metrics.gpu_memory_used_mb = gpu_memory.used / (1024**2)
                metrics.gpu_memory_total_mb = gpu_memory.total / (1024**2)
                metrics.gpu_temperature = gpu_temp
            except Exception as e:
                logger.debug("GPU metrics collection failed due to an internal error.")
        
        return metrics
    
    def get_process_memory_mb(self) -> float:
        """Get current process memory usage in MB."""
        return self.process.memory_info().rss / (1024**2)


class PerformanceMonitor:
    """Comprehensive performance monitoring for vision pipeline."""
    
    def __init__(self, 
                 window_size: int = 30,
                 log_interval: int = 100,
                 save_interval: int = 300,
                 stats_file: Optional[str] = None):
        
        self.window_size = window_size
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.stats_file = stats_file
        
        # Performance counters
        self.fps_counter = FPSCounter(window_size)
        self.resource_monitor = ResourceMonitor()
        
        # Timing data
        self.timing_data: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window_size))
        self.timing_lock = threading.Lock()
        
        # Metrics history
        self.metrics_history: List[PerformanceMetrics] = []
        self.history_lock = threading.Lock()
        
        # Counters
        self.frame_count = 0
        self.detection_count = 0
        self.dropped_frame_count = 0
        self.start_time = time.time()
        
        # Statistics
        self.stats = {
            "total_frames_processed": 0,
            "total_detections": 0,
            "total_dropped_frames": 0,
            "session_duration": 0.0,
            "average_fps": 0.0,
            "peak_fps": 0.0,
            "min_fps": float('inf'),
            "processing_times": {}
        }
        
        # Auto-save thread
        self.save_thread = None
        self.should_stop = threading.Event()
        if stats_file:
            self.start_auto_save()
    
    def start_auto_save(self):
        """Start automatic statistics saving."""
        if self.save_thread is None:
            self.save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
            self.save_thread.start()
    
    def stop_auto_save(self):
        """Stop automatic statistics saving."""
        self.should_stop.set()
        if self.save_thread:
            self.save_thread.join(timeout=1.0)
    
    def _auto_save_loop(self):
        """Auto-save loop for statistics."""
        while not self.should_stop.wait(self.save_interval):
            try:
                self.save_stats()
            except Exception as e:
                logger.error("Failed to auto-save stats due to an internal error.")
    
    @contextmanager
    def timer(self, name: str):
        """Context manager for timing operations."""
        yield PerformanceTimer(name, self)
    
    def record_timing(self, name: str, duration: float):
        """Record timing data for an operation."""
        with self.timing_lock:
            self.timing_data[name].append(duration)
    
    def record_frame(self):
        """Record frame processing completion."""
        self.fps_counter.tick()
        self.frame_count += 1
        
        # Log performance periodically
        if self.frame_count % self.log_interval == 0:
            self._log_performance()
    
    def record_detection(self, num_detections: int):
        """Record detection results."""
        self.detection_count += num_detections
    
    def record_dropped_frame(self):
        """Record a dropped frame."""
        self.dropped_frame_count += 1
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics snapshot."""
        current_time = time.time()
        fps = self.fps_counter.get_fps()
        system_metrics = self.resource_monitor.get_system_metrics()
        
        # Get timing averages
        frame_time = self._get_timing_average("frame_processing")
        detection_time = self._get_timing_average("detection")
        tracking_time = self._get_timing_average("tracking")
        queue_time = self._get_timing_average("queue_estimation")
        
        return PerformanceMetrics(
            timestamp=current_time,
            fps=fps,
            frame_processing_time=frame_time,
            detection_time=detection_time,
            tracking_time=tracking_time,
            queue_estimation_time=queue_time,
            total_vehicles_detected=self.detection_count,
            active_tracks=0,  # To be set by caller
            dropped_frames=self.dropped_frame_count,
            memory_usage_mb=self.resource_monitor.get_process_memory_mb(),
            cpu_usage_percent=system_metrics.cpu_percent,
            gpu_usage_percent=system_metrics.gpu_utilization,
            gpu_memory_mb=system_metrics.gpu_memory_used_mb
        )
    
    def _get_timing_average(self, name: str) -> float:
        """Get average timing for an operation."""
        with self.timing_lock:
            times = self.timing_data.get(name, [])
            if not times:
                return 0.0
            return sum(times) / len(times) * 1000  # Convert to ms
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        current_time = time.time()
        session_duration = current_time - self.start_time
        
        fps = self.fps_counter.get_fps()
        frame_stats = self.fps_counter.get_frame_time_stats()
        system_metrics = self.resource_monitor.get_system_metrics()
        
        # Update statistics
        self.stats.update({
            "total_frames_processed": self.frame_count,
            "total_detections": self.detection_count,
            "total_dropped_frames": self.dropped_frame_count,
            "session_duration": session_duration,
            "average_fps": self.frame_count / session_duration if session_duration > 0 else 0,
            "current_fps": fps,
            "peak_fps": max(self.stats.get("peak_fps", 0), fps),
            "min_fps": min(self.stats.get("min_fps", float('inf')), fps) if fps > 0 else self.stats.get("min_fps", 0)
        })
        
        # Timing statistics
        timing_stats = {}
        with self.timing_lock:
            for name, times in self.timing_data.items():
                if times:
                    timing_stats[name] = {
                        "avg_ms": sum(times) / len(times) * 1000,
                        "min_ms": min(times) * 1000,
                        "max_ms": max(times) * 1000,
                        "count": len(times)
                    }
        
        return {
            "session": self.stats,
            "current": {
                "fps": fps,
                "frame_timing": frame_stats,
                "system_resources": {
                    "cpu_percent": system_metrics.cpu_percent,
                    "memory_percent": system_metrics.memory_percent,
                    "memory_available_gb": system_metrics.memory_available_gb,
                    "process_memory_mb": self.resource_monitor.get_process_memory_mb()
                }
            },
            "timing_breakdown": timing_stats
        }
    
    def _log_performance(self):
        """Log current performance metrics."""
        try:
            metrics = self.get_current_metrics()
            logger.info(
                f"Performance: FPS={metrics.fps:.1f}, "
                f"Frame={metrics.frame_processing_time:.1f}ms, "
                f"Detection={metrics.detection_time:.1f}ms, "
                f"Memory={metrics.memory_usage_mb:.1f}MB, "
                f"Dropped={metrics.dropped_frames}"
            )
            
            # Store in history
            with self.history_lock:
                self.metrics_history.append(metrics)
                # Keep only recent history
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
                    
        except Exception as e:
            logger.error("Failed to log performance due to an internal error.")
    
    def save_stats(self):
        """Save statistics to file."""
        if not self.stats_file:
            return
        
        try:
            stats = self.get_performance_summary()
            stats_path = Path(self.stats_file)
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
                
            logger.debug(f"Performance stats saved to {self.stats_file}")
            
        except Exception as e:
            logger.error("Failed to save stats due to an internal error.")
    
    def reset_stats(self):
        """Reset all performance statistics."""
        self.frame_count = 0
        self.detection_count = 0
        self.dropped_frame_count = 0
        self.start_time = time.time()
        
        with self.timing_lock:
            self.timing_data.clear()
        
        with self.history_lock:
            self.metrics_history.clear()
        
        self.stats = {
            "total_frames_processed": 0,
            "total_detections": 0,
            "total_dropped_frames": 0,
            "session_duration": 0.0,
            "average_fps": 0.0,
            "peak_fps": 0.0,
            "min_fps": float('inf'),
            "processing_times": {}
        }
        
        logger.info("Performance statistics reset")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get detailed memory usage information."""
        import gc
        
        # Force garbage collection
        gc.collect()
        
        process_memory = self.resource_monitor.get_process_memory_mb()
        system_metrics = self.resource_monitor.get_system_metrics()
        
        memory_info = {
            "process_memory_mb": process_memory,
            "system_memory_percent": system_metrics.memory_percent,
            "system_memory_available_gb": system_metrics.memory_available_gb,
            "python_objects": len(gc.get_objects())
        }
        
        if system_metrics.gpu_memory_used_mb is not None:
            memory_info.update({
                "gpu_memory_used_mb": system_metrics.gpu_memory_used_mb,
                "gpu_memory_total_mb": system_metrics.gpu_memory_total_mb,
                "gpu_memory_percent": (system_metrics.gpu_memory_used_mb / system_metrics.gpu_memory_total_mb) * 100
            })
        
        return memory_info
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop_auto_save()
        if self.stats_file:
            try:
                self.save_stats()
            except:
                pass


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None

def get_performance_monitor() -> Optional[PerformanceMonitor]:
    """Get the global performance monitor instance."""
    return _global_monitor

def initialize_performance_monitoring(window_size: int = 30,
                                    log_interval: int = 100,
                                    save_interval: int = 300,
                                    stats_file: Optional[str] = None) -> PerformanceMonitor:
    """Initialize global performance monitoring."""
    global _global_monitor
    
    if _global_monitor is not None:
        _global_monitor.stop_auto_save()
    
    _global_monitor = PerformanceMonitor(
        window_size=window_size,
        log_interval=log_interval,
        save_interval=save_interval,
        stats_file=stats_file
    )
    
    logger.info("Performance monitoring initialized")
    return _global_monitor

def cleanup_performance_monitoring():
    """Cleanup global performance monitoring."""
    global _global_monitor
    
    if _global_monitor is not None:
        _global_monitor.stop_auto_save()
        _global_monitor = None
    
    logger.info("Performance monitoring cleaned up")