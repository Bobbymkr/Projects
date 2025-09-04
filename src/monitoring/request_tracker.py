#!/usr/bin/env python3
"""
Professional Request Tracking and Quota Management System

This module provides enterprise-grade request tracking capabilities with:
- Persistent quota management
- Professional alerting mechanisms
- Configurable limits and thresholds
- Thread-safe operations
- Comprehensive logging
- Industry-standard error handling
"""

import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any, Callable
import logging
import warnings
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class QuotaConfig:
    """Configuration for quota management."""
    total_requests: int = 150
    warning_threshold: int = 10
    reset_interval_hours: int = 24
    storage_file: str = "quota_state.json"
    enable_alerts: bool = True
    alert_sound: bool = False


class RequestTracker:
    """
    Professional request tracking system with quota management.
    
    Features:
    - Thread-safe request counting
    - Persistent state management
    - Configurable alerting thresholds
    - Professional logging and monitoring
    - Automatic quota reset capabilities
    """
    
    def __init__(self, config: Optional[QuotaConfig] = None):
        """Initialize the request tracker with professional configuration."""
        self.config = config or QuotaConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._lock = threading.Lock()
        self._state_file = Path(self.config.storage_file)
        self._alert_callbacks: list[Callable] = []
        
        # Load existing state or initialize
        self._load_state()
        
        self.logger.info(f"RequestTracker initialized with {self.remaining_requests}/{self.config.total_requests} requests remaining")
    
    def _load_state(self) -> None:
        """Load quota state from persistent storage."""
        try:
            if self._state_file.exists():
                with open(self._state_file, 'r') as f:
                    data = json.load(f)
                
                self.used_requests = data.get('used_requests', 0)
                self.start_time = datetime.fromisoformat(data.get('start_time', datetime.now().isoformat()))
                self.last_reset = datetime.fromisoformat(data.get('last_reset', datetime.now().isoformat()))
                
                # Check if quota should be reset
                if datetime.now() - self.last_reset > timedelta(hours=self.config.reset_interval_hours):
                    self._reset_quota()
            else:
                self._initialize_fresh_state()
                
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            self._initialize_fresh_state()
    
    def _initialize_fresh_state(self) -> None:
        """Initialize fresh quota state."""
        self.used_requests = 0
        self.start_time = datetime.now()
        self.last_reset = datetime.now()
        self._save_state()
    
    def _save_state(self) -> None:
        """Save current state to persistent storage."""
        try:
            state = {
                'used_requests': self.used_requests,
                'start_time': self.start_time.isoformat(),
                'last_reset': self.last_reset.isoformat(),
                'config': asdict(self.config)
            }
            
            with open(self._state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def _reset_quota(self) -> None:
        """Reset the quota counter."""
        self.used_requests = 0
        self.last_reset = datetime.now()
        self._save_state()
        self.logger.info("Quota reset successfully")
    
    @property
    def remaining_requests(self) -> int:
        """Get remaining requests in quota."""
        return max(0, self.config.total_requests - self.used_requests)
    
    @property
    def usage_percentage(self) -> float:
        """Get current usage as percentage."""
        return (self.used_requests / self.config.total_requests) * 100
    
    def add_alert_callback(self, callback: Callable[[int, Dict[str, Any]], None]) -> None:
        """Add a callback function for quota alerts."""
        self._alert_callbacks.append(callback)
    
    def _trigger_alerts(self) -> None:
        """Trigger professional alert mechanisms."""
        remaining = self.remaining_requests
        
        if remaining <= self.config.warning_threshold and self.config.enable_alerts:
            alert_data = {
                'remaining_requests': remaining,
                'total_requests': self.config.total_requests,
                'used_requests': self.used_requests,
                'usage_percentage': self.usage_percentage,
                'timestamp': datetime.now().isoformat(),
                'time_until_reset': str(timedelta(hours=self.config.reset_interval_hours) - 
                                      (datetime.now() - self.last_reset))
            }
            
            # Professional console alert
            self._display_professional_alert(alert_data)
            
            # Execute custom callbacks
            for callback in self._alert_callbacks:
                try:
                    callback(remaining, alert_data)
                except Exception as e:
                    self.logger.error(f"Alert callback failed: {e}")
            
            # Optional sound alert (Windows compatible)
            if self.config.alert_sound:
                self._play_alert_sound()
    
    def _display_professional_alert(self, alert_data: Dict[str, Any]) -> None:
        """Display a professional alert message."""
        print("\n" + "="*80)
        print("🚨 QUOTA ALERT - ADAPTIVE TRAFFIC CONTROL SYSTEM 🚨")
        print("="*80)
        print(f"📊 REMAINING REQUESTS: {alert_data['remaining_requests']} / {alert_data['total_requests']}")
        print(f"⚠️  USAGE: {alert_data['usage_percentage']:.1f}%")
        print(f"⏰ ALERT TRIGGERED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔄 TIME UNTIL RESET: {alert_data['time_until_reset']}")
        print("="*80)
        print("⚡ IMMEDIATE ACTION REQUIRED - QUOTA THRESHOLD REACHED")
        print("="*80 + "\n")
        
        # Log the alert
        self.logger.warning(f"Quota alert triggered: {alert_data['remaining_requests']} requests remaining")
    
    def _play_alert_sound(self) -> None:
        """Play alert sound on Windows."""
        try:
            import winsound
            # Professional alert tone sequence
            for freq in [800, 1000, 800]:
                winsound.Beep(freq, 200)
                time.sleep(0.1)
        except ImportError:
            self.logger.info("Sound alerts not available (winsound not found)")
        except Exception as e:
            self.logger.error(f"Failed to play alert sound: {e}")
    
    @contextmanager
    def request_context(self, operation_name: str = "operation"):
        """
        Context manager for tracking individual requests.
        
        Usage:
            with tracker.request_context("DQN_training"):
                # Your operation here
                pass
        """
        start_time = time.time()
        
        try:
            # Check quota before operation
            if self.remaining_requests <= 0:
                raise RuntimeError(f"Quota exceeded! No remaining requests. Reset in {self.config.reset_interval_hours} hours.")
            
            self.logger.info(f"Starting operation: {operation_name} ({self.remaining_requests} requests remaining)")
            yield
            
        finally:
            # Record the request
            with self._lock:
                self.used_requests += 1
                self._save_state()
                self._trigger_alerts()
            
            duration = time.time() - start_time
            self.logger.info(f"Completed operation: {operation_name} (Duration: {duration:.2f}s, {self.remaining_requests} requests remaining)")
    
    def increment_usage(self, count: int = 1, operation_name: str = "manual") -> None:
        """Manually increment usage counter."""
        with self._lock:
            self.used_requests += count
            self._save_state()
            self._trigger_alerts()
        
        self.logger.info(f"Manual usage increment: {operation_name} (+{count}, {self.remaining_requests} remaining)")
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        now = datetime.now()
        return {
            'quota_status': {
                'total_requests': self.config.total_requests,
                'used_requests': self.used_requests,
                'remaining_requests': self.remaining_requests,
                'usage_percentage': self.usage_percentage,
                'warning_threshold': self.config.warning_threshold,
                'alert_active': self.remaining_requests <= self.config.warning_threshold
            },
            'timing': {
                'current_time': now.isoformat(),
                'quota_start_time': self.start_time.isoformat(),
                'last_reset': self.last_reset.isoformat(),
                'time_since_reset': str(now - self.last_reset),
                'time_until_next_reset': str(timedelta(hours=self.config.reset_interval_hours) - (now - self.last_reset)),
                'reset_interval_hours': self.config.reset_interval_hours
            },
            'configuration': asdict(self.config)
        }
    
    def print_status_dashboard(self) -> None:
        """Print a professional status dashboard."""
        status = self.get_status_report()
        
        print("\n" + "="*60)
        print("📊 ADAPTIVE TRAFFIC CONTROL - QUOTA DASHBOARD")
        print("="*60)
        print(f"🔢 Total Quota: {status['quota_status']['total_requests']}")
        print(f"✅ Used: {status['quota_status']['used_requests']}")
        print(f"🆓 Remaining: {status['quota_status']['remaining_requests']}")
        print(f"📈 Usage: {status['quota_status']['usage_percentage']:.1f}%")
        
        if status['quota_status']['alert_active']:
            print("🚨 ALERT: Low quota warning active!")
        else:
            print("✅ STATUS: Normal operation")
            
        print(f"⏰ Time until reset: {status['timing']['time_until_next_reset']}")
        print("="*60 + "\n")
    
    def reset_quota_manual(self) -> None:
        """Manually reset the quota (admin function)."""
        with self._lock:
            self._reset_quota()
        self.logger.info("Manual quota reset performed")
        print("✅ Quota has been manually reset!")


# Global instance for easy access
_global_tracker: Optional[RequestTracker] = None

def get_global_tracker(config: Optional[QuotaConfig] = None) -> RequestTracker:
    """Get or create the global request tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = RequestTracker(config)
    return _global_tracker

def track_request(operation_name: str = "operation"):
    """Decorator for tracking function calls as requests."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracker = get_global_tracker()
            with tracker.request_context(operation_name or func.__name__):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Example usage and demo
if __name__ == "__main__":
    # Demo the professional request tracking system
    print("🚦 Adaptive Traffic Control - Request Tracking System Demo")
    
    # Create tracker with custom config
    config = QuotaConfig(
        total_requests=150,
        warning_threshold=10,
        alert_sound=True
    )
    
    tracker = RequestTracker(config)
    tracker.print_status_dashboard()
    
    # Demo adding a custom alert callback
    def custom_alert(remaining: int, alert_data: Dict[str, Any]) -> None:
        print(f"🔔 Custom Alert: Only {remaining} requests left!")
    
    tracker.add_alert_callback(custom_alert)
    
    # Simulate some operations
    print("\n🔄 Simulating operations...")
    
    with tracker.request_context("DQN_Training"):
        time.sleep(0.1)
    
    with tracker.request_context("MARL_Evaluation"):
        time.sleep(0.1)
    
    with tracker.request_context("LSTM_Forecasting"):
        time.sleep(0.1)
    
    tracker.print_status_dashboard()
    
    # Demonstrate manual usage increment
    tracker.increment_usage(5, "Batch_Processing")
    
    # Final status
    status_report = tracker.get_status_report()
    print(f"\n📋 Final Status Report:")
    print(json.dumps(status_report, indent=2))
