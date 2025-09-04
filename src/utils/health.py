"""Health check endpoints and utilities.

This module provides health check endpoints and utilities for monitoring the
system's health, including quota status, system resources, and component status.
"""

import os
import json
import logging
import platform
import psutil
from typing import Dict, Any, Optional, List
from pathlib import Path

# Configure module logger
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    from fastapi import APIRouter, Response
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


class HealthCheck:
    """Health check utilities for the system."""
    
    def __init__(self, quota_manager=None):
        """Initialize the health check with optional components.
        
        Args:
            quota_manager: Optional RequestQuotaManager instance
        """
        self.quota_manager = quota_manager
        self.components = {}
    
    def register_component(self, name: str, check_func):
        """Register a component health check function.
        
        Args:
            name: Name of the component
            check_func: Function that returns (status, details) tuple
        """
        self.components[name] = check_func
    
    def check_quota(self) -> Dict[str, Any]:
        """Check quota status."""
        if not self.quota_manager:
            return {"status": "unknown", "details": "No quota manager configured"}
        
        remaining = self.quota_manager.remaining()
        max_requests = self.quota_manager.max_requests
        
        status = "healthy"
        if remaining <= 0:
            status = "critical"
        elif remaining <= self.quota_manager.alert_at_remaining:
            status = "warning"
        
        return {
            "status": status,
            "details": {
                "remaining": remaining,
                "max": max_requests,
                "used": max_requests - remaining,
                "percent_used": ((max_requests - remaining) / max_requests) * 100
            }
        }
    
    def check_system(self) -> Dict[str, Any]:
        """Check system resources."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        status = "healthy"
        if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
            status = "critical"
        elif cpu_percent > 75 or memory.percent > 75 or disk.percent > 75:
            status = "warning"
        
        details = {
            "cpu": {
                "percent": cpu_percent
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent
            },
            "disk": {
                "total": disk.total,
                "free": disk.free,
                "percent": disk.percent
            },
            "platform": platform.platform()
        }
        
        # Add GPU info if available
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_info = []
                    for i, gpu in enumerate(gpus):
                        gpu_info.append({
                            "id": i,
                            "name": gpu.name,
                            "memory_total": gpu.memoryTotal,
                            "memory_used": gpu.memoryUsed,
                            "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                            "load": gpu.load * 100
                        })
                    details["gpu"] = gpu_info
            except Exception as e:
                logger.warning(f"Failed to get GPU info: {e}")
        
        return {"status": status, "details": details}
    
    def check_all(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {
            "system": self.check_system()
        }
        
        # Add quota check if available
        if self.quota_manager:
            results["quota"] = self.check_quota()
        
        # Add component checks
        for name, check_func in self.components.items():
            try:
                status, details = check_func()
                results[name] = {"status": status, "details": details}
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                results[name] = {"status": "error", "details": str(e)}
        
        # Determine overall status
        if any(r["status"] == "critical" for r in results.values()):
            overall = "critical"
        elif any(r["status"] == "warning" for r in results.values()):
            overall = "warning"
        elif any(r["status"] == "error" for r in results.values()):
            overall = "error"
        else:
            overall = "healthy"
        
        return {"status": overall, "components": results}


def create_health_router(health_check: HealthCheck) -> Optional[Any]:
    """Create a FastAPI router for health endpoints.
    
    Args:
        health_check: HealthCheck instance
        
    Returns:
        FastAPI router or None if FastAPI is not available
    """
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available, skipping health router creation")
        return None
    
    router = APIRouter(tags=["health"])
    
    @router.get("/health")
    async def health():
        """Overall health check endpoint."""
        result = health_check.check_all()
        status_code = 200
        if result["status"] == "critical":
            status_code = 503
        elif result["status"] == "warning":
            status_code = 429
        elif result["status"] == "error":
            status_code = 500
        
        return Response(
            content=json.dumps(result),
            media_type="application/json",
            status_code=status_code
        )
    
    @router.get("/health/quotas")
    async def quota_health():
        """Quota-specific health check endpoint."""
        if not health_check.quota_manager:
            return Response(
                content=json.dumps({"status": "unknown", "details": "No quota manager configured"}),
                media_type="application/json",
                status_code=404
            )
        
        result = health_check.check_quota()
        status_code = 200
        if result["status"] == "critical":
            status_code = 429
        elif result["status"] == "warning":
            status_code = 200  # Still OK but warning
        
        return Response(
            content=json.dumps(result),
            media_type="application/json",
            status_code=status_code
        )
    
    @router.get("/health/system")
    async def system_health():
        """System resource health check endpoint."""
        result = health_check.check_system()
        status_code = 200
        if result["status"] == "critical":
            status_code = 503
        elif result["status"] == "warning":
            status_code = 429
        
        return Response(
            content=json.dumps(result),
            media_type="application/json",
            status_code=status_code
        )
    
    return router