"""
System Management API Routes.

Endpoints for system health, status, and configuration.
"""

from fastapi import APIRouter, Depends
from datetime import datetime
import psutil
import time
import logging

from ..schemas import SystemHealthResponse
from ..dependencies import rate_limit
from ..monitoring import metrics

logger = logging.getLogger(__name__)

router = APIRouter()

# Track startup time for uptime calculation
_startup_time = time.time()


@router.get("/health", response_model=SystemHealthResponse)
async def health_check(_rate_limit: None = Depends(rate_limit)):
    """
    System health check endpoint.
    
    Returns comprehensive system health metrics including:
    - System status
    - Resource utilization (CPU, Memory)
    - Active intersections
    - Request statistics
    """
    try:
        # Get system resources
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Calculate uptime
        uptime_seconds = time.time() - _startup_time
        
        # Get metrics from Prometheus
        # These would be actual values from the metrics registry
        total_requests = metrics.http_requests_total._value.get() or 0
        
        # Determine overall status
        if cpu_percent < 80 and memory_percent < 80:
            status = "healthy"
        elif cpu_percent < 90 and memory_percent < 90:
            status = "warning"
        else:
            status = "critical"
        
        return SystemHealthResponse(
            status=status,
            version="1.0.0",
            uptime_seconds=uptime_seconds,
            cpu_usage_percent=round(cpu_percent, 2),
            memory_usage_percent=round(memory_percent, 2),
            active_intersections=4,  # TODO: Get from actual system
            total_requests=int(total_requests),
            error_rate=0.0,  # TODO: Calculate from metrics
            timestamp=datetime.utcnow(),
        )
        
    except Exception as e:
        logger.error(f"Error in health check: {e}", exc_info=True)
        return SystemHealthResponse(
            status="critical",
            version="1.0.0",
            uptime_seconds=0,
            cpu_usage_percent=0,
            memory_usage_percent=0,
            active_intersections=0,
            total_requests=0,
            error_rate=1.0,
            timestamp=datetime.utcnow(),
        )


@router.get("/status")
async def system_status(_rate_limit: None = Depends(rate_limit)):
    """Get detailed system status information."""
    return {
        "status": "operational",
        "version": "1.0.0",
        "environment": "production",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/info")
async def system_info(_rate_limit: None = Depends(rate_limit)):
    """Get system information and capabilities."""
    return {
        "name": "Adaptive Traffic Control System",
        "version": "1.0.0",
        "description": "AI-powered traffic signal control system",
        "capabilities": [
            "Real-time traffic decision making",
            "Multi-algorithm support (DQN, Fuzzy, GNN)",
            "WebSocket real-time updates",
            "REST API",
            "Batch processing",
        ],
        "algorithms": [
            "Deep Q-Network (DQN)",
            "Fuzzy Logic Control",
            "Graph Neural Networks",
            "Bayesian Inference",
        ],
        "timestamp": datetime.utcnow().isoformat(),
    }

