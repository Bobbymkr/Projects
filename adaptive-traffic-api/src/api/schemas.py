"""
Pydantic Schemas for Request/Response Validation.

Comprehensive data models for API request validation and response serialization.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


# Enums
class TrafficPhase(str, Enum):
    """Traffic signal phase enumeration."""
    PHASE_1 = "phase_1"
    PHASE_2 = "phase_2"
    PHASE_3 = "phase_3"
    PHASE_4 = "phase_4"


class IntersectionStatus(str, Enum):
    """Intersection status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# Request Schemas
class TrafficDecisionRequest(BaseModel):
    """Request schema for traffic signal decision."""
    
    intersection_id: str = Field(..., description="Unique intersection identifier")
    queue_lengths: List[float] = Field(..., description="Queue lengths for each lane", min_items=1, max_items=8)
    wait_times: List[float] = Field(..., description="Average wait times per lane", min_items=1, max_items=8)
    throughput: float = Field(..., description="Current throughput (vehicles/hour)", ge=0)
    current_phase: int = Field(..., description="Current signal phase", ge=0, le=3)
    
    @validator("queue_lengths", "wait_times")
    def validate_consistent_lengths(cls, v, values):
        """Ensure queue lengths and wait times have consistent lengths."""
        if "queue_lengths" in values and len(v) != len(values["queue_lengths"]):
            raise ValueError("Queue lengths and wait times must have same length")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "intersection_id": "int-001",
                "queue_lengths": [12.5, 8.3, 15.2, 10.1],
                "wait_times": [25.3, 18.7, 32.1, 22.5],
                "throughput": 450.0,
                "current_phase": 1,
            }
        }


class BatchTrafficDecisionRequest(BaseModel):
    """Request schema for batch traffic decisions."""
    
    requests: List[TrafficDecisionRequest] = Field(..., description="List of traffic decision requests", min_items=1, max_items=100)


class TrafficStateUpdate(BaseModel):
    """Schema for real-time traffic state updates."""
    
    intersection_id: str
    timestamp: datetime
    queue_lengths: List[float]
    wait_times: List[float]
    current_phase: int
    throughput: float
    status: IntersectionStatus


# Response Schemas
class TrafficDecisionResponse(BaseModel):
    """Response schema for traffic signal decision."""
    
    intersection_id: str
    recommended_phase: int = Field(..., description="Recommended signal phase", ge=0, le=3)
    green_time: float = Field(..., description="Recommended green time duration (seconds)", ge=5.0, le=60.0)
    confidence: float = Field(..., description="Decision confidence score", ge=0.0, le=1.0)
    algorithm_used: str = Field(..., description="Algorithm used for decision")
    reasoning: Optional[str] = Field(None, description="Human-readable reasoning")
    estimated_improvement: Optional[float] = Field(None, description="Estimated wait time improvement (%)")
    processing_time_ms: float = Field(..., description="Decision processing time in milliseconds")
    timestamp: datetime


class BatchTrafficDecisionResponse(BaseModel):
    """Response schema for batch traffic decisions."""
    
    decisions: List[TrafficDecisionResponse]
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_processing_time_ms: float
    timestamp: datetime


class IntersectionMetrics(BaseModel):
    """Metrics for a single intersection."""
    
    intersection_id: str
    name: str
    status: IntersectionStatus
    total_vehicles_processed: int = Field(..., ge=0)
    average_wait_time: float = Field(..., ge=0)
    average_queue_length: float = Field(..., ge=0)
    current_phase: int
    uptime_hours: float = Field(..., ge=0)
    efficiency_score: float = Field(..., ge=0.0, le=1.0)


class SystemHealthResponse(BaseModel):
    """System health status response."""
    
    status: str = Field(..., description="Overall system status")
    version: str
    uptime_seconds: float
    cpu_usage_percent: float = Field(..., ge=0.0, le=100.0)
    memory_usage_percent: float = Field(..., ge=0.0, le=100.0)
    active_intersections: int = Field(..., ge=0)
    total_requests: int = Field(..., ge=0)
    error_rate: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime


class KPIMetricsResponse(BaseModel):
    """Key Performance Indicators response."""
    
    total_intersections: int = Field(..., ge=0)
    active_intersections: int = Field(..., ge=0)
    total_vehicles_processed: int = Field(..., ge=0)
    average_response_time_ms: float = Field(..., ge=0)
    cost_savings: float = Field(..., ge=0)
    environmental_impact: Dict[str, float]
    user_satisfaction: float = Field(..., ge=0.0, le=5.0)
    system_efficiency: float = Field(..., ge=0.0, le=100.0)
    timestamp: datetime


class AlertResponse(BaseModel):
    """Alert/notification response."""
    
    id: str
    type: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    acknowledged: bool
    intersection_id: Optional[str] = None


class PerformanceMetricsResponse(BaseModel):
    """Performance metrics response."""
    
    endpoint: str
    requests_total: int = Field(..., ge=0)
    requests_per_second: float = Field(..., ge=0)
    average_response_time_ms: float = Field(..., ge=0)
    p50_response_time_ms: float = Field(..., ge=0)
    p95_response_time_ms: float = Field(..., ge=0)
    p99_response_time_ms: float = Field(..., ge=0)
    error_rate: float = Field(..., ge=0.0, le=1.0)
    success_rate: float = Field(..., ge=0.0, le=1.0)


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    
    error: str
    message: str
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Optional[Dict[str, Any]] = None


# WebSocket Message Schemas
class WebSocketMessage(BaseModel):
    """Base WebSocket message schema."""
    
    type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any]


class TrafficUpdateMessage(WebSocketMessage):
    """Real-time traffic update message."""
    
    type: str = "traffic_update"
    intersection_id: str
    metrics: IntersectionMetrics


class SystemStatusMessage(WebSocketMessage):
    """System status update message."""
    
    type: str = "system_status"
    health: SystemHealthResponse

