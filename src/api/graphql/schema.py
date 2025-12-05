"""
GraphQL Schema for Adaptive Traffic Control System.

Provides flexible, client-driven data queries for efficient data fetching.
"""

from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import strawberry, but handle gracefully if not available
try:
    import strawberry
    STRAWBERRY_AVAILABLE = True
except ImportError:
    STRAWBERRY_AVAILABLE = False
    logger.warning("Strawberry GraphQL not available. GraphQL schema will be disabled.")

from datetime import datetime

# Try to import schemas, but handle gracefully
try:
    from ..schemas import IntersectionMetrics, KPIMetricsResponse, TrafficDecisionResponse
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False
    logger.warning("API schemas not available.")

if STRAWBERRY_AVAILABLE:
    from typing import List, Optional
    
    @strawberry.type
    class Intersection:
        """GraphQL type for intersection data."""
        
        id: str
        name: str
        status: str
        total_vehicles_processed: int
        average_wait_time: float
        average_queue_length: float
        current_phase: int
        uptime_hours: float
        efficiency_score: float
    
    
    @strawberry.type
    class TrafficDecision:
        """GraphQL type for traffic decision."""
        
        intersection_id: str
        action: int
        phase_duration: float
        confidence: float
        algorithm: str
        timestamp: datetime
        estimated_wait_reduction: float
    
    
    @strawberry.type
    class KPIMetrics:
        """GraphQL type for KPI metrics."""
        
        average_wait_time: float
        total_vehicles_processed: int
        system_efficiency: float
        algorithm_performance: dict
    
    
    @strawberry.type
    class Query:
        """GraphQL query type."""
        
        @strawberry.field
        def intersection(self, id: str) -> Optional[Intersection]:
            """Get intersection by ID."""
            # Fetch from traffic controller service
            try:
                from ...services.traffic_controller import TrafficController
                controller = TrafficController()
                intersection_data = controller.registry.get_intersection(id)
                if intersection_data:
                    return Intersection(
                        id=intersection_data.get("intersection_id"),
                        name=intersection_data.get("name", ""),
                        status=intersection_data.get("status", "inactive"),
                    )
            except Exception:
                pass
            return None
        
        @strawberry.field
        def intersections(self) -> List[Intersection]:
            """Get all intersections."""
            # Fetch from traffic controller service
            try:
                from ...services.traffic_controller import TrafficController
                controller = TrafficController()
                all_intersections = controller.registry.get_all_intersections()
                return [
                    Intersection(
                        id=inter.get("intersection_id"),
                        name=inter.get("name", ""),
                        status=inter.get("status", "inactive"),
                    )
                    for inter in all_intersections
                ]
            except Exception:
                pass
            return []
        
        @strawberry.field
        def traffic_decision(self, intersection_id: str) -> Optional[TrafficDecision]:
            """Get latest traffic decision for intersection."""
            # Fetch from traffic controller service
            # Note: This would require storing decision history
            # For now, return None as decision history not yet implemented
            return None
        
        @strawberry.field
        def kpi_metrics(self, time_range: str = "day") -> Optional[KPIMetrics]:
            """Get KPI metrics for time range."""
            # Fetch from metrics service
            # For synthetic data context, would aggregate from Prometheus or metrics store
            # In production: metrics = await metrics_service.get_kpis(time_range)
            return None
    
    
    # Create GraphQL schema
    schema = strawberry.Schema(query=Query)
else:
    # Fallback: Create dummy schema
    schema = None
    logger.warning("GraphQL schema not available without strawberry.")
