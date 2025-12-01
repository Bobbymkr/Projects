"""
Traffic Control API Routes.

Endpoints for traffic signal decision making and control.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List
from datetime import datetime
import time
import logging

from ..schemas import (
    TrafficDecisionRequest,
    TrafficDecisionResponse,
    BatchTrafficDecisionRequest,
    BatchTrafficDecisionResponse,
    IntersectionMetrics,
)
from ..dependencies import get_traffic_controller, rate_limit
from ..monitoring import metrics

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/decision", response_model=TrafficDecisionResponse)
async def make_traffic_decision(
    request: TrafficDecisionRequest,
    background_tasks: BackgroundTasks,
    _rate_limit: None = Depends(rate_limit),
):
    """
    Get optimal traffic signal decision for a single intersection.
    
    Uses AI algorithms (DQN, Fuzzy Logic, GNN) to determine the best
    signal timing based on current traffic conditions.
    """
    start_time = time.time()
    
    try:
        # Record metrics
        metrics.traffic_decision_requests_total.inc()
        
        # Get traffic controller (service layer)
        controller = await get_traffic_controller()
        
        # Make decision
        decision = await controller.make_decision(
            intersection_id=request.intersection_id,
            queue_lengths=request.queue_lengths,
            wait_times=request.wait_times,
            throughput=request.throughput,
            current_phase=request.current_phase,
        )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Record processing time
        metrics.traffic_decision_duration_seconds.observe(processing_time / 1000)
        
        response = TrafficDecisionResponse(
            intersection_id=request.intersection_id,
            recommended_phase=decision["phase"],
            green_time=decision["green_time"],
            confidence=decision["confidence"],
            algorithm_used=decision["algorithm"],
            reasoning=decision.get("reasoning"),
            estimated_improvement=decision.get("improvement"),
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow(),
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error making traffic decision: {e}", exc_info=True)
        metrics.traffic_decision_errors_total.inc()
        raise HTTPException(status_code=500, detail=f"Decision processing failed: {str(e)}")


@router.post("/batch", response_model=BatchTrafficDecisionResponse)
async def make_batch_decisions(
    request: BatchTrafficDecisionRequest,
    _rate_limit: None = Depends(rate_limit),
):
    """
    Process multiple traffic decisions in a single batch request.
    
    Optimized for processing multiple intersections efficiently.
    """
    start_time = time.time()
    successful = 0
    failed = 0
    decisions = []
    
    try:
        controller = await get_traffic_controller()
        
        # Process batch requests
        for req in request.requests:
            try:
                decision = await controller.make_decision(
                    intersection_id=req.intersection_id,
                    queue_lengths=req.queue_lengths,
                    wait_times=req.wait_times,
                    throughput=req.throughput,
                    current_phase=req.current_phase,
                )
                
                decisions.append(
                    TrafficDecisionResponse(
                        intersection_id=req.intersection_id,
                        recommended_phase=decision["phase"],
                        green_time=decision["green_time"],
                        confidence=decision["confidence"],
                        algorithm_used=decision["algorithm"],
                        reasoning=decision.get("reasoning"),
                        estimated_improvement=decision.get("improvement"),
                        processing_time_ms=(time.time() - start_time) * 1000,
                        timestamp=datetime.utcnow(),
                    )
                )
                successful += 1
                
            except Exception as e:
                logger.error(f"Error processing batch request for {req.intersection_id}: {e}")
                failed += 1
                continue
        
        total_time = (time.time() - start_time) * 1000
        
        return BatchTrafficDecisionResponse(
            decisions=decisions,
            total_requests=len(request.requests),
            successful_requests=successful,
            failed_requests=failed,
            total_processing_time_ms=total_time,
            timestamp=datetime.utcnow(),
        )
        
    except Exception as e:
        logger.error(f"Error processing batch decisions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@router.get("/intersections", response_model=List[IntersectionMetrics])
async def get_intersections(
    status: str = None,
    _rate_limit: None = Depends(rate_limit),
):
    """Get metrics for all intersections."""
    try:
        controller = await get_traffic_controller()
        intersections = await controller.get_all_intersections(status=status)
        
        return intersections
        
    except Exception as e:
        logger.error(f"Error fetching intersections: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch intersections: {str(e)}")


@router.get("/intersections/{intersection_id}", response_model=IntersectionMetrics)
async def get_intersection(
    intersection_id: str,
    _rate_limit: None = Depends(rate_limit),
):
    """Get detailed metrics for a specific intersection."""
    try:
        controller = await get_traffic_controller()
        intersection = await controller.get_intersection(intersection_id)
        
        if not intersection:
            raise HTTPException(status_code=404, detail=f"Intersection {intersection_id} not found")
        
        return intersection
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching intersection {intersection_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch intersection: {str(e)}")

