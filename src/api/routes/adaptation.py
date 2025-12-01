"""API routes for regional adaptation system."""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging

from ...adaptation.adaptation_manager import AdaptationManager
from ...adaptation.checklist_parser import ChecklistParser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/adaptation", tags=["adaptation"])

# Initialize managers
adaptation_manager = AdaptationManager()
checklist_parser = ChecklistParser()


class ChecklistRequest(BaseModel):
    """Request model for checklist-based adaptation."""
    checklist: Dict[str, Any] = Field(..., description="Checklist data")
    region_name: str = Field(default="new_region", description="Name of the region")
    generate_config: bool = Field(default=True, description="Generate configuration file")


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    region_name: str
    recommended_control: str
    recommended_vision: str
    recommended_forecasting: str
    recommended_deployment: str
    confidence_score: float
    estimated_cost: float
    estimated_wait_time: float
    reasoning: List[str]
    alternatives: List[Dict[str, Any]]


class AdaptationResponse(BaseModel):
    """Response model for complete adaptation."""
    region_name: str
    summary: Dict[str, Any]
    recommendation: Dict[str, Any]
    configuration: Dict[str, Any]
    next_steps: List[str]


@router.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: ChecklistRequest):
    """Get technology recommendations based on checklist.
    
    This endpoint analyzes the provided checklist and returns
    intelligent technology recommendations with reasoning.
    """
    try:
        # Parse checklist
        requirements = checklist_parser.parse_from_dict(request.checklist)
        
        # Generate recommendations
        tech_stack = adaptation_manager.recommender.recommend(requirements)
        
        return RecommendationResponse(
            region_name=request.region_name,
            recommended_control=tech_stack.control_algorithm.value,
            recommended_vision=tech_stack.vision_model.value,
            recommended_forecasting=tech_stack.forecasting_model.value,
            recommended_deployment=tech_stack.deployment_architecture.value,
            confidence_score=tech_stack.confidence_score,
            estimated_cost=tech_stack.estimated_cost,
            estimated_wait_time=tech_stack.estimated_performance.get(
                "estimated_wait_time_seconds", 0
            ),
            reasoning=tech_stack.reasoning,
            alternatives=[alt.to_dict() for alt in tech_stack.alternatives]
        )
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/adapt", response_model=AdaptationResponse)
async def adapt_region(request: ChecklistRequest):
    """Complete adaptation workflow.
    
    This endpoint performs the complete adaptation workflow:
    1. Parses checklist
    2. Generates recommendations
    3. Generates configuration (if requested)
    4. Returns comprehensive report
    """
    try:
        report = adaptation_manager.adapt_region(
            checklist_data=request.checklist,
            region_name=request.region_name,
            output_dir=None  # Don't save files in API mode
        )
        
        return AdaptationResponse(
            region_name=report["region_name"],
            summary=report["summary"],
            recommendation=report["recommendation"],
            configuration=report["configuration"] if request.generate_config else {},
            next_steps=report["next_steps"]
        )
    except Exception as e:
        logger.error(f"Error in adaptation workflow: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/template")
async def get_checklist_template():
    """Get checklist template.
    
    Returns a template checklist structure that can be filled out
    and submitted for recommendations.
    """
    try:
        template = checklist_parser.create_template()
        return {
            "template": template,
            "description": "Fill out this template with your regional requirements"
        }
    except Exception as e:
        logger.error(f"Error generating template: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "regional_adaptation",
        "version": "1.0.0"
    }

