"""
Monitoring and Observability Routes.

Endpoints for Prometheus metrics, health checks, and system observability.
"""

from fastapi import APIRouter, Response, Depends
from fastapi.responses import PlainTextResponse
from typing import Optional
import logging

from ..dependencies import rate_limit
from ..monitoring import metrics

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/metrics")
async def prometheus_metrics(
    _rate_limit: None = Depends(rate_limit),
):
    """
    Prometheus metrics endpoint.
    
    Exports metrics in Prometheus exposition format for scraping.
    This endpoint should be scraped by Prometheus server at regular intervals.
    """
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        
        # Generate metrics in Prometheus format
        metrics_output = generate_latest()
        
        return Response(
            content=metrics_output,
            media_type=CONTENT_TYPE_LATEST,
        )
    except ImportError:
        # If prometheus_client not installed, return empty response
        logger.warning("Prometheus client not installed. Metrics endpoint unavailable.")
        return PlainTextResponse(
            content="# Prometheus metrics not available\n",
            media_type="text/plain",
        )
    except Exception as e:
        logger.error(f"Error generating Prometheus metrics: {e}", exc_info=True)
        return PlainTextResponse(
            content=f"# Error generating metrics: {str(e)}\n",
            media_type="text/plain",
        )

