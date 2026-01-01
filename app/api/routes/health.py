"""
Health Check Routes
"""

from fastapi import APIRouter, Request
from app.controllers.health_controller import check_health
from app.models.schemas import HealthCheckResponse

router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse)
def health_check(request: Request):
    """
    Health Check Endpoint
    
    Verifies that database connection and LLM model are working properly.
    """
    response = check_health()
    
    # Update LLM status based on app state
    if hasattr(request.app.state, "llm_model"):
        response.llm = {"connected": True, "message": "LLM model initialized", "error": None}
    else:
        response.llm = {"connected": False, "message": "LLM model not initialized", "error": None}
    
    # Recalculate overall status
    response.status = "healthy" if response.database["connected"] and response.llm["connected"] else "unhealthy"
    
    return response

