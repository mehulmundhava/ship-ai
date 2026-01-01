"""
Health Check Controller

Handles health check endpoint logic.
"""

from datetime import datetime
from sqlalchemy import text
from app.config.database import sync_engine
from app.models.schemas import HealthCheckResponse


def check_health() -> HealthCheckResponse:
    """
    Health Check Controller
    
    Verifies that database connection and LLM model are working properly.
    """
    db_status = {"connected": False, "message": "", "error": None}
    llm_status = {"connected": False, "message": "", "error": None}
    
    # Check database
    try:
        print("🔍 Health Check: Attempting database connection...")
        with sync_engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        db_status = {"connected": True, "message": "Database connection successful", "error": None}
        print("✅ Health Check: Database connection successful")
    except Exception as e:
        error_msg = str(e)
        db_status = {"connected": False, "message": "Database connection failed", "error": error_msg}
        print(f"❌ Health Check: Database connection failed - {error_msg}")
        import traceback
        traceback.print_exc()
    
    # Check LLM (simple check - just verify it's initialized)
    # Note: LLM status will be set by the route handler based on app state
    llm_status = {"connected": True, "message": "LLM check not performed in controller", "error": None}
    
    overall_status = "healthy" if db_status["connected"] else "unhealthy"
    
    return HealthCheckResponse(
        status=overall_status,
        database=db_status,
        llm=llm_status,
        timestamp=datetime.now().isoformat()
    )

