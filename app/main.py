"""
FastAPI Application - Agentic RAG SQL Chat API

This is the main FastAPI application that provides a REST API endpoint
for natural language to SQL queries using LangGraph and RAG.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from langchain_community.utilities.sql_database import SQLDatabase
from app.config.database import sync_engine
from app.services.llm_service import LLMService
from app.services.vector_store_service import VectorStoreService
from app.api import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    
    Handles startup and shutdown logic:
    - Initialize LLM model
    - Initialize vector stores (PostgreSQL pgvector)
    - Initialize database connection
    """
    # ========================================================================
    # STARTUP: Initialize all components once
    # ========================================================================
    
    print("🚀 Starting application...")
    
    # Initialize the LLM model
    app.state.llm_model = LLMService()
    print("✅ LLM model initialized")
    
    # Initialize vector stores (PostgreSQL pgvector)
    app.state.vector_store = VectorStoreService()
    try:
        app.state.vector_store.initialize_stores()
        print("✅ Vector stores initialized")
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize vector stores: {e}")
        print("   The app will start, but vector search will fail until connection is available.")
        import traceback
        traceback.print_exc()
    
    # Initialize database connection (lazy - will connect on first use)
    # Note: SQLDatabase tries to connect during init, so we catch errors
    print("🔍 Attempting to initialize SQLDatabase wrapper...")
    try:
        app.state.sql_db = SQLDatabase(sync_engine)
        print("✅ PostgreSQL database connection initialized")
        
        # Test the connection
        try:
            print("🔍 Testing database connection...")
            with sync_engine.connect() as conn:
                from sqlalchemy import text
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            print("✅ Database connection test successful")
        except Exception as test_error:
            print(f"⚠️  Database connection test failed: {test_error}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize database connection: {e}")
        print("   The app will start, but database queries will fail until connection is available.")
        import traceback
        traceback.print_exc()
        app.state.sql_db = None
    
    print("✅ Application ready to serve requests")
    
    # Yield control back to FastAPI
    yield
    
    # ========================================================================
    # SHUTDOWN: Clean up resources
    # ========================================================================
    
    if hasattr(app.state, "sql_db") and app.state.sql_db is not None:
        if hasattr(app.state.sql_db, "dispose"):
            app.state.sql_db.dispose()
    
    print("👋 Application shutting down")


# ============================================================================
# FASTAPI APPLICATION INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Ship RAG AI - Agentic SQL Chat API",
    description="Natural language to SQL using LangGraph and RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Include API routes
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3009)

