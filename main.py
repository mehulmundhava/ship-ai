"""
FastAPI Application - Agentic RAG SQL Chat API

This is the main FastAPI application that provides a REST API endpoint
for natural language to SQL queries using LangGraph and RAG.
"""

# Load environment variables FIRST before any other imports
from dotenv import load_dotenv
from pathlib import Path

# Try to find .env file in multiple locations
base_path = Path(__file__).parent
env_paths = [
    base_path / ".env",
    base_path.parent / ".env",
    base_path.parent / "ship-ai" / ".env",
]

env_path = None
for path in env_paths:
    if path.exists():
        env_path = path
        break

if env_path:
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"📄 Main: Loaded .env from {env_path}")
else:
    load_dotenv(override=True)
    print("⚠️  Main: Using default .env loading")

from fastapi import FastAPI, HTTPException, status, Request
from contextlib import asynccontextmanager
from models import (
    ChatRequest, 
    ChatResponse, 
    HealthCheckResponse, 
    ReloadVectorStoreResponse,
    GenerateEmbeddingsRequest,
    GenerateEmbeddingsResponse
)
from llm_model import LLMModel
from db import sync_engine
from sqlalchemy import text
from langchain_community.utilities.sql_database import SQLDatabase
from vector_store import VectorStoreManager
from agent_graph import SQLAgentGraph
import datetime


# ============================================================================
# APPLICATION LIFECYCLE MANAGEMENT
# ============================================================================

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
    app.state.llm_model = LLMModel()
    print("✅ LLM model initialized")
    
    # Initialize vector stores (PostgreSQL pgvector)
    app.state.vector_store = VectorStoreManager()
    app.state.vector_store.initialize_stores()
    print("✅ Vector stores initialized")
    
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


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthCheckResponse)
def health_check(request: Request):
    """
    Health Check Endpoint
    
    Verifies that database connection and LLM model are working properly.
    """
    from datetime import datetime
    from sqlalchemy import text
    
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
    try:
        if hasattr(request.app.state, "llm_model"):
            llm_status = {"connected": True, "message": "LLM model initialized", "error": None}
        else:
            llm_status = {"connected": False, "message": "LLM model not initialized", "error": None}
    except Exception as e:
        llm_status = {"connected": False, "message": "LLM check failed", "error": str(e)}
    
    overall_status = "healthy" if db_status["connected"] and llm_status["connected"] else "unhealthy"
    
    return HealthCheckResponse(
        status=overall_status,
        database=db_status,
        llm=llm_status,
        timestamp=datetime.now().isoformat()
    )


@app.post("/chat", response_model=ChatResponse)
def chat_api(request: Request, payload: ChatRequest):
    """
    Chat API Endpoint
    
    Accepts natural language questions and returns:
    - A natural language answer
    - The SQL query that was generated
    - Debug information
    
    Process Flow:
        1. Validate authentication token
        2. Create SQL agent graph with LangGraph
        3. Process question through agent
        4. Return formatted response
    """
    
    # Get pre-initialized components from app state
    llm_model = request.app.state.llm_model
    vector_store = request.app.state.vector_store
    
    # Get database connection (initialize if not already done)
    sql_db = request.app.state.sql_db
    if sql_db is None:
        try:
            sql_db = SQLDatabase(sync_engine)
            request.app.state.sql_db = sql_db
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Database connection unavailable: {str(e)}"
            )
    
    # ========================================================================
    # AUTHENTICATION VALIDATION
    # ========================================================================
    
    if payload.token_id != "Test123":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID is not proper"
        )
    
    # ========================================================================
    # PROCESS USER QUESTION
    # ========================================================================
    
    try:
        print(f"\n{'='*80}")
        print(f"📥 API REQUEST RECEIVED")
        print(f"{'='*80}")
        print(f"   Question: {payload.question}")
        print(f"   User ID: {payload.user_id}")
        print(f"   Chat History Length: {len(payload.chat_history or [])}")
        print(f"{'='*80}\n")
        
        # Get LLM instance (OpenAI or Groq based on LLM_PROVIDER env var)
        llm = llm_model.get_llm_model()
        model_name = getattr(llm, 'model_name', None) or getattr(llm, 'model', None) or 'Unknown'
        print(f"🤖 LLM Provider: {llm_model.get_provider()}")
        print(f"🤖 LLM Model: {model_name}")
        
        # Create SQL agent graph
        agent = SQLAgentGraph(
            llm=llm,
            db=sql_db,
            vector_store_manager=vector_store,
            user_id=payload.user_id,
            top_k=20
        )
        
        # Process the question
        result = agent.invoke(payload.question)
        
        answer = result.get("answer", "No answer generated")
        sql_query = result.get("sql_query", "")
        query_result = result.get("query_result", "")
        debug_info = result.get("debug", {})
        
    except Exception as e:
        # Handle any errors that occur during processing
        print(f"Error processing question: {e}")
        import traceback
        traceback.print_exc()
        answer = f"An error occurred while processing your request: {str(e)}"
        sql_query = None
        query_result = None
        debug_info = {
            "error": str(e),
            "question": payload.question,
            "chat_history_length": len(payload.chat_history or [])
        }
    
    # ========================================================================
    # FORMAT AND RETURN RESPONSE
    # ========================================================================
    
    # Format SQL query for response
    sql_query_str = sql_query if sql_query else None
    
    return ChatResponse(
        token_id=payload.token_id,
        answer=answer,
        sql_query=sql_query_str,
        debug=debug_info
    )


@app.post("/reload-vector-store", response_model=ReloadVectorStoreResponse)
def reload_vector_store(request: Request):
    """
    Reload Vector Store Endpoint
    
    Verifies PostgreSQL vector store tables and returns record counts.
    Note: Data is stored in PostgreSQL tables (ai_vector_examples and ai_vector_extra_prompts).
    To update data, insert/update records directly in PostgreSQL.
    
    Returns:
        ReloadVectorStoreResponse with status and counts
    """
    print(f"\n{'='*80}")
    print(f"🔄 RELOAD VECTOR STORE REQUEST RECEIVED")
    print(f"{'='*80}\n")
    
    try:
        # Get vector store manager from app state
        vector_store = request.app.state.vector_store
        
        # Reload stores
        result = vector_store.reload_stores()
        
        # Re-initialize to ensure stores are loaded
        vector_store.initialize_stores()
        
        return ReloadVectorStoreResponse(
            status=result["status"],
            message=result["message"],
            examples_count=result.get("examples_count"),
            extra_prompts_count=result.get("extra_prompts_count")
        )
    except Exception as e:
        error_msg = f"Error reloading vector stores: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return ReloadVectorStoreResponse(
            status="error",
            message=error_msg
        )


@app.post("/generate-embeddings-examples", response_model=GenerateEmbeddingsResponse)
def generate_embeddings_examples(request: Request, payload: GenerateEmbeddingsRequest):
    """
    Generate Embeddings for Examples Table
    
    Generates embeddings for records in ai_vector_examples table.
    
    - If no `id` is provided: Processes all records with NULL/empty embeddings
    - If `id` is provided: Updates embedding for that specific record (replaces if exists)
    
    Args:
        payload: GenerateEmbeddingsRequest with optional id
        
    Returns:
        GenerateEmbeddingsResponse with status and processed count
    """
    print(f"\n{'='*80}")
    print(f"🔧 GENERATE EMBEDDINGS - Examples Table")
    print(f"{'='*80}")
    print(f"   ID: {payload.id if payload.id else 'All records with NULL embeddings'}")
    print(f"{'='*80}\n")
    
    try:
        vector_store = request.app.state.vector_store
        processed_count = 0
        updated_ids = []
        errors = []
        
        with sync_engine.connect() as conn:
            if payload.id:
                # Process specific ID
                query = text("""
                    SELECT id, question, sql_query
                    FROM ai_vector_examples
                    WHERE id = :id
                """)
                result = conn.execute(query, {"id": payload.id})
                rows = result.fetchall()
                
                if not rows:
                    return GenerateEmbeddingsResponse(
                        status="error",
                        message=f"Record with id {payload.id} not found",
                        processed_count=0
                    )
            else:
                # Process all records with NULL embeddings
                query = text("""
                    SELECT id, question, sql_query
                    FROM ai_vector_examples
                    WHERE minilm_embedding IS NULL
                """)
                result = conn.execute(query)
                rows = result.fetchall()
            
            if not rows:
                return GenerateEmbeddingsResponse(
                    status="success",
                    message="No records to process",
                    processed_count=0
                )
            
            print(f"   Found {len(rows)} record(s) to process")
            
            # Process each record
            for row in rows:
                try:
                    record_id = row.id
                    question = row.question
                    sql_query = row.sql_query
                    
                    # Combine question and SQL for embedding (same format as search)
                    content = f"Question: {question}\n\nSQL Query:\n{sql_query}"
                    
                    # Generate embedding
                    embedding = vector_store.embed_query(content)
                    
                    # Convert to PostgreSQL array format string
                    embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                    
                    # Update the record
                    update_query = text("""
                        UPDATE ai_vector_examples
                        SET minilm_embedding = :embedding::vector
                        WHERE id = :id
                    """)
                    conn.execute(update_query, {
                        "id": record_id,
                        "embedding": embedding_str
                    })
                    conn.commit()
                    
                    processed_count += 1
                    updated_ids.append(record_id)
                    print(f"   ✓ Processed ID {record_id}: {question[:50]}...")
                    
                except Exception as e:
                    error_msg = f"Error processing ID {row.id}: {str(e)}"
                    errors.append(error_msg)
                    print(f"   ❌ {error_msg}")
                    import traceback
                    traceback.print_exc()
            
            message = f"Successfully processed {processed_count} record(s)"
            if errors:
                message += f" with {len(errors)} error(s)"
            
            print(f"\n✅ Embedding generation completed!")
            print(f"   Processed: {processed_count} record(s)")
            print(f"{'='*80}\n")
            
            return GenerateEmbeddingsResponse(
                status="success",
                message=message,
                processed_count=processed_count,
                updated_ids=updated_ids if updated_ids else None,
                errors=errors if errors else None
            )
            
    except Exception as e:
        error_msg = f"Error generating embeddings: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return GenerateEmbeddingsResponse(
            status="error",
            message=error_msg,
            processed_count=0
        )


@app.post("/generate-embeddings-extra-prompts", response_model=GenerateEmbeddingsResponse)
def generate_embeddings_extra_prompts(request: Request, payload: GenerateEmbeddingsRequest):
    """
    Generate Embeddings for Extra Prompts Table
    
    Generates embeddings for records in ai_vector_extra_prompts table.
    
    - If no `id` is provided: Processes all records with NULL/empty embeddings
    - If `id` is provided: Updates embedding for that specific record (replaces if exists)
    
    Args:
        payload: GenerateEmbeddingsRequest with optional id
        
    Returns:
        GenerateEmbeddingsResponse with status and processed count
    """
    print(f"\n{'='*80}")
    print(f"🔧 GENERATE EMBEDDINGS - Extra Prompts Table")
    print(f"{'='*80}")
    print(f"   ID: {payload.id if payload.id else 'All records with NULL embeddings'}")
    print(f"{'='*80}\n")
    
    try:
        vector_store = request.app.state.vector_store
        processed_count = 0
        updated_ids = []
        errors = []
        
        with sync_engine.connect() as conn:
            if payload.id:
                # Process specific ID
                query = text("""
                    SELECT id, content
                    FROM ai_vector_extra_prompts
                    WHERE id = :id
                """)
                result = conn.execute(query, {"id": payload.id})
                rows = result.fetchall()
                
                if not rows:
                    return GenerateEmbeddingsResponse(
                        status="error",
                        message=f"Record with id {payload.id} not found",
                        processed_count=0
                    )
            else:
                # Process all records with NULL embeddings
                query = text("""
                    SELECT id, content
                    FROM ai_vector_extra_prompts
                    WHERE minilm_embedding IS NULL
                """)
                result = conn.execute(query)
                rows = result.fetchall()
            
            if not rows:
                return GenerateEmbeddingsResponse(
                    status="success",
                    message="No records to process",
                    processed_count=0
                )
            
            print(f"   Found {len(rows)} record(s) to process")
            
            # Process each record
            for row in rows:
                try:
                    record_id = row.id
                    content = row.content
                    
                    # Generate embedding directly from content
                    embedding = vector_store.embed_query(content)
                    
                    # Convert to PostgreSQL array format string
                    embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                    
                    # Update the record
                    update_query = text("""
                        UPDATE ai_vector_extra_prompts
                        SET minilm_embedding = :embedding::vector
                        WHERE id = :id
                    """)
                    conn.execute(update_query, {
                        "id": record_id,
                        "embedding": embedding_str
                    })
                    conn.commit()
                    
                    processed_count += 1
                    updated_ids.append(record_id)
                    print(f"   ✓ Processed ID {record_id}: {content[:50]}...")
                    
                except Exception as e:
                    error_msg = f"Error processing ID {row.id}: {str(e)}"
                    errors.append(error_msg)
                    print(f"   ❌ {error_msg}")
                    import traceback
                    traceback.print_exc()
            
            message = f"Successfully processed {processed_count} record(s)"
            if errors:
                message += f" with {len(errors)} error(s)"
            
            print(f"\n✅ Embedding generation completed!")
            print(f"   Processed: {processed_count} record(s)")
            print(f"{'='*80}\n")
            
            return GenerateEmbeddingsResponse(
                status="success",
                message=message,
                processed_count=processed_count,
                updated_ids=updated_ids if updated_ids else None,
                errors=errors if errors else None
            )
            
    except Exception as e:
        error_msg = f"Error generating embeddings: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return GenerateEmbeddingsResponse(
            status="error",
            message=error_msg,
            processed_count=0
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3009)

