"""
Chat Controller

Handles chat endpoint logic for natural language to SQL conversion.
"""

from fastapi import HTTPException, status
from langchain_community.utilities.sql_database import SQLDatabase
from app.core.agent.agent_graph import SQLAgentGraph
from app.config.database import sync_engine
from app.config.settings import settings
from app.models.schemas import ChatRequest, ChatResponse
from app.services.cache_answer_service import CacheAnswerService

import logging

# Get logger for this module
logger = logging.getLogger("ship_rag_ai")


def detect_question_type(question: str) -> str:
    """
    Detect if question is about journeys or not.
    Returns 'journey' or 'non_journey'.
    
    Args:
        question: User's question
        
    Returns:
        'journey' if question is about journeys, 'non_journey' otherwise
    """
    journey_keywords = [
        "journey", "journeys", "movement", 
        "facility to facility", "entered", "exited",
        "path", "traveled", "transition", "route"
    ]
    
    question_lower = question.lower()
    for keyword in journey_keywords:
        if keyword in question_lower:
            return "journey"
    
    return "non_journey"


def process_chat(
    payload: ChatRequest,
    llm_model,
    vector_store,
    sql_db
) -> ChatResponse:
    """
    Process chat request and return response.
    
    Args:
        payload: Chat request payload
        llm_model: LLM service instance
        vector_store: Vector store service instance
        sql_db: SQLDatabase instance
        
    Returns:
        ChatResponse with answer, SQL query, and debug info
    """
    # Get database connection (initialize if not already done)
    if sql_db is None:
        try:
            sql_db = SQLDatabase(sync_engine)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Database connection unavailable: {str(e)}"
            )
    
    # Authentication validation
    if payload.token_id != "Test123":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID is not proper"
        )
    
    # Detect question type and user type for cache context matching
    question_type = detect_question_type(payload.question)
    user_type = "admin" if payload.user_id == "admin" else "user"
    
    # Initialize cache service with sql_db and user_id for SQL adaptation
    cache_service = CacheAnswerService(vector_store, sql_db=sql_db, user_id=payload.user_id)
    
    # Check cache BEFORE LLM execution
    cached_result = None
    if settings.VECTOR_CACHE_ENABLED:
        try:
            cached_result = cache_service.check_cached_answer(
                question=payload.question,
                user_type=user_type,
                question_type=question_type
            )
            
            if cached_result:
                # Cache hit - return immediately (skip LLM)
                logger.info(f"✅ Cache HIT - Returning cached answer (similarity: {cached_result['similarity']:.4f})")
                print(f"\n{'='*80}")
                print(f"✅ CACHE HIT - Skipping LLM")
                print(f"   Similarity: {cached_result['similarity']:.4f}")
                print(f"   Question Type: {question_type}")
                print(f"   User Type: {user_type}")
                print(f"{'='*80}\n")
                
                return ChatResponse(
                    token_id=payload.token_id,
                    answer=cached_result["answer"],
                    sql_query=cached_result.get("sql_query"),
                    results=cached_result.get("result_data"),
                    cached=True,
                    similarity=cached_result["similarity"],
                    llm_used=False,
                    tokens_saved="~8000-11000",
                    debug={
                        "cache_hit": True,
                        "original_question": cached_result["original_question"],
                        "question_type": question_type,
                        "user_type": user_type
                    }
                )
        except Exception as e:
            logger.warning(f"Error checking cache (continuing with LLM): {e}")
            # Continue with LLM flow if cache check fails
    
    # Cache miss - proceed with existing LLM flow
    logger.info("Cache MISS - Proceeding with LLM execution")
    print(f"\n{'='*80}")
    print(f"❌ CACHE MISS - Proceeding with LLM")
    print(f"{'='*80}\n")
    
    # Process user question
    try:
        logger.info("="*80)
        logger.info("📥 API REQUEST RECEIVED")
        logger.info(f"Question: {payload.question}")
        logger.info(f"User ID: {payload.user_id}")
        logger.info(f"Chat History Length: {len(payload.chat_history or [])}")
        
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
        provider = llm_model.get_provider()
        
        logger.info(f"LLM Provider: {provider}")
        logger.info(f"LLM Model: {model_name}")
        print(f"🤖 LLM Provider: {provider}")
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
        logger.info("Starting agent invocation")
        result = agent.invoke(payload.question)
        
        answer = result.get("answer", "No answer generated")
        sql_query = result.get("sql_query", "")
        query_result = result.get("query_result", "")
        debug_info = result.get("debug", {})
        
        # Extract result data for caching (if available)
        result_data = None
        if query_result:
            try:
                import json
                # Try to parse query_result as JSON if it's a string
                if isinstance(query_result, str):
                    # First try JSON parsing
                    try:
                        result_data = json.loads(query_result)
                    except (json.JSONDecodeError, ValueError):
                        # If not JSON, try to parse as Python literal (for journey results)
                        try:
                            import ast
                            parsed = ast.literal_eval(query_result)
                            if isinstance(parsed, (dict, list)):
                                result_data = parsed
                            else:
                                result_data = {"raw_result": str(query_result)}
                        except (ValueError, SyntaxError):
                            # If all parsing fails, store as string
                            result_data = {"raw_result": str(query_result)}
                elif isinstance(query_result, (dict, list)):
                    result_data = query_result
                else:
                    result_data = {"raw_result": str(query_result)}
            except Exception as e:
                logger.debug(f"Error parsing query_result for cache: {e}")
                result_data = {"raw_result": str(query_result)}
        
        logger.info(f"Agent completed - Answer length: {len(answer)}, SQL query: {bool(sql_query)}, Query result: {bool(query_result)}")
        
        # Save to cache AFTER successful LLM response (if deterministic)
        if settings.VECTOR_CACHE_ENABLED and settings.VECTOR_CACHE_AUTO_SAVE:
            if answer and not result.get("error"):
                try:
                    cache_id = cache_service.save_answer_to_cache(
                        question=payload.question,
                        answer=answer,
                        sql_query=sql_query,
                        result_data=result_data,
                        question_type=question_type,
                        user_type=user_type
                    )
                    
                    if cache_id:
                        logger.info(f"✅ Answer saved to cache (ID: {cache_id})")
                        print(f"✅ Answer saved to cache (ID: {cache_id})")
                    else:
                        logger.debug("Answer not saved to cache (not deterministic or validation failed)")
                except Exception as e:
                    logger.warning(f"Error saving to cache (non-critical): {e}")
                    # Don't fail the request if cache save fails
        
    except Exception as e:
        # Handle any errors that occur during processing
        logger.error(f"Error processing question: {e}")
        logger.exception("Full error traceback")
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
    
    # Format and return response
    sql_query_str = sql_query if sql_query else None
    
    return ChatResponse(
        token_id=payload.token_id,
        answer=answer,
        sql_query=sql_query_str,
        results=result_data if 'result_data' in locals() else None,
        cached=False,
        llm_used=True,
        debug=debug_info
    )

