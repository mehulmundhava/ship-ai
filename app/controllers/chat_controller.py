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
    
    # Detect question type for 80% match and agent
    question_type = detect_question_type(payload.question)

    # Match-and-execute service (80% path: base example columns only, no cache)
    cache_service = CacheAnswerService(vector_store, sql_db=sql_db, user_id=payload.user_id)

    # 80% match-and-execute: always check. Base columns only, no cache. If similarity ≥ 0.80,
    # run example SQL (adapted or as-is) and return without LLM.
    logger.info("🔍 Checking 80% match-and-execute path...")
    try:
        match_result = cache_service.check_80_match_and_execute(
            question=payload.question,
            question_type=question_type,
        )
        if match_result:
            logger.info(
                f"✅ 80% match-and-execute hit (similarity: {match_result['similarity']:.4f})"
            )
            return ChatResponse(
                token_id=payload.token_id,
                answer=match_result["answer"],
                sql_query=match_result.get("sql_query"),
                results=match_result.get("result_data"),
                cached=False,
                similarity=match_result["similarity"],
                llm_used=False,
                llm_type=None,
                debug={
                    "cache_hit": False,
                    "match_80": True,
                    "original_question": match_result.get("original_question", payload.question),
                    "question_type": question_type,
                },
            )
    except Exception as e:
        logger.warning(f"80% match-and-execute failed (continuing with LLM): {e}")

    # No ≥80% match or execution failed — proceed with LLM
    logger.info("Proceeding with LLM execution")
    print(f"\n{'='*80}")
    print(f"Proceeding with LLM")
    print(f"{'='*80}\n")

    llm_type = None  # Set when LLM is used (e.g. OPENAI/gpt-4o)

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
        llm_type = f"{provider}/{model_name}"

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
        llm_type=llm_type,
        debug=debug_info
    )

