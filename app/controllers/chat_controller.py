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
from app.services.message_history_service import save_message_to_history

import logging
import time

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
    t0 = time.perf_counter()
    try:
        match_result = cache_service.check_80_match_and_execute(
            question=payload.question,
            question_type=question_type,
        )
        if match_result:
            elapsed = time.perf_counter() - t0
            logger.info(f"[80% path] hit in {elapsed:.2f}s similarity={match_result['similarity']:.4f}")
            result_data = match_result.get("result_data") or {}
            csv_path = result_data.get("csv_download_link")
            csv_id_80 = result_data.get("csv_id")
            if csv_path and not csv_path.startswith("/"):
                csv_path = f"/download-csv/{csv_id_80}" if csv_id_80 else csv_path
            resp = ChatResponse(
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
                csv_id=csv_id_80,
                csv_download_path=csv_path,
            )
            save_message_to_history(
                user_id=payload.user_id,
                login_id=payload.login_id,
                token_id=payload.token_id,
                question=payload.question,
                response=resp.answer,
                sql_query=resp.sql_query,
                cached=False,
                similarity=match_result["similarity"],
                llm_used=False,
                llm_type=None,
                question_type=question_type,
                debug_info=resp.debug,
                result_data=match_result.get("result_data"),
                chat_history_length=len(payload.chat_history or []),
            )
            return resp
        elapsed_80 = time.perf_counter() - t0
        logger.info(f"[80% path] miss in {elapsed_80:.2f}s → LLM")
    except Exception as e:
        logger.warning(f"80% match failed (continuing with LLM): {e}")

    # No ≥80% match or execution failed — proceed with LLM
    llm_type = None  # Set when LLM is used (e.g. OPENAI/gpt-4o)
    t_request = time.perf_counter()

    # Process user question
    try:
        logger.info(f"[chat] user_id={payload.user_id} question_len={len(payload.question)} history_len={len(payload.chat_history or [])}")
        
        # Get LLM instance (OpenAI or Groq based on LLM_PROVIDER env var)
        llm = llm_model.get_llm_model()
        model_name = getattr(llm, 'model_name', None) or getattr(llm, 'model', None) or 'Unknown'
        provider = llm_model.get_provider()
        llm_type = f"{provider}/{model_name}"

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
        elapsed_request = time.perf_counter() - t_request
        logger.info(f"[chat] agent done in {elapsed_request:.2f}s")
        
        answer = result.get("answer", "No answer generated")
        sql_query = result.get("sql_query", "")
        query_result = result.get("query_result", "")
        debug_info = result.get("debug", {})
        csv_id = result.get("csv_id")
        csv_download_path = result.get("csv_download_path")
        
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
        
    except Exception as e:
        # Handle any errors that occur during processing
        logger.error(f"Error processing question: {e}")
        logger.exception("Full error traceback")
        answer = f"An error occurred while processing your request: {str(e)}"
        sql_query = None
        query_result = None
        debug_info = {
            "error": str(e),
            "question": payload.question,
            "chat_history_length": len(payload.chat_history or [])
        }
        error_message = str(e)
        result_data = None

    # Format and return response
    sql_query_str = sql_query if sql_query else None
    resp = ChatResponse(
        token_id=payload.token_id,
        answer=answer,
        sql_query=sql_query_str,
        results=result_data if 'result_data' in locals() else None,
        cached=False,
        llm_used=True,
        llm_type=llm_type,
        debug=debug_info,
        csv_id=csv_id if 'csv_id' in locals() else None,
        csv_download_path=csv_download_path if 'csv_download_path' in locals() else None,
    )
    save_message_to_history(
        user_id=payload.user_id,
        login_id=payload.login_id,
        token_id=payload.token_id,
        question=payload.question,
        response=resp.answer,
        sql_query=resp.sql_query,
        cached=False,
        similarity=None,
        llm_used=True,
        llm_type=llm_type,
        question_type=question_type,
        debug_info=resp.debug,
        result_data=resp.results,
        error_message=error_message if 'error_message' in locals() else None,
        chat_history_length=len(payload.chat_history or []),
    )
    return resp

