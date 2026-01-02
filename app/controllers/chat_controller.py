"""
Chat Controller

Handles chat endpoint logic for natural language to SQL conversion.
"""

from fastapi import HTTPException, status
from langchain_community.utilities.sql_database import SQLDatabase
from app.core.agent.agent_graph import SQLAgentGraph
from app.config.database import sync_engine
from app.models.schemas import ChatRequest, ChatResponse


import logging

# Get logger for this module
logger = logging.getLogger("ship_rag_ai")


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
        debug=debug_info
    )

