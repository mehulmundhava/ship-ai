"""
Chat Routes
"""

from fastapi import APIRouter, Request
from app.controllers.chat_controller import process_chat
from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat_api(request: Request, payload: ChatRequest):
    """
    Chat API Endpoint
    
    Accepts natural language questions and returns:
    - A natural language answer
    - The SQL query that was generated
    - Debug information
    """
    # Get pre-initialized components from app state
    llm_model = request.app.state.llm_model
    vector_store = request.app.state.vector_store
    sql_db = request.app.state.sql_db
    
    return process_chat(
        payload=payload,
        llm_model=llm_model,
        vector_store=vector_store,
        sql_db=sql_db
    )

