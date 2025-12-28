"""
Data Models for API Request/Response

This module defines Pydantic models for request and response validation.
"""

from typing import List, Tuple, Optional, Dict, Any
from pydantic import BaseModel


class ChatRequest(BaseModel):
    """
    Request model for the /chat API endpoint.
    """
    token_id: str
    question: str
    user_id: Optional[str] = None
    chat_history: Optional[List[Tuple[str, str]]] = []


class ChatResponse(BaseModel):
    """
    Response model for the /chat API endpoint.
    """
    token_id: str
    answer: str
    sql_query: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None


class HealthCheckResponse(BaseModel):
    """
    Response model for the /health API endpoint.
    """
    status: str
    database: dict
    llm: dict
    timestamp: str

