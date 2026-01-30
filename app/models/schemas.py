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
    results: Optional[Dict[str, Any]] = None  # Result data from query execution
    cached: Optional[bool] = False  # True if answer came from cache
    similarity: Optional[float] = None  # Similarity score (only when cached)
    llm_used: Optional[bool] = True  # False if answer came from cache
    llm_type: Optional[str] = None  # Model used when LLM was used (e.g. OPENAI/gpt-4o)
    tokens_saved: Optional[str] = None  # Estimated tokens saved (only when cached)
    debug: Optional[Dict[str, Any]] = None


class HealthCheckResponse(BaseModel):
    """
    Response model for the /health API endpoint.
    """
    status: str
    database: dict
    llm: dict
    timestamp: str


class ReloadVectorStoreResponse(BaseModel):
    """
    Response model for the /reload-vector-store API endpoint.
    """
    status: str
    message: str
    examples_count: Optional[int] = None
    extra_prompts_count: Optional[int] = None


class GenerateEmbeddingsRequest(BaseModel):
    """
    Request model for embedding generation API endpoints.
    """
    id: Optional[int] = None


class GenerateEmbeddingsResponse(BaseModel):
    """
    Response model for embedding generation API endpoints.
    """
    status: str
    message: str
    processed_count: Optional[int] = None
    updated_ids: Optional[List[int]] = None
    errors: Optional[List[str]] = None


class VectorSearchRequest(BaseModel):
    """
    Request model for the /vector-search API endpoint.
    """
    question: str
    search_type: Optional[str] = "both"  # "examples", "extra_prompts", or "both"
    k_examples: Optional[int] = 3  # Number of example results
    k_extra_prompts: Optional[int] = 2  # Number of extra prompt results
    example_id: Optional[int] = None  # Filter by specific example ID (to check distance)
    extra_prompts_id: Optional[int] = None  # Filter by specific extra prompt ID (to check distance)


class VectorSearchResult(BaseModel):
    """
    Individual search result item.
    """
    id: Optional[int] = None
    content: str
    distance: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class VectorSearchResponse(BaseModel):
    """
    Response model for the /vector-search API endpoint.
    """
    status: str
    question: str
    search_type: str
    examples: Optional[List[VectorSearchResult]] = None
    extra_prompts: Optional[List[VectorSearchResult]] = None
    total_results: int


class GetTextEmbeddingRequest(BaseModel):
    """
    Request model for the /get-text-embedding API endpoint.
    """
    text: str


class GetTextEmbeddingResponse(BaseModel):
    """
    Response model for the /get-text-embedding API endpoint.
    """
    status: str
    text: str
    embedding: List[float]
    embedding_dimension: int


class SearchEmbeddingRequest(BaseModel):
    """
    Request model for the /search-embedding API endpoint.
    """
    text: str
    limit: Optional[int] = 5


class SearchEmbeddingResult(BaseModel):
    """
    Individual search result item for embedding search.
    """
    id: int
    content: str
    similarity: float


class SearchEmbeddingResponse(BaseModel):
    """
    Response model for the /search-embedding API endpoint.
    """
    status: str
    text: str
    limit: int
    results: List[SearchEmbeddingResult]
    total_results: int