"""
Vector Search Routes
"""

from fastapi import APIRouter, Request
from app.controllers.vector_search_controller import process_vector_search
from app.models.schemas import VectorSearchRequest, VectorSearchResponse

router = APIRouter()


@router.post("/vector-search", response_model=VectorSearchResponse)
def vector_search_api(request: Request, payload: VectorSearchRequest):
    """
    Vector Search API Endpoint
    
    Searches the PostgreSQL vector store for similar examples and extra prompts
    based on the provided question.
    
    Request Body:
    - question: The search query/question (required) 
    - search_type: "examples", "extra_prompts", or "both" (default: "both")
    - k_examples: Number of example results to return (default: 3)
    - k_extra_prompts: Number of extra prompt results to return (default: 2)
    - example_id: Optional specific example ID to filter by (to check distance between question and that example)
    - extra_prompts_id: Optional specific extra prompt ID to filter by (to check distance between question and that prompt)
    
    Returns:
    - status: "success" or "error"
    - question: The search question
    - search_type: The type of search performed
    - examples: List of example search results (if search_type includes "examples")
    - extra_prompts: List of extra prompt search results (if search_type includes "extra_prompts")
    - total_results: Total number of results found
    
    When example_id or extra_prompts_id is provided, the search will filter by that specific ID
    and return the distance between your question and that specific record's embedding.
    """
    # Get pre-initialized vector store from app state
    vector_store = request.app.state.vector_store
    
    return process_vector_search(
        payload=payload,
        vector_store=vector_store
    )

