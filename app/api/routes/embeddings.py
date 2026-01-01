"""
Embeddings Routes
"""

from fastapi import APIRouter, Request
from app.controllers.embeddings_controller import (
    reload_vector_store,
    generate_embeddings_examples,
    generate_embeddings_extra_prompts
)
from app.models.schemas import (
    GenerateEmbeddingsRequest,
    GenerateEmbeddingsResponse,
    ReloadVectorStoreResponse
)

router = APIRouter()


@router.post("/reload-vector-store", response_model=ReloadVectorStoreResponse)
def reload_vector_store_route(request: Request):
    """
    Reload Vector Store Endpoint
    
    Verifies PostgreSQL vector store tables and returns record counts.
    """
    vector_store = request.app.state.vector_store
    return reload_vector_store(vector_store)


@router.post("/generate-embeddings-examples", response_model=GenerateEmbeddingsResponse)
def generate_embeddings_examples_route(request: Request, payload: GenerateEmbeddingsRequest):
    """
    Generate Embeddings for Examples Table
    
    Generates embeddings for records in ai_vector_examples table.
    """
    vector_store = request.app.state.vector_store
    return generate_embeddings_examples(payload, vector_store)


@router.post("/generate-embeddings-extra-prompts", response_model=GenerateEmbeddingsResponse)
def generate_embeddings_extra_prompts_route(request: Request, payload: GenerateEmbeddingsRequest):
    """
    Generate Embeddings for Extra Prompts Table
    
    Generates embeddings for records in ai_vector_extra_prompts table.
    """
    vector_store = request.app.state.vector_store
    return generate_embeddings_extra_prompts(payload, vector_store)

