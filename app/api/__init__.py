"""
API Package - Router Aggregation
"""

from fastapi import APIRouter
from app.api.routes import health, chat, embeddings, vector_search

# Create main API router
api_router = APIRouter()

# Include all route modules
api_router.include_router(health.router, tags=["health"])
api_router.include_router(chat.router, tags=["chat"])
api_router.include_router(embeddings.router, tags=["embeddings"])
api_router.include_router(vector_search.router, tags=["vector-search"])
