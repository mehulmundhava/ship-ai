"""
Vector Search Controller

Handles vector search endpoint logic for searching examples and extra prompts.
"""

from fastapi import HTTPException, status
from app.models.schemas import VectorSearchRequest, VectorSearchResponse, VectorSearchResult
from langchain.schema import Document


def process_vector_search(
    payload: VectorSearchRequest,
    vector_store
) -> VectorSearchResponse:
    """
    Process vector search request and return results.
    
    Args:
        payload: Vector search request payload
        vector_store: Vector store service instance
        
    Returns:
        VectorSearchResponse with search results
    """
    try:
        print(f"\n{'='*80}")
        print(f"🔍 VECTOR SEARCH API REQUEST")
        print(f"{'='*80}")
        print(f"   Question: {payload.question}")
        print(f"   Search Type: {payload.search_type}")
        print(f"   K Examples: {payload.k_examples}")
        print(f"   K Extra Prompts: {payload.k_extra_prompts}")
        if payload.example_id:
            print(f"   Example ID Filter: {payload.example_id}")
        if payload.extra_prompts_id:
            print(f"   Extra Prompts ID Filter: {payload.extra_prompts_id}")
        print(f"{'='*80}\n")
        
        examples_results = []
        extra_prompts_results = []
        
        # Search examples if requested
        if payload.search_type in ["examples", "both"]:
            try:
                example_docs = vector_store.search_examples(
                    payload.question,
                    k=payload.k_examples,
                    example_id=payload.example_id
                )
                
                # Convert Document objects to VectorSearchResult
                for doc in example_docs:
                    examples_results.append(VectorSearchResult(
                        id=doc.metadata.get("id") if doc.metadata else None,
                        content=doc.page_content,
                        distance=doc.metadata.get("distance") if doc.metadata else None,
                        metadata=doc.metadata if doc.metadata else None
                    ))
                
                print(f"✅ Found {len(examples_results)} example results")
            except Exception as e:
                print(f"⚠️  Error searching examples: {e}")
                import traceback
                traceback.print_exc()
                # Continue with other searches even if examples fail
        
        # Search extra prompts if requested
        if payload.search_type in ["extra_prompts", "both"]:
            try:
                extra_prompt_docs = vector_store.search_extra_prompts(
                    payload.question,
                    k=payload.k_extra_prompts,
                    extra_prompts_id=payload.extra_prompts_id
                )
                
                # Convert Document objects to VectorSearchResult
                for doc in extra_prompt_docs:
                    extra_prompts_results.append(VectorSearchResult(
                        id=doc.metadata.get("id") if doc.metadata else None,
                        content=doc.page_content,
                        distance=doc.metadata.get("distance") if doc.metadata else None,
                        metadata=doc.metadata if doc.metadata else None
                    ))
                
                print(f"✅ Found {len(extra_prompts_results)} extra prompt results")
            except Exception as e:
                print(f"⚠️  Error searching extra prompts: {e}")
                import traceback
                traceback.print_exc()
                # Continue even if extra prompts fail
        
        # Calculate total results
        total_results = len(examples_results) + len(extra_prompts_results)
        
        # Prepare response
        response_data = {
            "status": "success",
            "question": payload.question,
            "search_type": payload.search_type,
            "total_results": total_results
        }
        
        # Add results based on search type
        if payload.search_type in ["examples", "both"]:
            response_data["examples"] = examples_results if examples_results else []
        
        if payload.search_type in ["extra_prompts", "both"]:
            response_data["extra_prompts"] = extra_prompts_results if extra_prompts_results else []
        
        print(f"✅ Vector search completed: {total_results} total results")
        print(f"{'='*80}\n")
        
        return VectorSearchResponse(**response_data)
        
    except Exception as e:
        print(f"❌ Error processing vector search: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing vector search: {str(e)}"
        )

