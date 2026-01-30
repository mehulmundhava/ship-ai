"""
Embeddings Controller

Handles embedding generation endpoints for vector store tables.
Uses update database connection for write operations.
"""

from sqlalchemy import text
from app.config.database import sync_engine_update, sync_engine
from app.config.settings import settings
from app.models.schemas import (
    GenerateEmbeddingsRequest,
    GenerateEmbeddingsResponse,
    ReloadVectorStoreResponse,
    GetTextEmbeddingRequest,
    GetTextEmbeddingResponse,
    SearchEmbeddingRequest,
    SearchEmbeddingResponse,
    SearchEmbeddingResult
)


def reload_vector_store(vector_store) -> ReloadVectorStoreResponse:
    """
    Reload Vector Store Controller
    
    Verifies PostgreSQL vector store tables and returns record counts.
    
    Args:
        vector_store: VectorStoreService instance
        
    Returns:
        ReloadVectorStoreResponse with status and counts
    """
    print(f"\n{'='*80}")
    print(f"🔄 RELOAD VECTOR STORE REQUEST RECEIVED")
    print(f"{'='*80}\n")
    
    try:
        # Reload stores
        result = vector_store.reload_stores()
        
        # Re-initialize to ensure stores are loaded
        vector_store.initialize_stores()
        
        return ReloadVectorStoreResponse(
            status=result["status"],
            message=result["message"],
            examples_count=result.get("examples_count"),
            extra_prompts_count=result.get("extra_prompts_count")
        )
    except Exception as e:
        error_msg = f"Error reloading vector stores: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return ReloadVectorStoreResponse(
            status="error",
            message=error_msg
        )


def generate_embeddings_examples(
    payload: GenerateEmbeddingsRequest,
    vector_store
) -> GenerateEmbeddingsResponse:
    """
    Generate Embeddings for Examples Table
    
    Generates embeddings for records in ai_vector_examples table.
    
    Args:
        payload: GenerateEmbeddingsRequest with optional id
        vector_store: VectorStoreService instance
        
    Returns:
        GenerateEmbeddingsResponse with status and processed count
    """
    print(f"\n{'='*80}")
    print(f"🔧 GENERATE EMBEDDINGS - Examples Table")
    print(f"{'='*80}")
    print(f"   ID: {payload.id if payload.id else 'All records with NULL embeddings'}")
    print(f"{'='*80}\n")
    
    try:
        processed_count = 0
        updated_ids = []
        errors = []
        
        # Get the embedding field name based on the model
        embedding_field = settings.get_embedding_field_name()
        print(f"   Using embedding field: {embedding_field}")
        
        # Use update engine for write operations
        with sync_engine_update.connect() as conn:
            if payload.id:
                # Process specific ID
                query = text("""
                    SELECT id, question, sql_query, metadata
                    FROM ai_vector_examples
                    WHERE id = :id
                """)
                result = conn.execute(query, {"id": payload.id})
                rows = result.fetchall()
                
                if not rows:
                    return GenerateEmbeddingsResponse(
                        status="error",
                        message=f"Record with id {payload.id} not found",
                        processed_count=0
                    )
            else:
                # Process all records with NULL embeddings
                query = text(f"""
                    SELECT id, question, sql_query, metadata
                    FROM ai_vector_examples
                    WHERE {embedding_field} IS NULL
                """)
                result = conn.execute(query)
                rows = result.fetchall()
            
            if not rows:
                return GenerateEmbeddingsResponse(
                    status="success",
                    message="No records to process",
                    processed_count=0
                )
            
            print(f"   Found {len(rows)} record(s) to process")
            
            # Process each record
            for row in rows:
                try:
                    record_id = row.id
                    
                    # Use question field directly for embedding
                    embedding_text = row.question if row.question else ""
                    
                    if not embedding_text.strip():
                        # Skip if question is empty
                        print(f"   ⚠️  Skipping ID {record_id}: Question is empty")
                        continue
                    
                    # Generate embedding from question
                    embedding = vector_store.embed_query(embedding_text)
                    
                    # Convert to PostgreSQL array format string
                    embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                    
                    # Update the record using dynamic field name
                    update_query = text(f"""
                        UPDATE ai_vector_examples
                        SET {embedding_field} = CAST(:embedding AS vector)
                        WHERE id = :id
                    """)
                    conn.execute(update_query, {
                        "id": record_id,
                        "embedding": embedding_str
                    })
                    conn.commit()
                    
                    processed_count += 1
                    updated_ids.append(record_id)
                    question_preview = embedding_text[:100] + "..." if len(embedding_text) > 100 else embedding_text
                    print(f"   ✓ Processed ID {record_id}: Question: {question_preview}")
                    
                except Exception as e:
                    error_msg = f"Error processing ID {row.id}: {str(e)}"
                    errors.append(error_msg)
                    print(f"   ❌ {error_msg}")
                    import traceback
                    traceback.print_exc()
            
            message = f"Successfully processed {processed_count} record(s)"
            if errors:
                message += f" with {len(errors)} error(s)"
            
            print(f"\n✅ Embedding generation completed!")
            print(f"   Processed: {processed_count} record(s)")
            print(f"{'='*80}\n")
            
            return GenerateEmbeddingsResponse(
                status="success",
                message=message,
                processed_count=processed_count,
                updated_ids=updated_ids if updated_ids else None,
                errors=errors if errors else None
            )
            
    except Exception as e:
        error_msg = f"Error generating embeddings: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return GenerateEmbeddingsResponse(
            status="error",
            message=error_msg,
            processed_count=0
        )


def generate_embeddings_extra_prompts(
    payload: GenerateEmbeddingsRequest,
    vector_store
) -> GenerateEmbeddingsResponse:
    """
    Generate Embeddings for Extra Prompts Table
    
    Generates embeddings for records in ai_vector_extra_prompts table.
    
    Args:
        payload: GenerateEmbeddingsRequest with optional id
        vector_store: VectorStoreService instance
        
    Returns:
        GenerateEmbeddingsResponse with status and processed count
    """
    print(f"\n{'='*80}")
    print(f"🔧 GENERATE EMBEDDINGS - Extra Prompts Table")
    print(f"{'='*80}")
    print(f"   ID: {payload.id if payload.id else 'All records with NULL embeddings'}")
    print(f"{'='*80}\n")
    
    try:
        processed_count = 0
        updated_ids = []
        errors = []
        
        # Get the embedding field name based on the model
        embedding_field = settings.get_embedding_field_name()
        print(f"   Using embedding field: {embedding_field}")
        
        # Use update engine for write operations
        with sync_engine_update.connect() as conn:
            if payload.id:
                # Process specific ID
                query = text("""
                    SELECT id, content, metadata
                    FROM ai_vector_extra_prompts
                    WHERE id = :id
                """)
                result = conn.execute(query, {"id": payload.id})
                rows = result.fetchall()
                
                if not rows:
                    return GenerateEmbeddingsResponse(
                        status="error",
                        message=f"Record with id {payload.id} not found",
                        processed_count=0
                    )
            else:
                # Process all records with NULL embeddings
                query = text(f"""
                    SELECT id, content, metadata
                    FROM ai_vector_extra_prompts
                    WHERE {embedding_field} IS NULL
                """)
                result = conn.execute(query)
                rows = result.fetchall()
            
            if not rows:
                return GenerateEmbeddingsResponse(
                    status="success",
                    message="No records to process",
                    processed_count=0
                )
            
            print(f"   Found {len(rows)} record(s) to process")
            
            # Process each record
            for row in rows:
                try:
                    record_id = row.id
                    
                    # Use content field directly for embedding
                    embedding_text = row.content if row.content else ""
                    
                    if not embedding_text.strip():
                        # Skip if content is empty
                        print(f"   ⚠️  Skipping ID {record_id}: Content is empty")
                        continue
                    
                    # Generate embedding from content
                    embedding = vector_store.embed_query(embedding_text)
                    
                    # Convert to PostgreSQL array format string
                    embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                    
                    # Update the record using dynamic field name
                    update_query = text(f"""
                        UPDATE ai_vector_extra_prompts
                        SET {embedding_field} = CAST(:embedding AS vector)
                        WHERE id = :id
                    """)
                    conn.execute(update_query, {
                        "id": record_id,
                        "embedding": embedding_str
                    })
                    conn.commit()
                    
                    processed_count += 1
                    updated_ids.append(record_id)
                    content_preview = embedding_text[:100] + "..." if len(embedding_text) > 100 else embedding_text
                    print(f"   ✓ Processed ID {record_id}: Content: {content_preview}")
                    
                except Exception as e:
                    error_msg = f"Error processing ID {row.id}: {str(e)}"
                    errors.append(error_msg)
                    print(f"   ❌ {error_msg}")
                    import traceback
                    traceback.print_exc()
            
            message = f"Successfully processed {processed_count} record(s)"
            if errors:
                message += f" with {len(errors)} error(s)"
            
            print(f"\n✅ Embedding generation completed!")
            print(f"   Processed: {processed_count} record(s)")
            print(f"{'='*80}\n")
            
            return GenerateEmbeddingsResponse(
                status="success",
                message=message,
                processed_count=processed_count,
                updated_ids=updated_ids if updated_ids else None,
                errors=errors if errors else None
            )
            
    except Exception as e:
        error_msg = f"Error generating embeddings: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return GenerateEmbeddingsResponse(
            status="error",
            message=error_msg,
            processed_count=0
        )


def get_text_embedding(
    payload: GetTextEmbeddingRequest,
    vector_store
) -> GetTextEmbeddingResponse:
    """
    Get Embedding for Given Text
    
    Generates and returns embedding vector for the provided text.
    
    Args:
        payload: GetTextEmbeddingRequest with text field
        vector_store: VectorStoreService instance
        
    Returns:
        GetTextEmbeddingResponse with status, text, and embedding vector
    """
    print(f"\n{'='*80}")
    print(f"🔧 GET TEXT EMBEDDING")
    print(f"{'='*80}")
    print(f"   Text: {payload.text[:100] + '...' if len(payload.text) > 100 else payload.text}")
    print(f"{'='*80}\n")
    
    try:
        if not payload.text or not payload.text.strip():
            return GetTextEmbeddingResponse(
                status="error",
                text=payload.text,
                embedding=[],
                embedding_dimension=0
            )
        
        # Generate embedding from text
        embedding = vector_store.embed_query(payload.text)
        
        print(f"✅ Embedding generated successfully!")
        print(f"   Dimension: {len(embedding)}")
        print(f"{'='*80}\n")
        
        return GetTextEmbeddingResponse(
            status="success",
            text=payload.text,
            embedding=embedding,
            embedding_dimension=len(embedding)
        )
        
    except Exception as e:
        error_msg = f"Error generating embedding: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return GetTextEmbeddingResponse(
            status="error",
            text=payload.text,
            embedding=[],
            embedding_dimension=0
        )


def search_embedding(
    payload: SearchEmbeddingRequest,
    vector_store
) -> SearchEmbeddingResponse:
    """
    Search Embedding in PostgreSQL Table
    
    Generates embedding for the provided text and searches the ai_vector_examples
    table for similar records using cosine similarity.
    
    Args:
        payload: SearchEmbeddingRequest with text and limit fields
        vector_store: VectorStoreService instance
        
    Returns:
        SearchEmbeddingResponse with status, text, limit, and search results
    """
    print(f"\n{'='*80}")
    print(f"🔍 SEARCH EMBEDDING")
    print(f"{'='*80}")
    print(f"   Text: {payload.text[:100] + '...' if len(payload.text) > 100 else payload.text}")
    print(f"   Limit: {payload.limit}")
    print(f"{'='*80}\n")
    
    try:
        if not payload.text or not payload.text.strip():
            return SearchEmbeddingResponse(
                status="error",
                text=payload.text,
                limit=payload.limit,
                results=[],
                total_results=0
            )
        
        # Generate embedding from text
        query_embedding = vector_store.embed_query(payload.text)
        
        # Convert to PostgreSQL array format string
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # Use bge_large_embedding field for search
        embedding_field = 'bge_large_embedding'
        print(f"   Using embedding field: {embedding_field}")
        
        # Execute the search query using CTE (Common Table Expression)
        # Using cosine similarity: 1 - (distance) where distance is <=> operator
        search_query = text(f"""
            WITH query AS (
                SELECT
                    '{embedding_str}'::vector
                    AS query_embedding
            )
            SELECT
                a.id,
                a.question,
                1 - (a.{embedding_field} <=> q.query_embedding) AS similarity
            FROM public.ai_vector_examples AS a
            CROSS JOIN query AS q
            WHERE a.{embedding_field} IS NOT NULL
            ORDER BY a.{embedding_field} <=> q.query_embedding
            LIMIT :limit
        """)
        
        # Use read-only engine for search operations
        with sync_engine.connect() as conn:
            result = conn.execute(search_query, {"limit": payload.limit})
            rows = result.fetchall()
        
        # Convert results to SearchEmbeddingResult objects
        results = []
        for row in rows:
            results.append(SearchEmbeddingResult(
                id=row.id,
                content=row.question if row.question else "",
                similarity=float(row.similarity) if row.similarity is not None else 0.0
            ))
        
        print(f"✅ Search completed successfully!")
        print(f"   Found {len(results)} result(s)")
        for i, result_item in enumerate(results, 1):
            content_preview = result_item.content[:80] + "..." if len(result_item.content) > 80 else result_item.content
            print(f"   [{i}] ID: {result_item.id}, Similarity: {result_item.similarity:.4f}")
            print(f"       Content: {content_preview}")
        print(f"{'='*80}\n")
        
        return SearchEmbeddingResponse(
            status="success",
            text=payload.text,
            limit=payload.limit,
            results=results,
            total_results=len(results)
        )
        
    except Exception as e:
        error_msg = f"Error searching embedding: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return SearchEmbeddingResponse(
            status="error",
            text=payload.text,
            limit=payload.limit,
            results=[],
            total_results=0
        )

