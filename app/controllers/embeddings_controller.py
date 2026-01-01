"""
Embeddings Controller

Handles embedding generation endpoints for vector store tables.
Uses update database connection for write operations.
"""

from sqlalchemy import text
from app.config.database import sync_engine_update
from app.models.schemas import (
    GenerateEmbeddingsRequest,
    GenerateEmbeddingsResponse,
    ReloadVectorStoreResponse
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
        
        # Use update engine for write operations
        with sync_engine_update.connect() as conn:
            if payload.id:
                # Process specific ID
                query = text("""
                    SELECT id, question, sql_query
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
                query = text("""
                    SELECT id, question, sql_query
                    FROM ai_vector_examples
                    WHERE minilm_embedding IS NULL
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
                    question = row.question
                    sql_query = row.sql_query
                    
                    # Combine question and SQL for embedding (same format as search)
                    content = f"Question: {question}\n\nSQL Query:\n{sql_query}"
                    
                    # Generate embedding
                    embedding = vector_store.embed_query(content)
                    
                    # Convert to PostgreSQL array format string
                    embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                    
                    # Update the record
                    update_query = text("""
                        UPDATE ai_vector_examples
                        SET minilm_embedding = CAST(:embedding AS vector)
                        WHERE id = :id
                    """)
                    conn.execute(update_query, {
                        "id": record_id,
                        "embedding": embedding_str
                    })
                    conn.commit()
                    
                    processed_count += 1
                    updated_ids.append(record_id)
                    print(f"   ✓ Processed ID {record_id}: {question[:50]}...")
                    
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
        
        # Use update engine for write operations
        with sync_engine_update.connect() as conn:
            if payload.id:
                # Process specific ID
                query = text("""
                    SELECT id, content
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
                query = text("""
                    SELECT id, content
                    FROM ai_vector_extra_prompts
                    WHERE minilm_embedding IS NULL
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
                    content = row.content
                    
                    # Generate embedding directly from content
                    embedding = vector_store.embed_query(content)
                    
                    # Convert to PostgreSQL array format string
                    embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                    
                    # Update the record
                    update_query = text("""
                        UPDATE ai_vector_extra_prompts
                        SET minilm_embedding = CAST(:embedding AS vector)
                        WHERE id = :id
                    """)
                    conn.execute(update_query, {
                        "id": record_id,
                        "embedding": embedding_str
                    })
                    conn.commit()
                    
                    processed_count += 1
                    updated_ids.append(record_id)
                    print(f"   ✓ Processed ID {record_id}: {content[:50]}...")
                    
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

