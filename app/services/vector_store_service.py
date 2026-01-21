"""
PostgreSQL Vector Store Service

This module handles PostgreSQL vector store initialization and management.
It uses pgvector extension for semantic search on example queries and extra prompt data.
Uses Hugging Face embeddings instead of OpenAI for cost efficiency.
"""

import json
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from sqlalchemy import text
from app.config.database import sync_engine
from app.config.settings import settings


class VectorStoreService:
    """
    Service for managing PostgreSQL vector stores for examples and extra prompt data.
    Uses Hugging Face embeddings and pgvector for semantic search.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize embeddings model using Hugging Face.
        
        Args:
            model_name: Hugging Face model name for embeddings.
                       Default: from settings
        """
        model_name = model_name or settings.embedding_model_name
        print(f"🔧 Initializing Hugging Face embeddings with model: {model_name}")
        
        # Initialize Hugging Face embeddings
        # This will download the model on first use if not cached
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # Use CPU by default
            encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity
        )
        
        self.model_name = model_name
        self.engine = sync_engine
        self.embedding_field_name = settings.get_embedding_field_name()
        print(f"🔧 Using embedding field: {self.embedding_field_name}")
    
    def initialize_stores(self):
        """
        Initialize PostgreSQL vector stores.
        Verifies that tables exist and are accessible.
        Note: Data should already be loaded into PostgreSQL tables.
        """
        print(f"🔍 Verifying PostgreSQL vector store tables...")
        
        try:
            with self.engine.connect() as conn:
                # Check if examples table exists and has data
                result = conn.execute(text("""
                    SELECT COUNT(*) as count 
                    FROM ai_vector_examples
                """))
                examples_count = result.fetchone()[0]
                print(f"✅ Examples table accessible: {examples_count} records")
                
                # Check if extra prompts table exists and has data
                result = conn.execute(text("""
                    SELECT COUNT(*) as count 
                    FROM ai_vector_extra_prompts
                """))
                extra_prompts_count = result.fetchone()[0]
                print(f"✅ Extra prompts table accessible: {extra_prompts_count} records")
                
                print(f"✅ PostgreSQL vector stores initialized successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not verify vector store tables: {e}")
            print("   Make sure the tables exist and data is loaded into them.")
            raise
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding vector for a query string.
        
        Args:
            query: Query string to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        embedding = self.embeddings.embed_query(query)
        return embedding
    
    def _embed_query(self, query: str) -> List[float]:
        """
        Alias for embed_query (kept for backward compatibility).
        """
        return self.embed_query(query)
    
    def search_examples(self, query: str, k: int = 3, example_id: Optional[int] = None) -> List[Document]:
        """
        Search for similar examples in the PostgreSQL vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            example_id: Optional specific example ID to filter by (to check distance)
            
        Returns:
            List of similar example documents
        """
        print(f"\n{'='*80}")
        print(f"🔍 VECTOR STORE SEARCH - Examples")
        print(f"{'='*80}")
        print(f"   Query: {query}")
        print(f"   K: {k}")
        if example_id:
            print(f"   Filtering by ID: {example_id}")
        
        try:
            # Generate embedding for the query
            query_embedding = self.embed_query(query)
            
            # Convert to PostgreSQL array format string for vector type
            # pgvector expects the format: '[1,2,3]'::vector
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Build WHERE conditions
            where_conditions = [f"{self.embedding_field_name} IS NOT NULL"]
            query_params = {"k": k}
            
            # Add ID filter if provided
            if example_id is not None:
                where_conditions.append("id = :example_id")
                query_params["example_id"] = example_id
            
            where_clause = " AND ".join(where_conditions)
            
            # Search using pgvector L2 distance (vector_l2_ops)
            # Using ORDER BY ... LIMIT for efficient similarity search
            # Note: We use string formatting for the vector literal as it's a constant value
            # The k parameter is safe as it's an integer from our code
            search_query = text(f"""
                SELECT 
                    id,
                    question,
                    sql_query,
                    description,
                    metadata,
                    {self.embedding_field_name} <-> '{embedding_str}'::vector AS distance
                FROM ai_vector_examples
                WHERE {where_clause}
                ORDER BY {self.embedding_field_name} <-> '{embedding_str}'::vector
                LIMIT :k
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(search_query, query_params)
                rows = result.fetchall()
            
            # Convert results to Document objects
            documents = []
            for row in rows:
                # Combine question, description (if available), and SQL for the document content
                content_parts = [f"Question: {row.question}"]
                
                # Add description if available
                if row.description:
                    content_parts.append(f"Description: {row.description}")
                
                content_parts.append(f"\nSQL Query:\n{row.sql_query}")
                content = "\n\n".join(content_parts)
                
                # Parse metadata - JSONB columns are typically returned as dicts by psycopg2
                # But handle string case for safety
                if isinstance(row.metadata, dict):
                    metadata = row.metadata.copy()
                elif isinstance(row.metadata, str):
                    try:
                        metadata = json.loads(row.metadata)
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}
                else:
                    metadata = {}
                metadata['distance'] = float(row.distance) if row.distance else None
                metadata['id'] = row.id
                
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)
            
            print(f"   Found {len(documents)} results:")
            for i, doc in enumerate(documents, 1):
                content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                print(f"   [{i}] {content_preview}")
                if doc.metadata:
                    distance = doc.metadata.get('distance')
                    if distance is not None:
                        print(f"       Distance: {distance:.4f}")
            print(f"{'='*80}\n")
            
            return documents
            
        except Exception as e:
            print(f"❌ Error searching examples: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def search_extra_prompts(self, query: str, k: int = 2, extra_prompts_id: Optional[int] = None) -> List[Document]:
        """
        Search for relevant extra prompt data.
        
        Args:
            query: Search query
            k: Number of results to return
            extra_prompts_id: Optional specific extra prompt ID to filter by (to check distance)
            
        Returns:
            List of relevant prompt documents
        """
        print(f"\n{'='*80}")
        print(f"🔍 VECTOR STORE SEARCH - Extra Prompts")
        print(f"{'='*80}")
        print(f"   Query: {query}")
        print(f"   K: {k}")
        if extra_prompts_id:
            print(f"   Filtering by ID: {extra_prompts_id}")
        
        try:
            # Generate embedding for the query
            query_embedding = self.embed_query(query)
            
            # Convert to PostgreSQL array format string for vector type
            # pgvector expects the format: '[1,2,3]'::vector
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Build WHERE conditions
            where_conditions = [f"{self.embedding_field_name} IS NOT NULL"]
            query_params = {"k": k}
            
            # Add ID filter if provided
            if extra_prompts_id is not None:
                where_conditions.append("id = :extra_prompts_id")
                query_params["extra_prompts_id"] = extra_prompts_id
            
            where_clause = " AND ".join(where_conditions)
            
            # Search using pgvector L2 distance
            # Note: We use string formatting for the vector literal as it's a constant value
            # The k parameter is safe as it's an integer from our code
            search_query = text(f"""
                SELECT 
                    id,
                    content,
                    note_type,
                    metadata,
                    {self.embedding_field_name} <-> '{embedding_str}'::vector AS distance
                FROM ai_vector_extra_prompts
                WHERE {where_clause}
                ORDER BY {self.embedding_field_name} <-> '{embedding_str}'::vector
                LIMIT :k
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(search_query, query_params)
                rows = result.fetchall()
            
            # Convert results to Document objects
            documents = []
            for row in rows:
                # Parse metadata - JSONB columns are typically returned as dicts by psycopg2
                # But handle string case for safety
                if isinstance(row.metadata, dict):
                    metadata = row.metadata.copy()
                elif isinstance(row.metadata, str):
                    try:
                        metadata = json.loads(row.metadata)
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}
                else:
                    metadata = {}
                if row.note_type:
                    metadata['note_type'] = row.note_type
                metadata['distance'] = float(row.distance) if row.distance else None
                metadata['id'] = row.id
                
                doc = Document(
                    page_content=row.content,
                    metadata=metadata
                )
                documents.append(doc)
            
            print(f"   Found {len(documents)} results:")
            for i, doc in enumerate(documents, 1):
                content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                print(f"   [{i}] {content_preview}")
                if doc.metadata:
                    distance = doc.metadata.get('distance')
                    if distance is not None:
                        print(f"       Distance: {distance:.4f}")
            print(f"{'='*80}\n")
            
            return documents
            
        except Exception as e:
            print(f"❌ Error searching extra prompts: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def reload_stores(self):
        """
        Reload vector stores from PostgreSQL.
        Note: This method is kept for API compatibility, but since data is
        stored in PostgreSQL, it just verifies the tables are accessible.
        
        Returns:
            dict: Status of the reload operation
        """
        print(f"\n{'='*80}")
        print(f"🔄 RELOADING VECTOR STORES")
        print(f"{'='*80}")
        
        try:
            with self.engine.connect() as conn:
                # Count examples
                result = conn.execute(text("SELECT COUNT(*) FROM ai_vector_examples"))
                examples_count = result.fetchone()[0]
                
                # Count extra prompts
                result = conn.execute(text("SELECT COUNT(*) FROM ai_vector_extra_prompts"))
                extra_prompts_count = result.fetchone()[0]
            
            print(f"✅ Vector stores verified:")
            print(f"   Examples: {examples_count} records")
            print(f"   Extra Prompts: {extra_prompts_count} records")
            print(f"{'='*80}")
            print(f"✅ Vector stores reloaded successfully!")
            print(f"{'='*80}\n")
            
            return {
                "status": "success",
                "message": "Vector stores verified successfully",
                "examples_count": examples_count,
                "extra_prompts_count": extra_prompts_count
            }
        except Exception as e:
            error_msg = f"Error reloading vector stores: {str(e)}"
            print(f"❌ {error_msg}")
            print(f"{'='*80}\n")
            return {
                "status": "error",
                "message": error_msg
            }

