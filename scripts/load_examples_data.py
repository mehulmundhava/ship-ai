"""
Script to Load Example Data from examples_data.py into PostgreSQL Tables

This script loads data from examples_data.py into:
- ai_vector_examples table (for SAMPLE_EXAMPLES)
- ai_vector_extra_prompts table (for EXTRA_PROMPT_DATA)

Note: This script does NOT generate embeddings. Use the API endpoints
to generate embeddings after loading the data.
"""

import json
from sqlalchemy import text
from app.config.database import sync_engine_update
from scripts.examples_data import SAMPLE_EXAMPLES, EXTRA_PROMPT_DATA


def load_examples_data():
    """Load SAMPLE_EXAMPLES into ai_vector_examples table."""
    print(f"\n{'='*80}")
    print(f"📥 LOADING EXAMPLES DATA INTO ai_vector_examples")
    print(f"{'='*80}")
    
    try:
        # Use update engine for INSERT/UPDATE operations
        with sync_engine_update.connect() as conn:
            inserted_count = 0
            updated_count = 0
            
            for example in SAMPLE_EXAMPLES:
                question = example.get('question', '')
                sql_query = example.get('sql', '')
                description = example.get('description', None)  # Optional description field
                metadata = example.get('metadata', {})
                
                # Convert metadata dict to JSON string
                metadata_json = json.dumps(metadata) if metadata else '{}'
                
                # Check if this example already exists (by question)
                check_query = text("""
                    SELECT id FROM ai_vector_examples 
                    WHERE question = :question
                """)
                result = conn.execute(check_query, {"question": question})
                existing = result.fetchone()
                
                if existing:
                    # Update existing record
                    update_query = text("""
                        UPDATE ai_vector_examples
                        SET 
                            sql_query = :sql_query,
                            description = :description,
                            metadata = CAST(:metadata AS jsonb)
                        WHERE id = :id
                    """)
                    conn.execute(update_query, {
                        "id": existing[0],
                        "sql_query": sql_query,
                        "description": description,
                        "metadata": metadata_json
                    })
                    conn.commit()
                    updated_count += 1
                    print(f"   ✓ Updated: {question[:60]}...")
                else:
                    # Insert new record
                    insert_query = text("""
                        INSERT INTO ai_vector_examples (question, sql_query, description, metadata)
                        VALUES (:question, :sql_query, :description, CAST(:metadata AS jsonb))
                    """)
                    conn.execute(insert_query, {
                        "question": question,
                        "sql_query": sql_query,
                        "description": description,
                        "metadata": metadata_json
                    })
                    conn.commit()
                    inserted_count += 1
                    print(f"   ✓ Inserted: {question[:60]}...")
            
            print(f"\n✅ Examples data loaded successfully!")
            print(f"   Inserted: {inserted_count} records")
            print(f"   Updated: {updated_count} records")
            print(f"{'='*80}\n")
            
            return {
                "status": "success",
                "inserted": inserted_count,
                "updated": updated_count
            }
            
    except Exception as e:
        error_msg = f"Error loading examples data: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"{'='*80}\n")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": error_msg
        }


def load_extra_prompts_data():
    """Load EXTRA_PROMPT_DATA into ai_vector_extra_prompts table."""
    print(f"\n{'='*80}")
    print(f"📥 LOADING EXTRA PROMPTS DATA INTO ai_vector_extra_prompts")
    print(f"{'='*80}")
    
    try:
        # Use update engine for INSERT/UPDATE operations
        with sync_engine_update.connect() as conn:
            inserted_count = 0
            updated_count = 0
            
            for item in EXTRA_PROMPT_DATA:
                content = item.get('content', '')
                metadata = item.get('metadata', {})
                note_type = metadata.get('type', None) if metadata else None
                
                # Convert metadata dict to JSON string
                metadata_json = json.dumps(metadata) if metadata else '{}'
                
                # Check if this content already exists
                check_query = text("""
                    SELECT id FROM ai_vector_extra_prompts 
                    WHERE content = :content
                """)
                result = conn.execute(check_query, {"content": content})
                existing = result.fetchone()
                
                if existing:
                    # Update existing record
                    update_query = text("""
                        UPDATE ai_vector_extra_prompts
                        SET 
                            note_type = :note_type,
                            metadata = CAST(:metadata AS jsonb)
                        WHERE id = :id
                    """)
                    conn.execute(update_query, {
                        "id": existing[0],
                        "note_type": note_type,
                        "metadata": metadata_json
                    })
                    conn.commit()
                    updated_count += 1
                    print(f"   ✓ Updated: {content[:60]}...")
                else:
                    # Insert new record
                    insert_query = text("""
                        INSERT INTO ai_vector_extra_prompts (content, note_type, metadata)
                        VALUES (:content, :note_type, CAST(:metadata AS jsonb))
                    """)
                    conn.execute(insert_query, {
                        "content": content,
                        "note_type": note_type,
                        "metadata": metadata_json
                    })
                    conn.commit()
                    inserted_count += 1
                    print(f"   ✓ Inserted: {content[:60]}...")
            
            print(f"\n✅ Extra prompts data loaded successfully!")
            print(f"   Inserted: {inserted_count} records")
            print(f"   Updated: {updated_count} records")
            print(f"{'='*80}\n")
            
            return {
                "status": "success",
                "inserted": inserted_count,
                "updated": updated_count
            }
            
    except Exception as e:
        error_msg = f"Error loading extra prompts data: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"{'='*80}\n")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": error_msg
        }


def main():
    """Main function to load all data."""
    print(f"\n{'#'*80}")
    print(f"🚀 LOADING EXAMPLE DATA INTO POSTGRESQL")
    print(f"{'#'*80}\n")
    
    # Load examples data
    examples_result = load_examples_data()
    
    # Load extra prompts data
    extra_prompts_result = load_extra_prompts_data()
    
    # Summary
    print(f"\n{'#'*80}")
    print(f"📊 LOADING SUMMARY")
    print(f"{'#'*80}")
    print(f"Examples Table:")
    print(f"   Status: {examples_result.get('status', 'unknown')}")
    print(f"   Inserted: {examples_result.get('inserted', 0)}")
    print(f"   Updated: {examples_result.get('updated', 0)}")
    print(f"\nExtra Prompts Table:")
    print(f"   Status: {extra_prompts_result.get('status', 'unknown')}")
    print(f"   Inserted: {extra_prompts_result.get('inserted', 0)}")
    print(f"   Updated: {extra_prompts_result.get('updated', 0)}")
    print(f"\n💡 Next Step: Use API endpoints to generate embeddings:")
    print(f"   POST /generate-embeddings-examples")
    print(f"   POST /generate-embeddings-extra-prompts")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()

