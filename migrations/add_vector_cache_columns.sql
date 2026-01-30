-- Migration: Add Vector Cache Columns to ai_vector_examples
-- Date: 2026-01-23
-- Description: Extends ai_vector_examples table to support vector-based caching
--              without modifying existing data or breaking existing logic.

-- Add new columns for cache functionality
ALTER TABLE ai_vector_examples
ADD COLUMN IF NOT EXISTS entry_type VARCHAR(20) DEFAULT 'example',
ADD COLUMN IF NOT EXISTS answer TEXT,
ADD COLUMN IF NOT EXISTS question_type VARCHAR(20),
ADD COLUMN IF NOT EXISTS user_type VARCHAR(20),
ADD COLUMN IF NOT EXISTS similarity_threshold FLOAT DEFAULT 0.80,
ADD COLUMN IF NOT EXISTS usage_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS last_used TIMESTAMP,
ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT true,
ADD COLUMN IF NOT EXISTS is_deterministic BOOLEAN DEFAULT false;

-- Set default entry_type for existing rows (they remain as 'example')
-- This ensures backward compatibility
UPDATE ai_vector_examples
SET entry_type = 'example'
WHERE entry_type IS NULL;

-- Create index for cache lookups (only on cache entries)
-- Using ivfflat for faster similarity search on cache entries
CREATE INDEX IF NOT EXISTS idx_cache_vector_search 
ON ai_vector_examples 
USING ivfflat (bge_large_embedding vector_cosine_ops)
WHERE entry_type = 'cache' AND is_active = true;

-- Add comment to table explaining the new columns
COMMENT ON COLUMN ai_vector_examples.entry_type IS 'Type of entry: ''example'' (existing few-shot examples) or ''cache'' (cached answers)';
COMMENT ON COLUMN ai_vector_examples.answer IS 'Final answer returned to user (only for cache entries)';
COMMENT ON COLUMN ai_vector_examples.question_type IS 'Type of question: ''journey'' or ''non_journey'' (required for cache entries)';
COMMENT ON COLUMN ai_vector_examples.user_type IS 'User type: ''admin'' or ''user'' (required for cache entries)';
COMMENT ON COLUMN ai_vector_examples.similarity_threshold IS 'Similarity threshold for cache matching (default 0.80)';
COMMENT ON COLUMN ai_vector_examples.usage_count IS 'Number of times this cache entry has been used';
COMMENT ON COLUMN ai_vector_examples.last_used IS 'Timestamp of last cache hit';
COMMENT ON COLUMN ai_vector_examples.is_active IS 'Whether this cache entry is active (can be disabled without deleting)';
COMMENT ON COLUMN ai_vector_examples.is_deterministic IS 'Whether the answer is deterministic (factual/SQL-derived)';
