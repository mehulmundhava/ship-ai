-- Migration: Add MiniLM vector index for cache entries
-- Date: 2026-01-30
-- Description: When using EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2,
--              cache lookups use minilm_embedding. This index speeds up vector search
--              for cache entries (entry_type = 'cache').
-- Run after: add_vector_cache_columns.sql (adds entry_type, is_active).

-- Ensure minilm_embedding column exists (base schema may already have it)
ALTER TABLE ai_vector_examples
ADD COLUMN IF NOT EXISTS minilm_embedding VECTOR(384);

-- Index for cache vector search when using MiniLM (only if cache columns exist)
-- Skip if entry_type/is_active not present (e.g. add_vector_cache_columns.sql not run)
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'ai_vector_examples' AND column_name = 'entry_type'
  ) THEN
    CREATE INDEX IF NOT EXISTS idx_cache_vector_search_minilm
    ON ai_vector_examples
    USING ivfflat (minilm_embedding vector_l2_ops)
    WHERE entry_type = 'cache' AND is_active = true;
  END IF;
END $$;
