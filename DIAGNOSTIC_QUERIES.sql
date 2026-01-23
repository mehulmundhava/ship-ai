-- ============================================================================
-- DIAGNOSTIC QUERIES FOR CACHE TROUBLESHOOTING
-- ============================================================================

-- 1. Check what entries exist in ai_vector_examples table
SELECT 
    id,
    entry_type,
    question,
    LEFT(sql_query, 100) as sql_preview,
    question_type,
    user_type,
    is_active,
    answer IS NOT NULL as has_answer,
    metadata->>'params' as cached_params
FROM ai_vector_examples
WHERE question ILIKE '%facility-to-facility movements%'
   OR question ILIKE '%journey timing%'
ORDER BY id DESC
LIMIT 10;

-- 2. Check all cache entries (entry_type = 'cache')
SELECT 
    id,
    question,
    LEFT(sql_query, 100) as sql_preview,
    question_type,
    user_type,
    is_active,
    metadata->>'params' as params,
    usage_count,
    last_used
FROM ai_vector_examples
WHERE entry_type = 'cache'
ORDER BY id DESC
LIMIT 10;

-- 3. Check all example entries that could be used as cache
SELECT 
    id,
    question,
    LEFT(sql_query, 100) as sql_preview,
    question_type,
    user_type,
    is_active,
    answer IS NOT NULL as has_answer,
    metadata
FROM ai_vector_examples
WHERE entry_type = 'example'
  AND (question_type IS NOT NULL OR user_type IS NOT NULL)
ORDER BY id DESC
LIMIT 10;

-- 4. Check specific question about device journeys
SELECT 
    id,
    entry_type,
    question,
    sql_query,
    question_type,
    user_type,
    is_active,
    answer IS NOT NULL as has_answer,
    metadata->>'params' as params
FROM ai_vector_examples
WHERE question ILIKE '%WT01%'
ORDER BY id DESC;

-- 5. Check if embeddings exist for cache entries
SELECT 
    id,
    entry_type,
    question_type,
    user_type,
    bge_large_embedding IS NOT NULL as has_embedding,
    minilm_embedding IS NOT NULL as has_minilm_embedding
FROM ai_vector_examples
WHERE entry_type IN ('cache', 'example')
  AND (question_type = 'journey' OR question_type IS NULL)
ORDER BY id DESC
LIMIT 20;
