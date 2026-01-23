-- ============================================================================
-- FIX EXISTING CACHE ENTRIES: Extract and store parameters in metadata
-- ============================================================================

-- 1. Update cache entry id=67 to have params extracted from its SQL
UPDATE ai_vector_examples
SET metadata = jsonb_set(
    COALESCE(metadata, '{}'::jsonb),
    '{params}',
    '{"device_id": ["WT01D5D2667C982E"]}'::jsonb
)
WHERE id = 67
  AND entry_type = 'cache'
  AND (metadata->>'params' IS NULL OR metadata->>'params' = 'null');

-- 2. Verify the update
SELECT 
    id,
    entry_type,
    question,
    sql_query,
    metadata->>'params' as params,
    metadata
FROM ai_vector_examples
WHERE id = 67;

-- 3. For any other cache entries missing params, you can manually update them
-- by extracting device_id, facility_id, etc. from their sql_query
