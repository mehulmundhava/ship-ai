-- ============================================================================
-- BACKFILL: Extract and store parameters for existing cache entries
-- ============================================================================
-- This script extracts parameters from SQL and stores them in metadata
-- Run this to fix cache entries that have params=NULL

-- Function to extract device_id from SQL (simplified version)
-- You can extend this to extract other parameters too

UPDATE ai_vector_examples
SET metadata = jsonb_set(
    COALESCE(metadata, '{}'::jsonb),
    '{params}',
    CASE 
        -- Extract device_id if present in SQL
        WHEN sql_query ~* 'device_id\s*=\s*''([^'']+)''' THEN
            jsonb_build_object('device_id', ARRAY[substring(sql_query from 'device_id\s*=\s*''([^'']+)''')])
        -- Extract facility_id if present
        WHEN sql_query ~* 'facility_id\s*=\s*''([^'']+)''' THEN
            jsonb_build_object('facility_id', ARRAY[substring(sql_query from 'facility_id\s*=\s*''([^'']+)''')])
        -- No params found
        ELSE '{}'::jsonb
    END
)
WHERE entry_type = 'cache'
  AND (metadata->>'params' IS NULL OR metadata->>'params' = 'null' OR metadata->>'params' = '{}')
  AND sql_query IS NOT NULL;

-- Verify the update
SELECT 
    id,
    question,
    LEFT(sql_query, 80) as sql_preview,
    metadata->>'params' as params
FROM ai_vector_examples
WHERE entry_type = 'cache'
ORDER BY id DESC
LIMIT 10;
