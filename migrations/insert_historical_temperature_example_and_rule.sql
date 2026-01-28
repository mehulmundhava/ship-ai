-- Add one example and one extra prompt so "What was the maximum temperature on date X for device Y?"
-- uses incoming_message_history_k (or incoming_message_table), not device_current_data.
--
-- After running this, regenerate embeddings so the new rows are searchable:
--   POST /generate-embeddings-examples   (processes rows with NULL embedding)
--   POST /generate-embeddings-extra-prompts
--
-- No changes are needed to your existing 7 temperature examples.

-- 1) New example: historical max temperature on a specific date (uses incoming_message_history_k)
INSERT INTO ai_vector_examples (question, sql_query, description, metadata)
SELECT
  'What was the maximum temperature on 24 January 2026 for device id WT01EB25EC389682?',
  'SELECT MAX(ik.temperature) AS max_temperature
FROM incoming_message_history_k ik
JOIN user_device_assignment ud ON ik.device_id = ud.device
WHERE ud.user_id = 27
  AND ik.device_id = ''WT01EB25EC389682''
  AND ik.event_time::date = ''2026-01-24'';',
  'Historical temperature by date: use incoming_message_history_k (or incoming_message_table). Filter by event_time::date for the requested date. device_current_data is for current snapshot only.',
  '{"keywords": ["maximum temperature", "historical", "specific date", "on date", "incoming_message_history_k", "incoming_message_table", "temperature by date", "device"], "type": "historical_temperature", "complexity": "medium"}'
WHERE NOT EXISTS (
  SELECT 1 FROM ai_vector_examples
  WHERE question = 'What was the maximum temperature on 24 January 2026 for device id WT01EB25EC389682?'
);

-- 2) New extra prompt: rule that steers "max/min temp on date X" to the history table
INSERT INTO ai_vector_extra_prompts (content, note_type, metadata)
SELECT
  'Temperature by date: For ''what was the maximum/minimum temperature on a specific date'' or any historical temperature-by-date question, use incoming_message_history_k (IK) or incoming_message_table—the table that stores time-series messages per device with temperature and a timestamp (e.g. event_time). device_current_data (CD) holds only the latest snapshot per device; use CD only for ''current'' or ''latest'' temperature, not for historical-by-date queries.',
  'schema_info',
  '{"keywords": ["historical", "maximum temperature", "minimum temperature", "specific date", "on date", "incoming_message_history_k", "incoming_message_table", "device_current_data", "current vs historical", "temperature by date"], "type": "schema_info"}'
WHERE NOT EXISTS (
  SELECT 1 FROM ai_vector_extra_prompts
  WHERE content LIKE 'Temperature by date: For %incoming_message_history_k%'
);
