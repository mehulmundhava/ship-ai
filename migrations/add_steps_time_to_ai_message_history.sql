-- Migration: Add steps_time column to ai_message_history
-- Date: 2026-01-29
-- Description: Stores exact time taken per step for all question paths
--              (80% match hit, 80% match miss + LLM). JSONB for flexible structure.

ALTER TABLE ai_message_history
ADD COLUMN IF NOT EXISTS steps_time JSONB NULL;

COMMENT ON COLUMN ai_message_history.steps_time IS 'Exact time taken per step (ms): path, total_ms, match_80_ms, agent_ms, stage_breakdown for LLM path';
