-- Migration: Create ai_message_history table
-- Date: 2026-01-28
-- Description: Stores every chat request/response for debugging and customer support.
--              Use sync_engine_update (UPDATE_USER) to insert; table lives in same DB as app.

-- Create table
CREATE TABLE IF NOT EXISTS ai_message_history (
    id                  BIGSERIAL PRIMARY KEY,
    user_id             VARCHAR(255),
    login_id             VARCHAR(255),
    token_id            VARCHAR(255) NOT NULL,
    question            TEXT NOT NULL,
    response            TEXT NOT NULL,
    sql_query           TEXT,
    cached              BOOLEAN DEFAULT false,
    similarity          FLOAT,
    llm_used            BOOLEAN DEFAULT true,
    llm_type            VARCHAR(100),
    question_type       VARCHAR(50),
    debug_info          JSONB,
    result_data         JSONB,
    error_message       TEXT,
    chat_history_length INTEGER,
    steps_time          JSONB,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT (NOW() AT TIME ZONE 'utc')
);

-- Indexes for common debugging queries
CREATE INDEX IF NOT EXISTS idx_ai_message_history_user_id    ON ai_message_history (user_id);
CREATE INDEX IF NOT EXISTS idx_ai_message_history_login_id   ON ai_message_history (login_id);
CREATE INDEX IF NOT EXISTS idx_ai_message_history_token_id   ON ai_message_history (token_id);
CREATE INDEX IF NOT EXISTS idx_ai_message_history_created_at ON ai_message_history (created_at DESC);

-- Comments for documentation
COMMENT ON TABLE ai_message_history IS 'Stores every chat Q&A for debugging and customer support';
COMMENT ON COLUMN ai_message_history.user_id       IS 'User identifier from request';
COMMENT ON COLUMN ai_message_history.login_id      IS 'Login/session identifier for support';
COMMENT ON COLUMN ai_message_history.token_id      IS 'Session token from request';
COMMENT ON COLUMN ai_message_history.question      IS 'User question';
COMMENT ON COLUMN ai_message_history.response      IS 'AI answer returned to user';
COMMENT ON COLUMN ai_message_history.sql_query    IS 'Generated SQL (if any)';
COMMENT ON COLUMN ai_message_history.cached       IS 'True if answer came from 80% match/cache';
COMMENT ON COLUMN ai_message_history.similarity   IS 'Similarity score when cached';
COMMENT ON COLUMN ai_message_history.llm_used     IS 'True if LLM was used';
COMMENT ON COLUMN ai_message_history.llm_type     IS 'e.g. OPENAI/gpt-4o';
COMMENT ON COLUMN ai_message_history.question_type IS 'journey or non_journey';
COMMENT ON COLUMN ai_message_history.debug_info   IS 'Full debug dict for troubleshooting';
COMMENT ON COLUMN ai_message_history.result_data  IS 'Query result summary (JSON)';
COMMENT ON COLUMN ai_message_history.error_message IS 'Error text if request failed';
COMMENT ON COLUMN ai_message_history.chat_history_length IS 'Length of chat_history in request';
COMMENT ON COLUMN ai_message_history.steps_time   IS 'Exact time taken per step (ms) for all paths: match_80, llm, stage_breakdown';
COMMENT ON COLUMN ai_message_history.created_at   IS 'UTC timestamp of the message';
