"""
Message History Service

Saves every chat request/response to ai_message_history for debugging and support.
Uses sync_engine_update (write connection). Failures are logged and do not affect the chat response.
"""

import json
import logging
from typing import Any, Dict, Optional

from sqlalchemy import text

from app.config.database import sync_engine_update

logger = logging.getLogger("ship_rag_ai")

# Max size for result_data JSON (chars) to avoid huge rows; store truncated if larger
RESULT_DATA_MAX_CHARS = 100_000


def _truncate_result_data(data: Any) -> Optional[Dict[str, Any]]:
    """Convert result_data to a JSON-serializable dict and truncate if too large."""
    if data is None:
        return None
    try:
        if isinstance(data, (dict, list)):
            payload = data
        else:
            payload = {"raw_result": str(data)}
        s = json.dumps(payload)
        if len(s) > RESULT_DATA_MAX_CHARS:
            return {"_truncated": True, "_length": len(s), "preview": s[:2000]}
        return payload
    except (TypeError, ValueError) as e:
        logger.debug("message_history: could not serialize result_data: %s", e)
        return {"_error": "serialization_failed", "repr": repr(data)[:500]}


def save_message_to_history(
    *,
    user_id: Optional[str],
    login_id: Optional[str],
    token_id: str,
    question: str,
    response: str,
    sql_query: Optional[str] = None,
    cached: bool = False,
    similarity: Optional[float] = None,
    llm_used: bool = True,
    llm_type: Optional[str] = None,
    question_type: Optional[str] = None,
    debug_info: Optional[Dict[str, Any]] = None,
    result_data: Any = None,
    error_message: Optional[str] = None,
    chat_history_length: Optional[int] = None,
) -> None:
    """
    Insert one row into ai_message_history. Does not raise; logs on failure.

    Call this after every chat response (80% match path and LLM path).
    """
    try:
        result_json = _truncate_result_data(result_data)
        debug_json = None
        if debug_info is not None:
            try:
                debug_json = json.loads(json.dumps(debug_info))
            except (TypeError, ValueError):
                debug_json = {"_error": "serialization_failed"}

        with sync_engine_update.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO ai_message_history (
                        user_id, login_id, token_id, question, response,
                        sql_query, cached, similarity, llm_used, llm_type,
                        question_type, debug_info, result_data, error_message,
                        chat_history_length
                    ) VALUES (
                        :user_id, :login_id, :token_id, :question, :response,
                        :sql_query, :cached, :similarity, :llm_used, :llm_type,
                        :question_type, :debug_info, :result_data, :error_message,
                        :chat_history_length
                    )
                """),
                {
                    "user_id": user_id,
                    "login_id": login_id,
                    "token_id": token_id,
                    "question": question,
                    "response": response,
                    "sql_query": sql_query,
                    "cached": cached,
                    "similarity": similarity,
                    "llm_used": llm_used,
                    "llm_type": llm_type,
                    "question_type": question_type,
                    "debug_info": json.dumps(debug_json) if debug_json is not None else None,
                    "result_data": json.dumps(result_json) if result_json is not None else None,
                    "error_message": error_message,
                    "chat_history_length": chat_history_length,
                },
            )
            conn.commit()
        logger.debug("message_history: saved row for question_id=%s", token_id)
    except Exception as e:
        logger.warning("message_history: failed to save (chat unaffected): %s", e, exc_info=True)
