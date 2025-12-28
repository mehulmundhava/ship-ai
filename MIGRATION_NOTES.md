# Migration Notes: ship-ai to ship-RAG-ai

## Overview

This document explains the key changes and improvements in the new `ship-RAG-ai` project compared to the original `ship-ai`.

## Key Architectural Changes

### 1. LangGraph Instead of LangChain

**Before (ship-ai):**
- Used LangChain's `create_sql_agent` with fixed tool sequence
- Agent always followed: list_tables → get_schema → validate → execute
- No conditional logic for tool selection

**After (ship-RAG-ai):**
- Uses LangGraph for state-based agent orchestration
- Agent can conditionally decide which tools to call
- More flexible workflow with better control flow

### 2. Vector Store for Examples

**Before (ship-ai):**
- All 10 example queries embedded in prompt (~8,000-10,000 chars)
- Examples sent to LLM on every request
- High token consumption

**After (ship-RAG-ai):**
- Examples stored in FAISS vector store
- Retrieved only when needed via semantic search
- Base prompt reduced from ~20k to ~5k characters
- Significant token savings

### 3. Conditional Tool Calling

**Before (ship-ai):**
- Fixed sequence: always list tables, get schema, validate, execute
- No intelligence about when examples are needed

**After (ship-RAG-ai):**
- LLM decides: retrieve examples first? or generate SQL directly?
- Two tools: `get_few_shot_examples` and `execute_db_query`
- Agent can skip example retrieval for simple questions

## File Structure Changes

### New Files

- `agent_graph.py` - LangGraph state machine implementation
- `agent_tools.py` - Tool definitions (FAISS + PostgreSQL)
- `vector_store.py` - FAISS vector store management
- `examples_data.py` - All examples and business rules (moved from prompt)
- `prompts.py` - Concise system prompts (examples removed)

### Removed/Replaced

- `lc_sql_agent.py` → Replaced by `agent_graph.py`
- `custom_toolkit.py` → Replaced by `agent_tools.py`
- Long prompt in `lc_sql_agent.py` → Split into `prompts.py` + `examples_data.py`

### Unchanged

- `db.py` - PostgreSQL connection (same logic)
- `llm_model.py` - LLM wrapper (simplified)
- `models.py` - Pydantic models (same)
- `main.py` - FastAPI app (updated to use new agent)

## Data Flow Comparison

### Original Flow (ship-ai)
```
User Question
  ↓
FastAPI
  ↓
LCSQLAgent (LangChain)
  ↓
[Fixed Sequence]
  1. LLM call with full prompt (20k chars)
  2. Tool: list_tables
  3. LLM call (22k chars)
  4. Tool: get_schema
  5. LLM call (24k chars)
  6. Tool: validate_query
  7. LLM call (25k chars)
  8. Tool: execute_query
  9. LLM call (26k chars) → Final answer
```

### New Flow (ship-RAG-ai)
```
User Question
  ↓
FastAPI
  ↓
SQLAgentGraph (LangGraph)
  ↓
[Conditional Sequence]
  1. LLM call with concise prompt (5k chars)
  2. Decision: Need examples?
     ├─ Yes → Tool: get_few_shot_examples (FAISS)
     └─ No → Skip
  3. LLM call with examples (if retrieved)
  4. Tool: execute_db_query (PostgreSQL)
  5. LLM call → Final answer
```

## Token Usage Comparison

### Example: "Count devices with temperature > 10"

**Original (ship-ai):**
- Call 1: ~20,000 chars (prompt + question)
- Call 2: ~22,000 chars (+ table list)
- Call 3: ~24,000 chars (+ schema)
- Call 4: ~25,000 chars (+ query)
- Call 5: ~26,000 chars (+ results)
- **Total: ~117,000 chars**

**New (ship-RAG-ai):**
- Call 1: ~5,000 chars (concise prompt + question)
- Call 2: ~7,000 chars (+ examples if retrieved)
- Call 3: ~8,000 chars (+ query execution)
- **Total: ~20,000 chars** (if examples retrieved)
- **Total: ~13,000 chars** (if examples skipped)

**Savings: ~80-90% reduction in token usage**

## Configuration

### Environment Variables

Same as original:
- `HOST`, `DATABASE`, `USER`, `PASSWORD`, `SSL_MODE`
- `API_KEY` (OpenAI)

### New Behavior

- FAISS indexes created automatically on first run
- Indexes stored in `./faiss_index/` directory
- Subsequent runs load existing indexes (no re-creation)

## API Compatibility

The API endpoint remains the same:

```bash
POST /chat
{
  "token_id": "Test123",
  "question": "...",
  "user_id": "27"
}
```

Response format is identical, so existing clients work without changes.

## Migration Steps

1. **Install new dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Copy `.env` file:**
   - Same environment variables as original
   - No changes needed

3. **Start application:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 3009
   ```

4. **First run:**
   - Creates FAISS indexes automatically
   - Takes a few seconds longer on first startup
   - Subsequent runs are faster

## Testing

Test with the same questions as before:

```bash
curl -X POST "http://localhost:3009/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "token_id": "Test123",
    "question": "Count devices that have current temperature more than 10 degrees",
    "user_id": "27"
  }'
```

## Benefits

1. **Reduced Token Usage**: 80-90% reduction in tokens sent to LLM
2. **Faster Responses**: Fewer LLM calls for simple questions
3. **Better Scalability**: Examples retrieved only when needed
4. **More Flexible**: Agent can adapt workflow based on question complexity
5. **Easier Maintenance**: Examples and business rules in separate data files

## Known Limitations

1. **First Run**: Takes longer to create FAISS indexes
2. **FAISS Storage**: Requires disk space for indexes (~few MB)
3. **LangGraph Learning Curve**: Different from LangChain patterns

## Future Improvements

- Add caching for frequently asked questions
- Implement streaming responses
- Add more sophisticated example retrieval logic
- Support for multiple vector stores (different example categories)
- Add metrics/monitoring for token usage

