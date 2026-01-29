# Request timing breakdown (from your logs)

This document summarizes where time is spent on a single chat request, from your terminal output.

## Timeline (one request, ~12.4s total)

| Phase | Log label | Duration | What it does |
|-------|-----------|----------|--------------|
| 1 | `[80% path] miss` | **~2.08s** | Cache/shortcut: embed question, vector search on `ai_vector_examples` (LIMIT 3). No match (top=0.33 < 0.8), so we continue to LLM. |
| 2 | `process=security_guard` | **~1.41s** | LLM call: security guard allows/blocks the question. |
| 3 | `process=vector_search` | **~3.04s** | Build system prompt: **embed question again** + `search_examples(question, k=2)` + `search_extra_prompts(question, k=1)` (each does embed + DB). |
| 4 | `process=agent_node` | **~2.11s** | LLM call: agent generates SQL and tool_calls. |
| 5 | `process=tool_exec` | **~3.58s** | Run `execute_db_query`: run SQL, fetch 23,803 rows, then `format_result_with_csv`. |
| 6 | `process=format_answer` | **~2.07s** | LLM call: format final answer with count + CSV link. |
| – | `process=request` | **~12.22s** | Total agent time (phases 2–6). |

**Total from user POV:** ~12.4s (includes ~2.08s for 80% path before agent).

## Hypotheses to validate with instrumentation

1. **Hypothesis A (80% path):** The ~2.08s is mostly **embed_query** (HuggingFace BGE on CPU), not the DB vector search.
2. **Hypothesis B (vector_search):** The ~3.04s is mostly **two more embed_query calls** (one in `search_examples`, one in `search_extra_prompts`) plus DB; same question is embedded again after already being embedded in the 80% path.
3. **Hypothesis C:** Total time is dominated by **sequential LLM calls** (security + agent + format); we already have stage timings for this.
4. **Hypothesis D (tool_exec):** The ~3.58s is mostly **SQL execution + fetching 23k rows** and **CSV generation**, not tool wiring.
5. **Hypothesis E (redundant work):** The **same question is embedded 3 times**: once in 80% path, once in `search_examples`, once in `search_extra_prompts`. That could be 3× embed cost (~5s+) before any LLM runs.

## Next step

Instrumentation has been added to log:

- **80% path:** `embed_ms` vs `db_ms` (hypothesis A, E).
- **search_examples / search_extra_prompts:** `embed_ms` vs `db_ms` per call (hypothesis B, E).
- **execute_db_query:** `sql_ms` vs `csv_ms` when CSV is generated (hypothesis D).

After you reproduce one request, we’ll read the debug log, confirm or reject each hypothesis, then propose targeted optimizations (e.g. reuse one embedding, or reduce duplicate vector searches) without changing correct behavior.
