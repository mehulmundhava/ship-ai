# 80% Similarity Logic Without Caching — Design & Changes

## Goal

- **80% logic must work**: When the user’s question is ≥80% similar to an existing **example** (training query) in `ai_vector_examples`, run that example’s SQL (or an adapted version) and return the result **without calling the LLM**.
- **No caching**: Do not store new cache rows, do not read cached answers, do not use cache-specific columns.
- **Minimal DB columns**: Use only the **base** columns of `ai_vector_examples`. Avoid adding or depending on the migration columns from `add_vector_cache_columns.sql` (lines 8–16).

---

## What “80% logic” does (without cache)

1. Embed the user question.
2. Search `ai_vector_examples` by vector similarity (nearest neighbors).
3. For the top candidate(s), compute similarity = `1 - distance`.
4. If best similarity **≥ 0.80**:
   - Extract parameters from the **question** (e.g. device_id, facility_id, user_id from context).
   - Extract parameters from the **example’s SQL**.
   - If needed, adapt the example SQL to the current request (e.g. replace user_id, device_id).
   - Execute the (possibly adapted) SQL, format the result, return it → **no LLM**.
5. If no candidate reaches ≥0.80 or adaptation/execution fails → return `None` → normal LLM flow runs.

---

## DB: columns used

### Base columns only (no cache columns)

Use **only** these on `ai_vector_examples`:

| Column | Purpose |
|--------|---------|
| `id` | Row identity |
| `question` | Example question text |
| `sql_query` | Example SQL to run or adapt |
| `description` | Optional; not required for 80% logic |
| `metadata` | Optional; can hold stored params if you already use it |
| `<embedding_column>` | `minilm_embedding` or `bge_large_embedding` (from settings) |

**Not used** for 80%-only logic:

- `entry_type`
- `answer`
- `question_type`
- `user_type`
- `similarity_threshold`
- `usage_count`
- `last_used`
- `is_active`
- `is_deterministic`

So the 80% path uses **no** columns from `add_vector_cache_columns.sql` (8–16).

---

## Migration / schema

- **Option A — You have not applied the cache migration**  
  Do **not** run `add_vector_cache_columns.sql`. Keep `ai_vector_examples` with only base columns + embedding.

- **Option B — You already applied the cache migration**  
  For 80%-only behavior you can either:
  - **Revert** the cache columns using a migration like the one in the next subsection, **or**
  - **Leave** them in the DB but **don’t use them** in the new code path: the new “80% match and execute” logic will query only `id`, `question`, `sql_query`, `metadata`, and the embedding column.

**Example revert migration** (only if you want to physically remove the cache columns):

```sql
-- migrations/revert_vector_cache_columns.sql
-- Run only if you no longer need cache and want to drop those columns.

ALTER TABLE ai_vector_examples
  DROP COLUMN IF EXISTS entry_type,
  DROP COLUMN IF EXISTS answer,
  DROP COLUMN IF EXISTS question_type,
  DROP COLUMN IF EXISTS user_type,
  DROP COLUMN IF EXISTS similarity_threshold,
  DROP COLUMN IF EXISTS usage_count,
  DROP COLUMN IF EXISTS last_used,
  DROP COLUMN IF EXISTS is_active,
  DROP COLUMN IF EXISTS is_deterministic;
```

---

## Code changes

### 1. New “80% match and execute” path (no cache, minimal columns)

Introduce a single function that uses only base columns and never touches cache columns or `save_answer_to_cache`.

**Where:** New module `app/services/match_and_execute_service.py` **or** a new function in `app/services/cache_answer_service.py` (e.g. `check_80_match_and_execute`) that uses a **different** query.

**Query used by the 80%-only path:**

```sql
SELECT 
    id,
    question,
    sql_query,
    metadata,
    {embedding_field} <-> :embedding::vector AS distance,
    1 - ({embedding_field} <-> :embedding::vector) AS similarity
FROM ai_vector_examples
WHERE {embedding_field} IS NOT NULL
ORDER BY {embedding_field} <-> :embedding::vector
LIMIT 3
```

- No `entry_type`, `question_type`, `user_type`, `is_active` in `WHERE` or `SELECT`.
- All rows are treated as examples.

**Logic (pseudocode):**

1. Embed `question`; run the query above.
2. For each row (e.g. top 3):
   - If `similarity < 0.80` → skip.
   - `current_params = extract_params_from_question(question)` and, if you have a convention, inject `user_id` from request (e.g. into a `user_id` param).
   - `example_params = extract_sql_parameters(row.sql_query)` (or from `row.metadata['params']` if you store it).
   - If params match → execute `row.sql_query` (with request `user_id` etc. if needed).
   - Else if `similarity >= 0.80` → call existing `_adapt_sql_parameters(row.sql_query, example_params, current_params)` then execute adapted SQL.
   - Use existing execution/formatting (e.g. `_try_adapt_and_execute_sql`-style logic, or `QuerySQLDatabaseTool` / `create_journey_list_tool` based on `question_type`).
3. If any candidate succeeds → return `{ answer, sql_query, result_data, similarity }`.
4. Else return `None` → controller falls back to LLM.

**Reuse from current code (no new .py “format” for logic):**

- `extract_params_from_question`, `extract_sql_parameters`, `parameters_match` (from `cache_answer_service` or moved to a shared util).
- `_adapt_sql_parameters`, `_try_adapt_and_execute_sql` (or their behavior) for adaptation and execution.
- Embedding and engine come from `VectorStoreService` and `sync_engine` / `SQLDatabase` as today.

So the “80% logic” lives in **one new function + one new query** that uses only base columns; the rest is reuse.

---

### 2. Chat controller

**File:** `app/controllers/chat_controller.py`

- When you want “80% without cache”:
  - **Do not** call `check_cached_answer` (or turn off cache for this mode).
  - **Do** call the new `check_80_match_and_execute(question, user_id, question_type, sql_db, vector_store)` (or equivalent).
  - If it returns a result → respond with that result (and e.g. `cached=False`, `llm_used=False`, `similarity=…`) and **do not** call `save_answer_to_cache`.

- To disable caching entirely in this mode:
  - `VECTOR_CACHE_ENABLED = False` and/or `VECTOR_CACHE_AUTO_SAVE = False`, and only use the new 80% path.

So the controller switches from “cache check + optional save” to “80% match-and-execute only, no cache read/write.”

---

### 3. Config

**File:** `app/config/settings.py`

- **Existing:**  
  - `VECTOR_CACHE_SIMILARITY_THRESHOLD = 0.80` — can be reused as the 80% bar for the new path (same number, no new column).
  - `VECTOR_CACHE_ENABLED`, `VECTOR_CACHE_AUTO_SAVE` — set to `False` when you want “80% only, no cache.”

- **Optional:**  
  - e.g. `VECTOR_80_MATCH_ENABLED = True` and `VECTOR_80_SIMILARITY_THRESHOLD = 0.80` to make “80% without cache” explicit and independent of cache flag names.

No new DB columns are needed; the 0.80 threshold stays in app config only.

---

### 4. Summary of code edits

| File | Change |
|------|--------|
| **New** `app/services/match_and_execute_service.py` **or** `app/services/cache_answer_service.py` | Add `check_80_match_and_execute(...)` that (a) runs the “base-columns-only” similarity query above, (b) applies the 80% threshold and param extraction/adapt/execute logic, (c) never reads or writes cache columns, and (d) reuses existing param/adapt/execute helpers. |
| `app/controllers/chat_controller.py` | Call the new 80% path instead of (or before) `check_cached_answer` when “80% without cache” is enabled; never call `save_answer_to_cache` in that mode. |
| `app/config/settings.py` | (Optional) Add `VECTOR_80_MATCH_ENABLED` / `VECTOR_80_SIMILARITY_THRESHOLD`; or reuse `VECTOR_CACHE_SIMILARITY_THRESHOLD` and rely on `VECTOR_CACHE_ENABLED=False` to mean “no cache, only 80% match.” |

No new tables and **no** new columns in `ai_vector_examples` for the 80% path. The migration `add_vector_cache_columns.sql` (lines 8–16) is **not** used by this design.

---

## Flow comparison

| Step | Current (cache) | 80% without cache (this design) |
|------|-----------------|----------------------------------|
| 1 | Embed question | Same |
| 2 | Query by similarity + filter `entry_type` / `question_type` / `user_type` / `is_active` | Query by similarity **only**; **no** filter on cache columns |
| 3 | If hit and params match → return **stored answer** (no execution) | If similarity ≥ 0.80 → **always** run (or adapt+run) example SQL and format result |
| 4 | Optionally try “adapt & execute” if similarity ≥ 0.8 and params differ | Same “adapt & execute” behavior when similarity ≥ 0.80 and params differ |
| 5 | After LLM → `save_answer_to_cache` | **Do not** save to cache |

---

## What you need to implement

1. **One new function** that:
   - Takes: `question`, `user_id`, `question_type`, `sql_db`, `vector_store` (and any other execution context you already use).
   - Uses only the “base-columns-only” similarity query.
   - Implements the 80% threshold and param match/adapt/execute logic above, reusing existing helpers.
2. **Controller**: Call that function when “80% without cache” is on; skip cache check and `save_answer_to_cache`.
3. **Config**: Use a 0.80 threshold from settings (existing or new name); disable cache flags if you want no caching at all.

No .py “format” is required beyond normal Python; the logic is “one function + one query + reuse of existing param/adapt/execute code,” and it uses **as few columns as possible** — only the base columns of `ai_vector_examples`, and **none** of the new columns from `add_vector_cache_columns.sql` (8–16).
