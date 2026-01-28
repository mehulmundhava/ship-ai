# Why "Max Temperature on Date X" Used device_current_data (and how it’s fixed)

## Path and logic that led to device_current_data

### 1. Request flow

- **Question:** "What was the maximum temperature on 24th January 2026 for device id WT01EB25EC389682?"
- **user_id:** 27

### 2. Where examples and schema come from

- **Examples** come from `ai_vector_examples`: each row has `question`, `sql_query`, `description`, and an embedding computed from `question`.
- **Schema / rules** come from `ai_vector_extra_prompts`: each row has `content` and an embedding. Both are filled from `scripts/examples_data.py` (and optionally `scripts/journey_examples_data.py`) via `scripts/load_examples_data.py`.

### 3. What the agent actually saw (your run)

**Vector search – Examples (k=2):**

- Query embedding = embedding of the user question.
- Top 2 by similarity were:
  - **[1]** “Which of my devices are **currently** approaching their maximum temperature threshold (within 2 degrees)?” → SQL uses `device_current_data`.
  - **[2]** “Which of my devices are **currently** operating above 10°C?” → SQL uses `device_current_data`.

**Vector search – Extra prompts (k=1):**

- Top 1 was:
  - “**device_current_data (CD)** – Contains **current** data of devices like temperature, battery, dwell-time, …”

So the agent’s only retrieved examples and schema snippet were about “current” temperature and `device_current_data`. It had no example and no rule that said “for **historical** temperature **on a specific date** use another table.”

### 4. Why device_current_data was chosen

- The question mentions “temperature” and “maximum,” so it is semantically close to “current temperature” / “maximum temperature threshold” examples.
- Those examples and the retrieved schema all point to `device_current_data`.
- “What was … on 24th January 2026” is a **historical-by-date** question; the agent was never steered toward a time-series / history table (e.g. `incoming_message_history_k` or `incoming_message_table`), so it adapted the “current temperature” pattern and used `device_current_data` with `cd.updated_at::date = '2026-01-24'`. That is consistent with what it was shown, but the wrong table for “temperature on a past date.”

### 5. Summary of cause

| What drove the decision | Source |
|-------------------------|--------|
| Examples used          | Top-2 similar **questions** in `ai_vector_examples` → both “current” temperature, both `device_current_data`. |
| Schema used             | Top-1 similar **content** in `ai_vector_extra_prompts` → “device_current_data … current data … temperature …”. |
| Missing signal          | No example “What was the max temperature **on date X** for device Y?” and no rule “for historical temperature **by date** use incoming_message_history_k / incoming_message_table; device_current_data is current snapshot only.” |

So the path was: **question → embed → nearest examples + nearest extra prompt → both pointed to `device_current_data` → agent used it**. The fix is to add the missing examples and rules in the **training data** (examples + extra prompts), not by hardcoding logic in the app.

---

## Fix (no hardcoding in app code)

All changes are in **data** loaded into the vector store:

1. **Extra prompt: “current vs historical” rule**  
   - New row in `EXTRA_PROMPT_DATA` (in `scripts/examples_data.py`).  
   - States: for “what was the max/min temperature **on a specific date**” use the time-series table (`incoming_message_history_k` or `incoming_message_table`); `device_current_data` is for **current** snapshot only.  
   - Keywords include e.g. `"historical"`, `"maximum temperature"`, `"specific date"`, `"incoming_message_history_k"`, `"device_current_data"` so this rule is retrieved for questions like “max temp on date X”.

2. **Better description of the history table**  
   - The existing “incoming_message_history_k (IK)” extra prompt is updated to mention **temperature**, **battery**, and **event-time** (or your actual date/timestamp column), so it becomes relevant for “temperature on date” and similar queries.

3. **Few-shot example for “max temperature on date X”**  
   - New row in `SAMPLE_EXAMPLES`: question = “What was the maximum temperature on 24 January 2026 for device id WT01EB25EC389682?” (or a generic “on &lt;date&gt; for device id &lt;id&gt;”).  
   - SQL uses the **historical** table (`incoming_message_history_k` or `incoming_message_table`) with a date filter on the timestamp column (e.g. `event_time::date = '2026-01-24'`), not `device_current_data`.

After loading these via `load_examples_data.py` and re-running embeddings for `ai_vector_examples` and `ai_vector_extra_prompts`, the same question will retrieve this new example and the new rule, and the agent will be directed to the correct table without any new application logic.

---

## Applying the fix when your DB is populated elsewhere

Your **7 temperature examples** in `ai_vector_examples` don’t need to be changed. You only need to **add** the new example and the new extra prompt, then regenerate embeddings.

**Option A – Run the migration SQL (recommended):**

1. Run `migrations/insert_historical_temperature_example_and_rule.sql` against your DB (adds 1 example + 1 extra prompt, skips if already present).
2. Regenerate embeddings:
   - `POST /generate-embeddings-examples` (no body, or `{"id": <new_example_id>}`)
   - `POST /generate-embeddings-extra-prompts` (no body, or `{"id": <new_extra_prompt_id>}`)

**Option B – Use the load script:**

If you sync from `scripts/examples_data.py`, run `load_examples_data.py` so the new example and extra prompts from that file are loaded, then run the same embedding calls as above.

## If your table is named `incoming_message_table` or columns differ

- The new **example** uses `incoming_message_history_k` and `event_time::date`. If your schema uses:
  - **Table:** `incoming_message_table` → in the SQL migration or in `examples_data.py`, change the `FROM`/alias to `incoming_message_table`, then (re)run the migration or load script and regenerate embeddings.
  - **Date column:** e.g. `created_at`, `timestamp_utc` → use that instead of `event_time` in the example SQL, then reload and re-embed.
- The **extra prompts** already mention both “incoming_message_history_k” and “incoming_message_table”, so no change there unless you want to stress one name.
