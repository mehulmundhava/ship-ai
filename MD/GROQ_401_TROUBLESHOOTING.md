# Groq 401 / API Key Failure

## What the app does

When the app uses **Groq** as the LLM provider, it calls the Groq API. If the API returns an error whose message contains **`401`** or **`expired_api_key`**, the app:

1. Treats it as an **auth failure** (invalid or expired key).
2. Skips Groq for the rest of that request and uses the **OpenAI fallback** instead.
3. For the rest of the **process**, skips Groq for all requests (so you don’t get repeated 401s until you restart the server).

So if you see logs like:

- `Groq 401/invalid key: skipping Groq for this request, using OpenAI fallback`
- `Skipping Groq (GROQ_DISABLED or 401 seen), using fallback directly`

the app has decided Groq auth failed and is using OpenAI instead.

## Why does the Groq API key “fail” (401)?

**HTTP 401 = Unauthorized.** The Groq API is rejecting the request because of authentication. Common causes:

| Cause | What to do |
|-------|------------|
| **Invalid API key** | Key is wrong, typo, or copied incorrectly. Get a new key from [Groq Console](https://console.groq.com/) and set `GROQ_API_KEY` in `.env`. |
| **Expired API key** | Groq may expire keys (e.g. trial/usage limits). Create a new key in the Groq Console and update `GROQ_API_KEY`. |
| **Key not set or empty** | App reads `GROQ_API_KEY` from env. Ensure `.env` has `GROQ_API_KEY=gsk-...` (no quotes needed, no spaces). |
| **Account / billing** | Account disabled, trial ended, or billing issue. Check Groq Console and account status. |
| **Wrong env var** | Must be `GROQ_API_KEY`. Not `GROQ_KEY` or `API_KEY` (that’s for OpenAI). |

So it’s **not only** “key expired” — it can be invalid key, missing key, or account/billing. The app only knows “Groq returned 401” and then uses the fallback.

**To see the exact reason:** The app now logs the **full Groq error** when 401 is detected. In your server logs, look for a line like:

`Groq 401/invalid key: ... Full error (check for expired/deleted/limit): <actual API message>`

The `<actual API message>` may say e.g. "Invalid API key", "expired_api_key", "rate_limit_exceeded", or a JSON body from Groq — use that to tell whether the key is expired, deleted, or a limit was reached.

## How to fix

1. **Check `.env`**
   - `GROQ_API_KEY` must be set to a valid Groq key (usually starts with `gsk-`).
   - Restart the app after changing `.env`.

2. **Get a new key**
   - Go to [Groq Console](https://console.groq.com/) → API Keys.
   - Create a new key and put it in `GROQ_API_KEY` in `.env`.

3. **Temporarily disable Groq**
   - If you want to avoid 401s and always use OpenAI, set in `.env`:
     - `GROQ_DISABLED=true`
   - Then the app will skip Groq and use the fallback directly (no 401, no retry delay).

## Where this is handled in code

- **agent_graph.py**: `_invoke_groq_with_recovery` catches exceptions and checks for `"401"` or `"expired_api_key"` in the error string; on match it sets `_groq_401_seen_this_request` and `_groq_401_seen_process` and returns `None` so the caller uses the OpenAI fallback.
- **llm_service.py**: `groq_llm_model()` uses `settings.groq_api_key` (from `GROQ_API_KEY`). If the key is missing at startup, the app can still start if the first LLM call is the fallback; the 401 is seen when the first Groq call is made.
