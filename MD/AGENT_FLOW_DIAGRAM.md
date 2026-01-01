# Complete Agent Flow Documentation

## Overview

This document explains the complete flow of the Ship RAG AI application, from the moment a user sends a chat request to receiving the final answer. The flow is centered around the LangGraph agent implementation in `app/core/agent/agent_graph.py`.

---

## 🚀 High-Level Flow Diagram

```
User Request (POST /chat)
    ↓
FastAPI Route (app/api/routes/chat.py)
    ↓
Chat Controller (app/controllers/chat_controller.py)
    ↓
SQLAgentGraph.invoke() (app/core/agent/agent_graph.py)
    ↓
LangGraph Workflow Execution
    ├─→ Security Guard (if non-admin user)
    ├─→ Agent Node (LLM decides what to do)
    ├─→ Tool Node (execute tools if needed)
    └─→ Format Answer Node (generate final response)
    ↓
Return Response to User
```

---

## 📋 Step-by-Step Flow

### Step 1: API Request Entry Point

**File:** `app/api/routes/chat.py`

**Function:** `chat_api()`

**What Happens:**
1. FastAPI receives POST request to `/chat` endpoint
2. Extracts `ChatRequest` payload (question, user_id, token_id)
3. Gets pre-initialized components from `app.state`:
   - `llm_model` - LLM service instance
   - `vector_store` - Vector store service instance
   - `sql_db` - SQLDatabase wrapper instance
4. Calls `process_chat()` controller function

**Code:**
```python
@router.post("/chat", response_model=ChatResponse)
def chat_api(request: Request, payload: ChatRequest):
    llm_model = request.app.state.llm_model
    vector_store = request.app.state.vector_store
    sql_db = request.app.state.sql_db
    
    return process_chat(
        payload=payload,
        llm_model=llm_model,
        vector_store=vector_store,
        sql_db=sql_db
    )
```

---

### Step 2: Controller Processing

**File:** `app/controllers/chat_controller.py`

**Function:** `process_chat()`

**What Happens:**
1. **Authentication Check**: Validates `token_id == "Test123"`
2. **Database Connection**: Ensures SQLDatabase is initialized
3. **LLM Setup**: Gets LLM instance from LLMService
4. **Agent Creation**: Creates `SQLAgentGraph` instance with:
   - LLM model
   - SQLDatabase instance
   - VectorStoreService instance
   - User ID
   - Top K parameter (20)
5. **Invoke Agent**: Calls `agent.invoke(payload.question)`
6. **Format Response**: Returns `ChatResponse` with answer, SQL query, and debug info

**Code:**
```python
# Create SQL agent graph
agent = SQLAgentGraph(
    llm=llm,
    db=sql_db,
    vector_store_manager=vector_store,
    user_id=payload.user_id,
    top_k=20
)

# Process the question
result = agent.invoke(payload.question)
```

---

### Step 3: Agent Graph Initialization

**File:** `app/core/agent/agent_graph.py`

**Class:** `SQLAgentGraph`

**Function:** `__init__()`

**What Happens:**
1. Stores LLM, database, vector store, and user_id
2. **Creates Tools**:
   - `get_few_shot_examples_tool` - Retrieves examples from vector store
   - `execute_db_query_tool` - Executes SQL queries
3. **Binds Tools to LLM**: `llm.bind_tools(self.tools)`
4. **Builds Graph**: Calls `_build_graph()` to create LangGraph workflow

**Graph Structure:**
```
Entry Point: "agent" node
    ↓
[Conditional Edge: _should_continue()]
    ├─→ "continue" → "tools" node
    └─→ "end" → "format_answer" node
        ↓
[Conditional Edge: _should_format_after_tools()]
    ├─→ "format" → "format_answer" node
    └─→ "continue" → "agent" node (loop back)
        ↓
"format_answer" node → END
```

---

### Step 4: Agent Invocation

**File:** `app/core/agent/agent_graph.py`

**Function:** `invoke(question: str)`

**What Happens:**
1. Creates initial state dictionary:
   ```python
   {
       "question": question,
       "user_id": self.user_id,
       "messages": [],
       "examples_retrieved": False,
       "sql_query": None,
       "query_result": None,
       "final_answer": None,
       "iteration_count": 0,
       "token_usage": {"input": 0, "output": 0, "total": 0},
       "query_validated": False,
       "llm_call_history": []
   }
   ```
2. Executes graph: `self.graph.invoke(initial_state)`
3. Extracts results from final state
4. Builds debug information
5. Returns dictionary with answer, SQL query, and debug info

---

### Step 5: Agent Node Execution

**File:** `app/core/agent/agent_graph.py`

**Function:** `_agent_node(state: AgentState)`

**This is the core decision-making node. Here's what happens:**

#### 5.1: Security Guard (First Time Only, Non-Admin Users)

**When:** First time through agent node, user is not "admin", query not yet validated

**What Happens:**
1. Creates security guard prompt (minimal, focused on validation)
2. Calls LLM with security guard prompt
3. LLM responds with "ALLOW" or "BLOCK"
4. If "BLOCK": Returns early with error message
5. If "ALLOW": Continues to full prompt setup
6. Records security guard call in `llm_call_history`

**Security Guard Prompt:**
```
You are a Database Security Guard. Your only job is to classify user questions.

IMPORTANT: The current user_id is {user_id}. This user can ONLY access their own data.

RULES:
1. SAFE: Questions about device metrics for the current user...
2. RISKY: Questions asking for data for another user...

If the question is RISKY, respond with ONLY the word 'BLOCK'.
If the question is SAFE, respond with ONLY the word 'ALLOW'.
```

#### 5.2: Message Initialization

**When:** First time through (after security guard passes) or messages need setup

**What Happens:**
1. **Gets System Prompt**: Calls `get_system_prompt()` with:
   - User ID
   - Top K (20)
   - Question
   - Vector store manager
   - `preload_examples=True` (pre-loads 1-2 examples)
2. **Creates Base Messages**:
   ```python
   [
       SystemMessage(content=system_prompt),  # Full prompt with examples
       HumanMessage(content=question)          # User's question
   ]
   ```
3. **Preserves Existing Messages**: Adds any existing AIMessages and ToolMessages

#### 5.3: Message Validation

**When:** Messages already exist (subsequent iterations)

**What Happens:**
1. Validates message sequence
2. Ensures ToolMessages follow their corresponding AIMessages
3. Filters out `check_user_query_restriction` tool messages (to save tokens)

#### 5.4: LLM Invocation

**What Happens:**
1. Logs message sequence before invocation
2. Invokes LLM with tools: `self.llm_with_tools.invoke(messages)`
3. LLM can:
   - Return a text response (final answer)
   - Request tool calls (get_few_shot_examples or execute_db_query)
4. **Tracks Token Usage**: Extracts from `response_metadata`
5. **Records LLM Call**: Adds to `llm_call_history` with:
   - Input messages summary
   - Output content/tool_calls
   - Token usage
6. Appends LLM response to messages

#### 5.5: State Update

**What Happens:**
1. Extracts SQL query from tool calls (if `execute_db_query` was called)
2. Extracts query result from ToolMessages (if available)
3. Increments `iteration_count`
4. Updates `token_usage` in state
5. Returns updated state

---

### Step 6: Decision Point - Should Continue?

**File:** `app/core/agent/agent_graph.py`

**Function:** `_should_continue(state: AgentState)`

**What Happens:**
This function decides whether to:
- **Continue** → Call tools (go to "tools" node)
- **End** → Format answer (go to "format_answer" node)

**Decision Logic:**
1. If `final_answer` already set → **END**
2. If `iteration_count >= 3` (safety limit) → **END**
3. If last message has `tool_calls` → **CONTINUE** (go to tools)
4. If `query_result` exists → **END** (we have results, format answer)
5. If ToolMessage with `execute_db_query` exists → **END** (already executed)
6. If last message is AIMessage without tool_calls → **END** (final answer)
7. Otherwise → **CONTINUE** (might need tools)

---

### Step 7: Tool Node Execution

**File:** `app/core/agent/agent_graph.py`

**Function:** `logged_tool_node(state: AgentState)`

**What Happens:**
1. **Preserves Message History**: Keeps all existing messages (System, Human, AI)
2. **Extracts Tool Calls**: Gets tool calls from last AIMessage
3. **Logs Tool Execution**: Prints which tools are being called
4. **Executes Tools**: Calls `ToolNode.invoke(state)` which:
   - Executes each tool call
   - Returns ToolMessages with results
5. **Merges Messages**: Combines preserved messages + new ToolMessages
6. **Logs Results**: Prints tool execution results
7. Returns state with complete message history

**Available Tools:**

#### Tool 1: `get_few_shot_examples`
**File:** `app/core/agent/agent_tools.py`

**Function:** `get_few_shot_examples(question: str)`

**What Happens:**
1. Searches vector store for similar examples: `vector_store_manager.search_examples(question, k=3)`
2. Searches for extra prompts: `vector_store_manager.search_extra_prompts(question, k=2)`
3. Formats results into a string
4. Returns formatted examples

**When Used:**
- LLM decides it needs more examples beyond what's pre-loaded
- Pre-loaded examples don't match the use case

#### Tool 2: `execute_db_query`
**File:** `app/core/agent/agent_tools.py`

**Function:** `execute_db_query(query: str, user_id: str)`

**What Happens:**
1. **Security Check**: Validates query is SELECT only
2. **Restriction Check**: Ensures query doesn't directly access restricted tables
3. **User ID Injection**: Adds `user_id` filter if not present
4. **Query Execution**: Executes SQL query via `SQLDatabase.run(query)`
5. **Result Formatting**: Formats results as string
6. Returns query results

**Security Features:**
- Only SELECT queries allowed
- Restricted tables (admin, user_device_assignment) cannot be directly queried
- User ID filter automatically added for data isolation

---

### Step 8: Decision Point - Format After Tools?

**File:** `app/core/agent/agent_graph.py`

**Function:** `_should_format_after_tools(state: AgentState)`

**What Happens:**
After tool execution, decides:
- **Format** → Go directly to format_answer (skip agent node, saves tokens)
- **Continue** → Go back to agent node (might need more processing)

**Decision Logic:**
1. If ToolMessage with `execute_db_query` exists → **FORMAT** (we have results)
2. Otherwise → **CONTINUE** (go back to agent)

**Token Optimization:**
This decision saves tokens by skipping the agent node when we already have query results. Instead of sending full message history to LLM again, we go directly to formatting.

---

### Step 9: Format Answer Node

**File:** `app/core/agent/agent_graph.py`

**Function:** `_format_answer(state: AgentState)`

**What Happens:**

#### 9.1: Extract Results
1. Extracts `query_result` from state or ToolMessages
2. Extracts `sql_query` from state or tool calls

#### 9.2: Generate Final Answer

**If Query Results Exist:**
1. Creates **MINIMAL** prompt (saves tokens):
   ```
   You are a helpful assistant. Your task is ONLY to explain database query results
   to the user in clear, natural language.
   
   User Question: {question}
   SQL Query: {sql_query}
   SQL Query Results: {query_result}
   
   Now provide a short, user-friendly answer.
   ```
2. Calls LLM with minimal prompt (no system prompt, no examples, no history)
3. Tracks token usage
4. Records in `llm_call_history`
5. Sets `final_answer` in state

**If No Query Results:**
1. Uses last AIMessage content as final answer
2. Or returns error message

#### 9.3: Return Final State
Returns state with `final_answer` set

---

## 🔄 Complete Flow Example

Let's trace through a complete example:

### Example: "Count devices with temperature > 10"

#### Step 1: API Request
```
POST /chat
{
    "token_id": "Test123",
    "question": "Count devices with temperature > 10",
    "user_id": "27"
}
```

#### Step 2: Controller
- Validates token_id
- Creates SQLAgentGraph
- Calls `agent.invoke("Count devices with temperature > 10")`

#### Step 3: Agent Graph - Initial State
```python
{
    "question": "Count devices with temperature > 10",
    "user_id": "27",
    "messages": [],
    "iteration_count": 0,
    ...
}
```

#### Step 4: Agent Node (Iteration 1)
1. **Security Guard**: Validates query → "ALLOW"
2. **Message Setup**: Creates SystemMessage with full prompt + pre-loaded examples
3. **LLM Call**: Sends to LLM with tools
4. **LLM Response**: Requests `execute_db_query` tool call
5. **State Update**: Adds AIMessage with tool_calls

#### Step 5: Decision Point
- `_should_continue()` → **CONTINUE** (has tool_calls)

#### Step 6: Tool Node
1. Executes `execute_db_query` tool
2. Tool generates SQL: `SELECT COUNT(*) FROM device_current_data WHERE temperature > 10 AND user_id = '27'`
3. Executes query
4. Returns ToolMessage with results: "15"

#### Step 7: Decision Point After Tools
- `_should_format_after_tools()` → **FORMAT** (has query results)

#### Step 8: Format Answer Node
1. Extracts SQL query and results
2. Creates minimal prompt
3. Calls LLM: "There are 15 devices with temperature above 10 degrees."
4. Sets `final_answer`

#### Step 9: Return Response
```json
{
    "token_id": "Test123",
    "answer": "There are 15 devices with temperature above 10 degrees.",
    "sql_query": "SELECT COUNT(*) FROM device_current_data WHERE temperature > 10 AND user_id = '27'",
    "debug": {
        "llm_call_history": [...],
        "token_usage": {...},
        ...
    }
}
```

---

## 🎯 Key Components Explained

### AgentState (TypedDict)
The state dictionary that flows through the graph:

```python
{
    "question": str,                    # User's original question
    "user_id": Optional[str],          # User ID for access control
    "messages": List,                  # Full message history
    "examples_retrieved": bool,         # Whether examples were retrieved
    "sql_query": Optional[str],         # Generated SQL query
    "query_result": Optional[str],      # Query execution results
    "final_answer": Optional[str],      # Final natural language answer
    "iteration_count": int,             # Number of iterations (safety limit)
    "token_usage": Dict[str, int],      # Token usage tracking
    "query_validated": bool,            # Whether security guard passed
    "llm_call_history": List[Dict]      # Detailed LLM call tracking
}
```

### Message Types

1. **SystemMessage**: Contains system prompt with instructions and examples
2. **HumanMessage**: User's question
3. **AIMessage**: LLM responses (may contain tool_calls)
4. **ToolMessage**: Tool execution results

### Tool Calls

1. **get_few_shot_examples**: Retrieves similar examples from vector store
2. **execute_db_query**: Executes SQL queries against PostgreSQL

---

## 🔐 Security Features

### 1. Security Guard
- Validates user queries before processing
- Blocks queries asking for other users' data
- Blocks queries accessing restricted tables
- Admin users skip security guard

### 2. Query Restrictions
- Only SELECT queries allowed
- Restricted tables cannot be directly queried
- User ID filter automatically injected

### 3. User Isolation
- All queries automatically filtered by user_id
- Prevents cross-user data access

---

## 💰 Token Optimization Strategies

1. **Security Guard**: Minimal prompt (saves tokens on blocked queries)
2. **Pre-loaded Examples**: 1-2 examples in system prompt (reduces tool calls)
3. **Conditional Tool Calling**: LLM decides when to retrieve examples
4. **Direct Format Path**: Skips agent node when query results exist
5. **Minimal Format Prompt**: Final answer uses minimal prompt (no history)

---

## 📊 Debug Information

The response includes comprehensive debug information:

```python
{
    "question": str,
    "user_id": str,
    "total_messages": int,
    "message_history": List[Dict],      # All messages with details
    "llm_call_history": List[Dict],     # Detailed LLM call tracking:
    #   - security_guard: Query validation
    #   - agent_node: SQL generation
    #   - format_answer: Final answer generation
    "token_usage": {
        "input_tokens": int,
        "output_tokens": int,
        "total_tokens": int
    },
    "iterations": int,
    "sql_query": str,
    "query_result": str
}
```

---

## 🔄 Iteration Flow

The graph can loop multiple times:

```
Iteration 1:
  Agent Node → Tool Node (get_examples) → Agent Node

Iteration 2:
  Agent Node → Tool Node (execute_query) → Format Answer

Max Iterations: 3 (safety limit)
```

---

## 📝 Summary

1. **Entry**: FastAPI route receives request
2. **Controller**: Validates and creates agent
3. **Agent Init**: Creates tools and builds graph
4. **Security**: Validates query (non-admin users)
5. **Agent Node**: LLM decides what to do
6. **Tool Node**: Executes tools if needed
7. **Format Node**: Generates final answer
8. **Return**: Response with answer, SQL, and debug info

The flow is designed for:
- **Security**: Multiple validation layers
- **Efficiency**: Token optimization at every step
- **Flexibility**: Conditional tool calling
- **Transparency**: Comprehensive debug information

