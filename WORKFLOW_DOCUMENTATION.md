# Complete Workflow Documentation: Journey & Non-Journey Questions

## Table of Contents
1. [System Overview](#system-overview)
2. [Tech Stack](#tech-stack)
3. [Architecture](#architecture)
4. [Question Type Detection](#question-type-detection)
5. [Complete Workflow: Non-Journey Questions](#complete-workflow-non-journey-questions)
6. [Complete Workflow: Journey Questions](#complete-workflow-journey-questions)
7. [Models & LLM Usage](#models--llm-usage)
8. [Tools & Their Functions](#tools--their-functions)
9. [Error Handling & Fallback Logic](#error-handling--fallback-logic)
10. [Token Optimizations](#token-optimizations)
11. [Security & Access Control](#security--access-control)
12. [Data Flow Diagrams](#data-flow-diagrams)

---

## System Overview

This is a **RAG (Retrieval-Augmented Generation) SQL Agent** system that converts natural language questions into SQL queries and provides intelligent answers. The system handles two distinct types of questions:

1. **Non-Journey Questions**: Standard database queries (device lists, counts, facility information, etc.)
2. **Journey Questions**: Complex movement calculations requiring Python-based journey algorithms

The system uses **LangGraph** for orchestration, **PostgreSQL** with **pgvector** for vector search, and supports both **Groq** and **OpenAI** LLM providers with automatic fallback.

---

## Tech Stack

### Core Technologies
- **Python 3.11+**: Main programming language
- **FastAPI**: REST API framework
- **LangGraph**: Agent orchestration and workflow management
- **LangChain**: LLM abstraction layer and tool integration
- **PostgreSQL 14+**: Primary database with pgvector extension
- **SQLAlchemy**: Database ORM and connection management

### LLM Providers
- **Primary**: Groq (Llama 3.3 70B Versatile)
- **Fallback**: OpenAI (GPT-4o)
- **Configuration**: Via `LLM_PROVIDER` environment variable

### Embeddings
- **Model**: Hugging Face `BAAI/bge-large-en-v1.5` (or `all-MiniLM-L6-v2`)
- **Purpose**: Semantic search for example queries and business rules
- **Storage**: PostgreSQL pgvector extension

### Vector Search
- **Extension**: pgvector (PostgreSQL vector extension)
- **Tables**: 
  - `ai_vector_examples`: Example SQL queries with embeddings
  - `ai_vector_extra_prompts`: Business rules and schema information

### Additional Libraries
- **Pydantic**: Settings and data validation
- **python-dotenv**: Environment variable management
- **langchain-groq**: Groq LLM integration
- **langchain-openai**: OpenAI LLM integration
- **langchain-huggingface**: Hugging Face embeddings

---

## Architecture

### High-Level Architecture

```
┌─────────────────┐
│   FastAPI API   │
│   (REST Endpoints)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Chat Controller │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐      ┌──────────────┐
│  SQLAgentGraph  │──────│  LLM Service │──────│  Groq/OpenAI │
│  (LangGraph)    │      │              │      │  (with fallback)
└────────┬────────┘      └──────────────┘      └──────────────┘
         │
         ├──────────────────┬──────────────────┐
         ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   PostgreSQL │  │ Vector Store │  │ Journey Calc │
│   Database   │  │  (pgvector)  │  │   (Python)   │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Component Responsibilities

1. **Chat Controller** (`app/controllers/chat_controller.py`)
   - Receives HTTP requests
   - Initializes LLM service and agent graph
   - Returns formatted responses

2. **SQLAgentGraph** (`app/core/agent/agent_graph.py`)
   - Main orchestration using LangGraph
   - Manages workflow state
   - Coordinates tool calls
   - Handles error recovery

3. **LLM Service** (`app/services/llm_service.py`)
   - Provider abstraction (Groq/OpenAI)
   - Fallback logic management
   - LLM instance creation

4. **Vector Store Service** (`app/services/vector_store_service.py`)
   - Semantic search for examples
   - Embedding generation
   - Business rules retrieval

5. **Journey Calculator** (`app/core/journey_calculator.py`)
   - Python-based journey calculation
   - Handles complex movement logic
   - Generates journey lists and counts

---

## Question Type Detection

### Detection Logic

The system uses keyword-based detection to determine if a question is about journeys:

**Location**: `app/core/agent/agent_graph.py` - `_is_journey_question()` method

```python
journey_keywords = [
    "journey", "journeys", "movement", "facility to facility",
    "entered", "exited", "path", "traveled", "transition"
]
```

### Detection Impact

| Aspect | Journey Questions | Non-Journey Questions |
|--------|------------------|---------------------|
| **Tools Bound** | `journey_list_tool`, `journey_count_tool`, `execute_db_query` | `get_examples_tool`, `execute_db_query`, `count_query`, `list_query`, etc. |
| **Examples Loaded** | 1 example (SQL structure reference) | 2 examples + 1 business rule |
| **Processing** | SQL → Python algorithm → JSON | SQL → Direct results |
| **Token Usage** | ~3,000 tokens saved (fewer tools) | Standard token usage |

---

## Complete Workflow: Non-Journey Questions

### Step-by-Step Flow

#### **Step 1: Request Reception**
- **Location**: `app/api/routes/chat.py` → `app/controllers/chat_controller.py`
- **Action**: FastAPI receives POST request with question and user_id
- **Output**: Request payload validated

#### **Step 2: LLM Initialization**
- **Location**: `app/services/llm_service.py`
- **Action**: 
  - Read `LLM_PROVIDER` from `.env` (default: "GROQ")
  - Create LLM instance (Groq or OpenAI)
  - Store provider info for fallback logic
- **Model Used**: 
  - Groq: `llama-3.3-70b-versatile`
  - OpenAI: `gpt-4o`
- **Output**: LLM instance ready

#### **Step 3: Vector Store Initialization**
- **Location**: `app/services/vector_store_service.py`
- **Action**:
  - Initialize Hugging Face embeddings model
  - Verify PostgreSQL vector tables exist
- **Embedding Model**: `BAAI/bge-large-en-v1.5` (or configured model)
- **Output**: Vector store ready for semantic search

#### **Step 4: Agent Graph Creation**
- **Location**: `app/core/agent/agent_graph.py` - `__init__()`
- **Action**:
  - Create SQLAgentGraph instance
  - Initialize tools (regular tools for non-journey)
  - Build LangGraph workflow
  - Store provider info
- **Tools Created**:
  - `get_few_shot_examples_tool`
  - `execute_db_query_tool`
  - `count_query_tool`
  - `list_query_tool`
  - `get_extra_examples_tool`
  - `get_table_list_tool`
  - `get_table_structure_tool`
- **Output**: Agent graph ready

#### **Step 5: Security Guard Validation** (Non-Admin Users Only)
- **Location**: `app/core/agent/agent_graph.py` - `_agent_node()`
- **Action**:
  - Check if `query_validated` is False
  - If user_id != "admin", call security guard LLM
  - **LLM Call**: `_invoke_llm_with_fallback()` with minimal prompt
  - **Prompt**: "Security Guard. User ID: {user_id}. Respond ONLY: 'ALLOW' or 'BLOCK'"
  - **Model**: Same as primary provider (Groq/OpenAI with fallback)
- **Decision**:
  - If "BLOCK": Return error, stop processing
  - If "ALLOW": Continue to Step 6
- **Token Usage**: ~200-300 tokens (minimal prompt)

#### **Step 6: System Prompt Generation**
- **Location**: `app/core/prompts.py` - `get_system_prompt()`
- **Action**:
  - Detect question type (non-journey in this case)
  - Load 2 examples from vector store using semantic search
  - Load 1 business rule from vector store
  - Build system prompt with examples embedded
- **Vector Search**:
  - Query: User's question
  - Model: Hugging Face embeddings
  - Search: `ai_vector_examples` table (k=2)
  - Search: `ai_vector_extra_prompts` table (k=1)
- **Output**: Complete system prompt with examples (~2,000-4,000 tokens)

#### **Step 7: Tool Binding**
- **Location**: `app/core/agent/agent_graph.py` - `_bind_tools_conditionally()`
- **Action**:
  - Detect question type: Non-journey
  - Bind regular tools to LLM
  - Tools: `get_examples_tool`, `execute_db_query`, `count_query`, `list_query`, etc.
- **Token Impact**: ~3,000 tokens for tool definitions
- **Output**: `llm_with_tools` ready

#### **Step 8: Main Agent Node - LLM Invocation**
- **Location**: `app/core/agent/agent_graph.py` - `_agent_node()`
- **Action**:
  - Build message sequence: `[SystemMessage, HumanMessage]`
  - Call `_invoke_llm_with_fallback()` with `use_tools=True`
  - LLM generates response (may include tool calls)
- **LLM Processing**:
  - **Model**: Groq (primary) or OpenAI (fallback)
  - **Input**: System prompt + user question + examples
  - **Output**: AIMessage with optional tool_calls
- **Error Handling**: Automatic fallback to OpenAI if Groq fails
- **Token Usage**: ~3,000-8,000 tokens (depends on examples)

#### **Step 9: Tool Execution** (If LLM Requests Tools)
- **Location**: `app/core/agent/agent_graph.py` - `logged_tool_node()`
- **Action**:
  - Extract tool calls from LLM response
  - Execute tools via `ToolNode`
  - Common tools for non-journey:
    - `execute_db_query`: Execute SQL query
    - `count_query`: Execute COUNT/aggregation query
    - `list_query`: Execute LIST query with CSV generation
    - `get_few_shot_examples`: Retrieve more examples
- **SQL Execution**:
  - **Location**: `app/core/agent/agent_tools.py` - `QuerySQLDatabaseTool`
  - **Action**: Execute SQL against PostgreSQL
  - **Result Formatting**: 
    - If > 5 rows: Generate CSV, return summary + CSV link
    - If ≤ 5 rows: Return formatted results
- **Output**: ToolMessage with results

#### **Step 10: Decision Point**
- **Location**: `app/core/agent/agent_graph.py` - `_should_continue()`
- **Action**: Check if agent should continue or format answer
- **Decision Logic**:
  - If tool calls present → Continue to tools
  - If query result found → Go to format_answer
  - If final answer → End
  - Max iterations: 5 (safety limit)

#### **Step 11: Format Answer** (If Query Results Available)
- **Location**: `app/core/agent/agent_graph.py` - `_format_answer()`
- **Action**:
  - Extract SQL query and results from state
  - Build minimal prompt: `"User asked: {question}\nQuery results: {results}"`
  - Call `_invoke_llm_with_fallback()` with `use_tools=False`
- **LLM Processing**:
  - **Model**: Groq (primary) or OpenAI (fallback)
  - **Input**: Minimal prompt (no system prompt, no examples, no history)
  - **Output**: Natural language answer
- **Token Usage**: ~500-2,000 tokens (minimal prompt)
- **Output**: Final answer string

#### **Step 12: Response Return**
- **Location**: `app/controllers/chat_controller.py`
- **Action**: Format response with answer, SQL query, results, debug info
- **Output**: JSON response to client

### Non-Journey Workflow Diagram

```
Request → LLM Init → Vector Store → Agent Graph
    ↓
Security Guard (if non-admin)
    ↓
System Prompt (2 examples + 1 rule)
    ↓
Tool Binding (regular tools)
    ↓
LLM Invocation (with tools)
    ↓
Tool Execution (SQL query)
    ↓
Decision Point
    ↓
Format Answer (minimal prompt)
    ↓
Response
```

---

## Complete Workflow: Journey Questions

### Step-by-Step Flow

#### **Steps 1-4: Same as Non-Journey**
- Request reception
- LLM initialization
- Vector store initialization
- Agent graph creation

#### **Step 5: Security Guard Validation** (Same as Non-Journey)
- Same security guard logic
- Token usage: ~200-300 tokens

#### **Step 6: System Prompt Generation** (Optimized for Journey)
- **Location**: `app/core/prompts.py` - `get_system_prompt()`
- **Action**:
  - Detect question type: Journey (keyword detection)
  - Load **1 example only** (SQL structure reference)
  - **No business rules** loaded (saves tokens)
- **Vector Search**:
  - Query: User's question
  - Search: `ai_vector_examples` table (k=1)
  - **No search** in `ai_vector_extra_prompts`
- **Token Savings**: ~1,000 tokens vs non-journey
- **Output**: Minimal system prompt (~1,000-2,000 tokens)

#### **Step 7: Tool Binding** (Journey Tools Only)
- **Location**: `app/core/agent/agent_graph.py` - `_bind_tools_conditionally()`
- **Action**:
  - Detect question type: Journey
  - Bind journey tools to LLM
  - Tools: `journey_list_tool`, `journey_count_tool`, `execute_db_query`
- **Token Impact**: ~1,500 tokens (fewer tools)
- **Token Savings**: ~1,500 tokens vs regular tools
- **Output**: `llm_with_tools` ready with journey tools

#### **Step 8: Main Agent Node - LLM Invocation**
- **Location**: `app/core/agent/agent_graph.py` - `_agent_node()`
- **Action**:
  - Build message sequence: `[SystemMessage, HumanMessage]`
  - Call `_invoke_llm_with_fallback()` with `use_tools=True`
  - LLM generates SQL query for journey tool
- **LLM Processing**:
  - **Model**: Groq (primary) or OpenAI (fallback)
  - **Input**: System prompt + user question + 1 example
  - **Output**: AIMessage with `journey_list_tool` or `journey_count_tool` call
- **SQL Generated**: 
  ```sql
  SELECT dg.device_id, dg.facility_id, dg.facility_type, 
         dg.entry_event_time, dg.exit_event_time
  FROM device_geofencings dg
  JOIN user_device_assignment uda ON uda.device = dg.device_id
  WHERE uda.user_id = {user_id} [filters]
  ORDER BY dg.entry_event_time ASC
  ```
- **Token Usage**: ~2,000-5,000 tokens (optimized)

#### **Step 9: Journey Tool Execution**
- **Location**: `app/core/agent/agent_tools.py` - `journey_list_tool()` or `journey_count_tool()`
- **Action**:
  1. **SQL Execution**: Execute SQL query to fetch geofencing rows
  2. **Result Parsing**: Parse SQL results into list of dictionaries
  3. **Python Algorithm**: Run journey calculation algorithm
  4. **Result Formatting**: Format results with CSV if needed

##### **9.1: SQL Execution**
- **Location**: `QuerySQLDatabaseTool.execute()`
- **Action**: Execute SQL against PostgreSQL
- **Output**: Raw geofencing rows (device_id, facility_id, entry_event_time, exit_event_time)

##### **9.2: Result Parsing**
- **Location**: `app/core/agent/agent_tools.py` - `_parse_sql_result_to_dicts()`
- **Action**: Convert SQL result (pipe-separated, JSON, or Python list) to list of dicts
- **Handles**: Multiple timestamp formats, data type conversion
- **Output**: List of dictionaries with geofencing data

##### **9.3: Journey Calculation Algorithm**
- **Location**: `app/core/journey_calculator.py`
- **Functions**:
  - `calculate_journey_list()`: For journey lists
  - `calculate_journey_counts()`: For journey counts

###### **Journey List Algorithm** (`calculate_journey_list`)
1. **Group by Device**: Group geofencing rows by device_id
2. **Sort by Time**: Sort each device's rows by entry_event_time
3. **Journey Detection**:
   - Track current facility and previous facility
   - Calculate time difference between exit and next entry
   - **Valid Journey**: Time >= 4 hours (14400 seconds) for different facilities
   - **Same Facility Journey**: Time >= 4 hours + extraJourneyTimeLimit
4. **Journey Generation**:
   - Create journey objects: `{from_facility, to_facility, from_time, to_time, device_id}`
   - Filter by `from_facility` if provided in params
5. **Facility Details**: Build facility metadata dictionary
6. **Output**: JSON with `journies` array and `facilities_details` object

###### **Journey Count Algorithm** (`calculate_journey_counts`)
1. **Same as Journey List** (steps 1-3)
2. **Count Generation**:
   - Create journey keys: `"facilityA||facilityB"`
   - Count journeys by facility pair
   - Calculate totals
3. **Output**: JSON with `counts` dict and `total` number

##### **9.4: Result Formatting**
- **Location**: `app/utils/csv_generator.py` - `format_journey_list_with_csv()`
- **Action**:
  - If > 5 journeys: Generate CSV, return summary + CSV link
  - If ≤ 5 journeys: Return full JSON
- **CSV Generation**: Creates CSV file, stores in database, returns download link
- **Output**: JSON string with journey data (and CSV link if applicable)

#### **Step 10: Decision Point**
- **Location**: `app/core/agent/agent_graph.py` - `_should_format_after_tools()`
- **Action**: Check if journey results are available
- **Decision**: If journey tool result found → Go directly to format_answer (skip agent node)

#### **Step 11: Format Answer** (Optimized for Journey)
- **Location**: `app/core/agent/agent_graph.py` - `_format_answer()`
- **Action**:
  - Extract journey results from state
  - **Token Optimization**: If CSV available or > 5 journeys, use minimal summary
  - Build minimal prompt with journey-specific instructions
  - Call `_invoke_llm_with_fallback()` with `use_tools=False`
- **Journey-Specific Instructions**:
  - Handle journey lists vs counts
  - Mention CSV link if available
  - Handle facility type transitions (M to R, etc.)
- **LLM Processing**:
  - **Model**: Groq (primary) or OpenAI (fallback)
  - **Input**: Minimal prompt with journey results
  - **Output**: Natural language answer
- **Token Usage**: ~500-1,500 tokens (optimized with minimal summary)

#### **Step 12: Response Return**
- **Location**: `app/controllers/chat_controller.py`
- **Action**: Format response with answer, SQL query, journey results, debug info
- **Output**: JSON response to client

### Journey Workflow Diagram

```
Request → LLM Init → Vector Store → Agent Graph
    ↓
Security Guard (if non-admin)
    ↓
System Prompt (1 example only)
    ↓
Tool Binding (journey tools only)
    ↓
LLM Invocation (with journey tools)
    ↓
Journey Tool Execution
    ├─ SQL Execution (fetch geofencing rows)
    ├─ Result Parsing (to dicts)
    ├─ Python Algorithm (journey calculation)
    └─ Result Formatting (with CSV if needed)
    ↓
Decision Point (direct to format_answer)
    ↓
Format Answer (minimal prompt, optimized)
    ↓
Response
```

---

## Models & LLM Usage

### Primary LLM: Groq

**Provider**: Groq API
**Model**: `llama-3.3-70b-versatile`
**Configuration**: 
- Temperature: 0 (deterministic)
- Tool binding: Supported
- Max tokens: Model default

**Usage Locations**:
1. Security guard validation
2. Main agent node (SQL generation)
3. Format answer node (natural language generation)

**API Key**: `GROQ_API_KEY` in `.env`

### Fallback LLM: OpenAI

**Provider**: OpenAI API
**Model**: `gpt-4o`
**Configuration**:
- Temperature: 0 (deterministic)
- Tool binding: Supported
- Max tokens: Model default

**Usage**: Automatic fallback when Groq fails
**API Key**: `API_KEY` or `OPENAI_API_KEY` in `.env`

### Embedding Model: Hugging Face

**Model**: `BAAI/bge-large-en-v1.5` (or `all-MiniLM-L6-v2`)
**Purpose**: Semantic search for examples and business rules
**Usage**:
- Example query retrieval
- Business rules retrieval
- Schema information search

**Configuration**: `EMBEDDING_MODEL_NAME` in `.env`

### LLM Call Summary

| Stage | Model | Purpose | Token Range |
|-------|-------|---------|-------------|
| Security Guard | Groq/OpenAI | Query validation | 200-300 |
| Main Agent | Groq/OpenAI | SQL generation | 3,000-8,000 |
| Format Answer | Groq/OpenAI | Natural language | 500-2,000 |

**Total per Request**: ~4,000-11,000 tokens (varies by question complexity)

---

## Tools & Their Functions

### Regular Tools (Non-Journey Questions)

#### 1. `execute_db_query`
- **Purpose**: Execute SQL queries against PostgreSQL
- **Input**: SQL query string
- **Output**: Query results (formatted string)
- **Features**: Automatic result splitting, CSV generation for large results

#### 2. `count_query`
- **Purpose**: Execute COUNT or aggregation queries
- **Input**: SQL query with COUNT/SUM/AVG/MAX/MIN
- **Output**: Single aggregated value
- **Optimization**: Returns only count/aggregation result

#### 3. `list_query`
- **Purpose**: Execute LIST queries with automatic limiting
- **Input**: SQL query, limit (default: 5)
- **Output**: First N rows + total count + CSV link (if > limit)
- **Features**: Automatic CSV generation for large results

#### 4. `get_few_shot_examples`
- **Purpose**: Retrieve similar example queries from vector store
- **Input**: User question
- **Output**: Similar examples and business rules
- **Usage**: Only when pre-loaded examples are insufficient

#### 5. `get_extra_examples`
- **Purpose**: Alias for `get_few_shot_examples`
- **Usage**: Additional examples beyond pre-loaded ones

#### 6. `get_table_list`
- **Purpose**: Get list of all tables with descriptions
- **Input**: None
- **Output**: Table names, descriptions, important fields
- **Usage**: Last resort when examples don't help

#### 7. `get_table_structure`
- **Purpose**: Get detailed column structure for tables
- **Input**: Comma-separated table names
- **Output**: Column names, data types, constraints
- **Usage**: Last resort when table list doesn't help

### Journey Tools

#### 1. `journey_list_tool`
- **Purpose**: Calculate journey list from geofencing data
- **Input**: 
  - `sql`: SELECT query to fetch geofencing rows
  - `params`: Optional dict with `from_facility`, `extraJourneyTimeLimit`
- **Process**:
  1. Execute SQL to fetch geofencing rows
  2. Parse results to list of dicts
  3. Run Python journey calculation algorithm
  4. Format results with CSV if needed
- **Output**: JSON with `journies` array and `facilities_details` object

#### 2. `journey_count_tool`
- **Purpose**: Calculate journey counts by facility pair
- **Input**: 
  - `sql`: SELECT query to fetch geofencing rows
  - `params`: Optional dict with `extraJourneyTimeLimit`
- **Process**:
  1. Execute SQL to fetch geofencing rows
  2. Parse results to list of dicts
  3. Run Python journey count algorithm
  4. Count journeys by facility pair
- **Output**: JSON with `counts` dict and `total` number

### Tool Selection Logic

**Journey Questions**:
- Tools bound: `journey_list_tool`, `journey_count_tool`, `execute_db_query`
- Token savings: ~1,500 tokens (fewer tool definitions)

**Non-Journey Questions**:
- Tools bound: All regular tools (7 tools)
- Token usage: Standard (~3,000 tokens for tool definitions)

---

## Error Handling & Fallback Logic

### Groq Error Detection

**Location**: `app/core/agent/agent_graph.py` - `_invoke_llm_with_fallback()`

**Error Patterns Detected**:
- `"groq"` in error message (case-insensitive)
- `BadRequestError` error type
- `"tool_use_failed"` in error message
- `"Failed to call a function"` in error message
- `"groq.BadRequestError"` error type

### Fallback Flow

```
1. Try Groq LLM invocation
   ↓
2. Catch exception
   ↓
3. Check if Groq error AND provider == "GROQ"
   ↓
4. Log warning: "Groq API error detected. Falling back to ChatGPT..."
   ↓
5. Get fallback LLM (OpenAI) with same tool binding
   ↓
6. Retry with OpenAI
   ↓
7. If OpenAI succeeds → Return response
   ↓
8. If OpenAI fails → Re-raise original Groq error
```

### Fallback LLM Creation

**Location**: `app/core/agent/agent_graph.py` - `_get_fallback_llm()`

**Process**:
1. Check if fallback LLM already created (lazy initialization)
2. If not, create OpenAI LLM instance via `LLMService.get_fallback_llm_model()`
3. Bind same tools as primary LLM (journey or regular)
4. Cache fallback LLM for reuse

**API Key**: Uses `API_KEY` or `OPENAI_API_KEY` from `.env`

### Error Logging

- **Warning**: Groq error detected (logged before fallback)
- **Info**: Successfully got response from ChatGPT fallback
- **Error**: ChatGPT fallback also failed (both providers failed)

---

## Token Optimizations

### 1. Conditional Tool Binding
- **Journey Questions**: Bind only 3 tools (saves ~1,500 tokens)
- **Non-Journey Questions**: Bind 7 tools (standard)

### 2. Example Pre-loading Optimization
- **Journey Questions**: Load 1 example only (saves ~1,000 tokens)
- **Non-Journey Questions**: Load 2 examples + 1 business rule

### 3. Minimal System Prompt
- **Journey Questions**: Ultra-concise prompt (~1,000-2,000 tokens)
- **Non-Journey Questions**: Standard prompt (~2,000-4,000 tokens)

### 4. Format Answer Optimization
- **With CSV**: Use minimal summary (saves ~40,000 tokens for large results)
- **Without CSV**: Standard formatting

### 5. Security Guard Optimization
- Minimal prompt: ~200-300 tokens
- Only for non-admin users

### 6. Direct Routing After Tools
- Skip agent node when query results available
- Saves ~3,000-8,000 tokens (no full history re-send)

### Total Token Savings (Journey Questions)
- Tool binding: ~1,500 tokens
- Examples: ~1,000 tokens
- System prompt: ~1,000 tokens
- **Total**: ~3,500 tokens saved per journey question

---

## Security & Access Control

### User ID Filtering

**Location**: System prompt and SQL generation

**Rules**:
- **Non-Admin Users**: 
  - MUST join `user_device_assignment` table
  - MUST filter by `ud.user_id = {user_id}`
  - Aggregations only for their own data
- **Admin Users**: 
  - No user_id filtering required
  - Can query across all users

### Security Guard

**Location**: `app/core/agent/agent_graph.py` - `_agent_node()`

**Purpose**: Validate user queries before processing

**Process**:
1. Check if user is admin → Skip validation
2. If non-admin → Call security guard LLM
3. Prompt: "Security Guard. User ID: {user_id}. Respond ONLY: 'ALLOW' or 'BLOCK'"
4. Decision:
   - "ALLOW" → Continue processing
   - "BLOCK" → Return error, stop processing

**Token Usage**: ~200-300 tokens (minimal prompt)

### Restricted Tables

**Location**: `app/core/agent/agent_tools.py` - `_is_restricted_query()`

**Restricted Tables**:
- `admin`
- `user_device_assignment`
- `users`
- `user`

**Rules**:
- Direct SELECT from restricted tables → Blocked
- JOIN with restricted tables → Allowed (for filtering)

### Query Validation

**Location**: `app/core/agent/agent_tools.py` - `check_user_query_restriction()`

**Purpose**: Validate user's natural language question

**Patterns Detected**:
- Requests for admin data
- Requests for user assignment data
- Generic requests for raw table data

---

## Data Flow Diagrams

### Non-Journey Question Flow

```
User Question
    ↓
FastAPI Endpoint
    ↓
Chat Controller
    ├─ LLM Service (Groq/OpenAI)
    ├─ Vector Store Service (Hugging Face)
    └─ SQLAgentGraph
        ↓
Security Guard (if non-admin)
    ↓
System Prompt Generation
    ├─ Vector Search (2 examples + 1 rule)
    └─ Embed Examples in Prompt
        ↓
Tool Binding (Regular Tools)
    ↓
Agent Node
    ├─ LLM Invocation (with tools)
    └─ Tool Calls Generated
        ↓
Tool Execution
    ├─ execute_db_query / count_query / list_query
    └─ SQL Execution → PostgreSQL
        ↓
Decision Point
    ↓
Format Answer Node
    ├─ Minimal Prompt (question + results)
    └─ LLM Invocation (no tools)
        ↓
Final Answer
    ↓
Response to User
```

### Journey Question Flow

```
User Question
    ↓
FastAPI Endpoint
    ↓
Chat Controller
    ├─ LLM Service (Groq/OpenAI)
    ├─ Vector Store Service (Hugging Face)
    └─ SQLAgentGraph
        ↓
Security Guard (if non-admin)
    ↓
System Prompt Generation
    ├─ Vector Search (1 example only)
    └─ Embed Example in Prompt
        ↓
Tool Binding (Journey Tools Only)
    ↓
Agent Node
    ├─ LLM Invocation (with journey tools)
    └─ journey_list_tool / journey_count_tool Call Generated
        ↓
Journey Tool Execution
    ├─ SQL Execution → PostgreSQL (geofencing rows)
    ├─ Result Parsing (to dicts)
    ├─ Python Algorithm (journey calculation)
    └─ Result Formatting (with CSV if needed)
        ↓
Decision Point (direct to format_answer)
    ↓
Format Answer Node
    ├─ Minimal Prompt (question + journey results)
    └─ LLM Invocation (no tools)
        ↓
Final Answer
    ↓
Response to User
```

### Fallback Logic Flow

```
LLM Invocation Request
    ↓
Try Groq LLM
    ↓
Success? ──Yes──→ Return Response
    │
    No
    ↓
Check Error Type
    ↓
Groq Error? ──No──→ Re-raise Error
    │
    Yes
    ↓
Log Warning: "Falling back to ChatGPT..."
    ↓
Get Fallback LLM (OpenAI)
    ↓
Retry with OpenAI
    ↓
Success? ──Yes──→ Return Response
    │
    No
    ↓
Log Error: "ChatGPT fallback also failed"
    ↓
Re-raise Original Groq Error
```

---

## Summary

### Key Differences: Journey vs Non-Journey

| Aspect | Journey Questions | Non-Journey Questions |
|--------|------------------|---------------------|
| **Detection** | Keyword-based ("journey", "movement", etc.) | Default |
| **Tools** | 3 tools (journey_list, journey_count, execute_db_query) | 7 tools (all regular tools) |
| **Examples** | 1 example (SQL structure) | 2 examples + 1 business rule |
| **Processing** | SQL → Python algorithm → JSON | SQL → Direct results |
| **Token Usage** | ~3,500 tokens saved | Standard |
| **Complexity** | High (Python algorithm) | Low (direct SQL) |

### System Strengths

1. **Intelligent Routing**: Automatic detection and routing based on question type
2. **Token Optimization**: Multiple optimizations reduce token usage by ~30-40%
3. **Error Resilience**: Automatic fallback to OpenAI when Groq fails
4. **Security**: Multi-layer security (guard, user filtering, restricted tables)
5. **Scalability**: Efficient handling of large result sets with CSV generation
6. **Flexibility**: Supports both simple queries and complex journey calculations

### Technology Highlights

- **LangGraph**: Sophisticated workflow orchestration
- **pgvector**: Efficient semantic search
- **Hugging Face**: Cost-effective embeddings
- **Groq + OpenAI**: Dual provider support with fallback
- **Python Algorithms**: Complex journey calculations

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Maintained By**: Shipmentia AI Team
