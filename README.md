# Ship RAG AI - Agentic SQL Chat with LangGraph

A modern AI-powered application that converts natural language questions into SQL queries using **LangGraph** and **Agentic RAG**. This project re-architects the original `ship-ai` to use:

- **LangGraph** instead of LangChain for agent orchestration
- **PostgreSQL pgvector** for storing example queries and business rules
- **Conditional tool calling** - LLM decides when to retrieve examples vs. execute queries
- **Reduced token consumption** by moving long prompts to vector store

## 🎯 Key Features

- **Multi-Provider LLM Support**: Supports both OpenAI and Groq (configurable via environment variable)
- **Free Embeddings**: Uses Hugging Face embeddings (no API key required!)
- **Agentic RAG**: LLM conditionally retrieves examples from PostgreSQL vector store when needed
- **Hybrid Storage**: PostgreSQL pgvector for examples, PostgreSQL for data execution
- **Token Optimization**: Long prompts moved to vector store, retrieved only when needed
- **LangGraph Workflow**: State-based agent with conditional tool calling
- **PostgreSQL Integration**: Full SQL query generation and execution
- **Vector Store Management**: Scripts and APIs to load data and generate embeddings

## 🏗️ Architecture

```
User Question
    ↓
FastAPI /chat endpoint
    ↓
LangGraph Agent
    ↓
[Conditional Decision]
    ├─→ get_few_shot_examples (PostgreSQL pgvector) → Retrieve examples
    └─→ execute_db_query (PostgreSQL) → Execute SQL
    ↓
Natural Language Answer + SQL Query
```

## 📁 Project Structure

```
ship-RAG-ai/
├── app/                         # Main application package
│   ├── __init__.py
│   ├── main.py                  # FastAPI application initialization
│   │
│   ├── api/                     # API routes
│   │   ├── __init__.py          # Router aggregation
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── chat.py          # Chat endpoint routes
│   │       ├── embeddings.py    # Embedding generation routes
│   │       └── health.py        # Health check routes
│   │
│   ├── config/                  # Configuration
│   │   ├── __init__.py
│   │   ├── settings.py          # Environment variables & settings
│   │   └── database.py          # Database connection management
│   │
│   ├── controllers/             # Business logic controllers
│   │   ├── __init__.py
│   │   ├── chat_controller.py   # Chat endpoint logic
│   │   ├── embeddings_controller.py  # Embedding generation logic
│   │   └── health_controller.py # Health check logic
│   │
│   ├── core/                    # Core agent logic
│   │   ├── __init__.py
│   │   ├── prompts.py           # System prompts
│   │   └── agent/
│   │       ├── __init__.py
│   │       ├── agent_graph.py   # LangGraph agent implementation
│   │       └── agent_tools.py   # Tool definitions (pgvector + PostgreSQL)
│   │
│   ├── models/                  # Data models
│   │   ├── __init__.py
│   │   └── schemas.py           # Pydantic request/response models
│   │
│   ├── services/                # Business services
│   │   ├── __init__.py
│   │   ├── llm_service.py       # LLM model wrapper
│   │   └── vector_store_service.py  # PostgreSQL pgvector store manager
│   │
│   ├── helpers/                  # Helper utilities
│   │   ├── __init__.py
│   │   └── validators.py        # Validation helpers
│   │
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       └── logger.py            # Logging utility
│
├── scripts/                     # Utility scripts
│   ├── examples_data.py         # Example queries and business rules
│   └── load_examples_data.py    # Script to load data into PostgreSQL
│
├── logs/                        # Application logs (auto-generated)
├── run.py                       # Application runner script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── MIGRATION_NOTES.md           # Migration documentation
└── DATABASE_CONNECTIONS.md      # Database connection documentation
```

## 🚀 Setup & Installation

### Prerequisites

- Python 3.12+
- PostgreSQL database with **pgvector extension**
- OpenAI API key **OR** Groq API key (choose one for LLM)
- **No API key needed for embeddings!** Uses free Hugging Face models

### Step 1: Install PostgreSQL pgvector Extension

First, ensure the pgvector extension is installed in your PostgreSQL database:

```sql
-- Connect to your database
\c your_database_name

-- Install pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
```

### Step 2: Create Vector Store Tables

Create the required tables in your PostgreSQL database:

```sql
-- Table for Few-Shot Examples (Questions & SQL)
CREATE TABLE IF NOT EXISTS ai_vector_examples (
    id BIGSERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    sql_query TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    minilm_embedding VECTOR(384),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table for Extra Prompt Data (Context & Notes)
CREATE TABLE IF NOT EXISTS ai_vector_extra_prompts (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    note_type VARCHAR(50), 
    metadata JSONB DEFAULT '{}',
    minilm_embedding VECTOR(384),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- HNSW Indexes for fast similarity search
CREATE INDEX IF NOT EXISTS idx_ai_examples_minilm ON ai_vector_examples 
USING hnsw (minilm_embedding vector_l2_ops);

CREATE INDEX IF NOT EXISTS idx_ai_extra_minilm ON ai_vector_extra_prompts 
USING hnsw (minilm_embedding vector_l2_ops);
```

### Step 3: Install Python Dependencies

1. Install uv
The user must have uv installed on their system.

Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh

2. Clone and Setup
Once they have your code, they only need to run one command to install everything:
uv sync

3. Run the Project
Since you added the start script to your pyproject.toml, they just run:
uv run start


  ---OR---

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```env
# Database Configuration - Read-only user (for health, chat endpoints)
HOST="your_postgresql_host"
PORT="5432"
DATABASE="your_database_name"
USER="your_readonly_username"
PASSWORD="your_readonly_password"
SSL_MODE="prefer"

# Database Configuration - Update user (for embedding generation routes)
# Optional but recommended for security
UPDATE_USER="your_update_username"
UPDATE_PASSWORD="your_update_password"

# LLM Provider Configuration
# Set LLM_PROVIDER to "OPENAI" or "GROQ" (default: "OPENAI")
LLM_PROVIDER="OPENAI"

# OpenAI API Configuration (required if LLM_PROVIDER=OPENAI)
API_KEY="your_openai_api_key"
OPENAI_API_KEY="your_openai_api_key"

# Groq API Configuration (required if LLM_PROVIDER=GROQ)
# Get your API key from: https://console.groq.com/
GROQ_API_KEY="your_groq_api_key"

# Embedding Model Configuration (optional)
# Uses Hugging Face embeddings - no API key required!
# Default: "sentence-transformers/all-MiniLM-L6-v2"
# This model produces 384-dimensional vectors (matching VECTOR(384) in PostgreSQL)
EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
```

**Note:** The application uses two separate database connections for security:
- **Read-only connection** (`USER`/`PASSWORD`): Used for health checks and chat endpoints
- **Update connection** (`UPDATE_USER`/`UPDATE_PASSWORD`): Used for embedding generation routes

If `UPDATE_USER` and `UPDATE_PASSWORD` are not set, the application will fall back to using the read-only connection (not recommended for production). See `DATABASE_CONNECTIONS.md` for more details.

### Step 5: Load Data into PostgreSQL Tables

Load example data from `scripts/examples_data.py` into PostgreSQL tables:

```bash
python scripts/load_examples_data.py
```

This script will:
- Load `SAMPLE_EXAMPLES` into `ai_vector_examples` table
- Load `EXTRA_PROMPT_DATA` into `ai_vector_extra_prompts` table
- Update existing records if they already exist (by question/content)
- **Note**: This does NOT generate embeddings - use API endpoints for that

### Step 6: Generate Embeddings

After loading data, generate embeddings using the API endpoints:

**For all records with NULL embeddings:**
```bash
# Examples table
curl -X POST http://localhost:3009/generate-embeddings-examples \
  -H "Content-Type: application/json" \
  -d '{}'

# Extra prompts table
curl -X POST http://localhost:3009/generate-embeddings-extra-prompts \
  -H "Content-Type: application/json" \
  -d '{}'
```

**For a specific record:**
```bash
# Examples table - ID 5
curl -X POST http://localhost:3009/generate-embeddings-examples \
  -H "Content-Type: application/json" \
  -d '{"id": 5}'

# Extra prompts table - ID 3
curl -X POST http://localhost:3009/generate-embeddings-extra-prompts \
  -H "Content-Type: application/json" \
  -d '{"id": 3}'
```

### Step 7: Start the Application

```bash
# Using UV (recommended)
uv run run.py

# Using the run.py script
python run.py

# Or using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 3009 --reload
```

The API will be available at: `http://localhost:3009`

### Step 8: Switching LLM Providers

The application supports both **OpenAI** and **Groq** LLM providers. Switch between them using the `LLM_PROVIDER` environment variable:

**Using OpenAI (default):**
```env
LLM_PROVIDER="OPENAI"
API_KEY="your_openai_api_key"
```

**Using Groq:**
```env
LLM_PROVIDER="GROQ"
GROQ_API_KEY="your_groq_api_key"
```

**Available Groq Models:**
- `llama-3.3-70b-versatile` (default) - Best for complex queries
- `llama-3.1-8b-versatile` - Faster, lighter model
- `mixtral-8x7b-32768` - Alternative option

**Embeddings:** The application uses **Hugging Face embeddings** (no API key required!) for the vector store. The default model is `sentence-transformers/all-MiniLM-L6-v2`, which produces 384-dimensional vectors matching the `VECTOR(384)` column type in PostgreSQL.

## 📖 Usage

### API Endpoint: `POST /chat`

Main endpoint for natural language to SQL conversion.

**Request:**
```json
{
  "token_id": "Test123",
  "question": "Count devices that have current temperature more than 10 degrees",
  "user_id": "27"
}
```

**Response:**
```json
{
  "token_id": "Test123",
  "answer": "There are 15 devices with current temperature above 10 degrees Celsius.",
  "sql_query": "SELECT COUNT(*) as device_count FROM device_current_data cd JOIN user_device_assignment ud ON cd.device_id = ud.device WHERE ud.user_id = '27' AND cd.temperature > 10",
  "debug": {
    "question": "...",
    "total_messages": 5,
    "token_usage": {
      "input_tokens": 1234,
      "output_tokens": 567,
      "total_tokens": 1801
    }
  }
}
```

### API Endpoint: `POST /generate-embeddings-examples`

Generate embeddings for records in `ai_vector_examples` table.

**Request (all NULL embeddings):**
```bash
curl -X POST http://localhost:3009/generate-embeddings-examples \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Request (specific ID):**
```bash
curl -X POST http://localhost:3009/generate-embeddings-examples \
  -H "Content-Type: application/json" \
  -d '{"id": 5}'
```

**Response:**
```json
{
  "status": "success",
  "message": "Successfully processed 10 record(s)",
  "processed_count": 10,
  "updated_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  "errors": null
}
```

**Behavior:**
- **Without `id`**: Processes all records with NULL/empty embeddings
- **With `id`**: Updates embedding for that specific record (replaces if exists)

### API Endpoint: `POST /generate-embeddings-extra-prompts`

Generate embeddings for records in `ai_vector_extra_prompts` table.

**Request (all NULL embeddings):**
```bash
curl -X POST http://localhost:3009/generate-embeddings-extra-prompts \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Request (specific ID):**
```bash
curl -X POST http://localhost:3009/generate-embeddings-extra-prompts \
  -H "Content-Type: application/json" \
  -d '{"id": 3}'
```

**Response:**
```json
{
  "status": "success",
  "message": "Successfully processed 12 record(s)",
  "processed_count": 12,
  "updated_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
  "errors": null
}
```

**Behavior:**
- **Without `id`**: Processes all records with NULL/empty embeddings
- **With `id`**: Updates embedding for that specific record (replaces if exists)

### API Endpoint: `POST /reload-vector-store`

Verify PostgreSQL vector store tables and return record counts.

**Request:**
```bash
curl -X POST http://localhost:3009/reload-vector-store
```

**Response:**
```json
{
  "status": "success",
  "message": "Vector stores verified successfully",
  "examples_count": 10,
  "extra_prompts_count": 12
}
```

### Script: `scripts/load_examples_data.py`

Load data from `scripts/examples_data.py` into PostgreSQL tables.

**Usage:**
```bash
python scripts/load_examples_data.py
```

**What it does:**
- Loads `SAMPLE_EXAMPLES` into `ai_vector_examples` table
- Loads `EXTRA_PROMPT_DATA` into `ai_vector_extra_prompts` table
- Checks for existing records (by question/content) and updates or inserts accordingly
- Uses the update database connection (`UPDATE_USER`/`UPDATE_PASSWORD`) for write operations
- Does NOT generate embeddings (use API endpoints for that)

**Output:**
```
🚀 LOADING EXAMPLE DATA INTO POSTGRESQL
================================================================================
📥 LOADING EXAMPLES DATA INTO ai_vector_examples
================================================================================
   ✓ Inserted: Can you provide the number of 'manufacturer to retailer'...
   ✓ Inserted: Count devices that have current temperature more than...
   ...

✅ Examples data loaded successfully!
   Inserted: 10 records
   Updated: 0 records
```

### Interactive API Documentation

- **Swagger UI**: `http://localhost:3009/docs`
- **ReDoc**: `http://localhost:3009/redoc`

## 🔄 How It Works

### 1. Vector Store Setup

**Initial Setup:**
1. Create PostgreSQL tables with pgvector extension
2. Load data using `load_examples_data.py` script
3. Generate embeddings using API endpoints
4. Application uses PostgreSQL for vector similarity search

**Data Flow:**
- Example queries stored in `ai_vector_examples` table
- Business rules stored in `ai_vector_extra_prompts` table
- Embeddings generated using Hugging Face model (384 dimensions)
- HNSW indexes for fast similarity search

### 2. Agent Workflow

1. **User asks a question** → FastAPI receives request
2. **LangGraph agent decides** → Should I retrieve examples?
3. **Conditional tool calling**:
   - If complex → Call `get_few_shot_examples` (PostgreSQL pgvector search)
   - Generate SQL → Call `execute_db_query` (PostgreSQL)
4. **Format answer** → Natural language response

### 3. Token Optimization

- **Before**: ~15,000-20,000 chars sent to LLM every call (with examples)
- **After**: ~3,000-5,000 chars base prompt, examples retrieved only when needed

## 🛠️ Key Components

### `app/core/agent/agent_graph.py`
- LangGraph state machine
- Conditional tool calling logic
- State management for conversation flow

### `app/core/agent/agent_tools.py`
- `get_few_shot_examples`: PostgreSQL pgvector semantic search
- `execute_db_query`: PostgreSQL query execution

### `app/services/vector_store_service.py`
- PostgreSQL pgvector store manager
- Embedding generation using Hugging Face
- Vector similarity search using L2 distance
- Uses read-only database connection

### `app/config/database.py`
- Manages two separate database connections:
  - `sync_engine`: Read-only connection (for health, chat endpoints)
  - `sync_engine_update`: Update connection (for embedding generation)
- Connection pooling and SSL support

### `app/config/settings.py`
- Centralized environment variable loading using Pydantic Settings
- Validates database and LLM configuration
- Supports fallback values for backward compatibility

### `scripts/load_examples_data.py`
- Script to load data from `scripts/examples_data.py` into PostgreSQL
- Handles insert/update logic using update database connection
- Does not generate embeddings

### `scripts/examples_data.py`
- All example queries (reference only)
- Business rules and schema descriptions
- Metadata for better retrieval
- **Note**: Data should be loaded into PostgreSQL tables

## 🔒 Security

- User-based data access control (same as original)
- Token-based API authentication
- SQL injection prevention via parameterized queries
- Environment variable management

## 📊 Differences from Original ship-ai

| Feature | ship-ai | ship-RAG-ai |
|---------|---------|------------|
| Framework | LangChain | LangGraph |
| Vector Store | N/A | PostgreSQL pgvector |
| Examples | In prompt | PostgreSQL vector store |
| Token Usage | High (~20k chars) | Optimized (~5k base) |
| Tool Calling | Fixed sequence | Conditional |
| State Management | Chain-based | Graph-based |

## 🐛 Troubleshooting

### PostgreSQL pgvector Extension Not Found

```sql
-- Check if extension is installed
SELECT * FROM pg_extension WHERE extname = 'vector';

-- If not installed, run:
CREATE EXTENSION IF NOT EXISTS vector;
```

### Vector Store Tables Not Found

```sql
-- Check if tables exist
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN ('ai_vector_examples', 'ai_vector_extra_prompts');

-- If missing, create them (see Step 2 in Setup)
```

### No Embeddings Generated

- Ensure data is loaded: `python load_examples_data.py`
- Generate embeddings using API endpoints
- Check that `minilm_embedding` column is not NULL after generation

### Database Connection Issues

- Verify `.env` file has correct credentials
- Check PostgreSQL server is running
- Test connection: `python -c "from app.config.database import sync_engine; print('OK')"`
- Ensure pgvector extension is installed
- Verify both `USER`/`PASSWORD` and `UPDATE_USER`/`UPDATE_PASSWORD` are set correctly

### Embedding Generation Errors

- Verify Hugging Face model is accessible
- Check that embedding model produces 384-dimensional vectors
- Ensure PostgreSQL connection is working
- Check logs for specific error messages

### OpenAI/Groq API Errors

- Verify `API_KEY` or `GROQ_API_KEY` in `.env`
- Check API quota/credits
- Ensure network connectivity

## 📝 Workflow Summary

### Complete Setup Workflow

1. **Install PostgreSQL pgvector extension**
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

2. **Create tables** (see Step 2 in Setup)

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure `.env` file** (include `UPDATE_USER` and `UPDATE_PASSWORD` for security)

5. **Load data into PostgreSQL**
   ```bash
   python scripts/load_examples_data.py
   ```

6. **Generate embeddings**
   ```bash
   # Start the application first
   python run.py
   
   # Then generate embeddings (in another terminal)
   curl -X POST http://localhost:3009/generate-embeddings-examples -H "Content-Type: application/json" -d '{}'
   curl -X POST http://localhost:3009/generate-embeddings-extra-prompts -H "Content-Type: application/json" -d '{}'
   ```

7. **Start using the API!**

### Updating Data

1. **Update `scripts/examples_data.py`** with new examples
2. **Reload data**: `python scripts/load_examples_data.py`
3. **Regenerate embeddings** for new/updated records:
   ```bash
   # For all NULL embeddings
   curl -X POST http://localhost:3009/generate-embeddings-examples -H "Content-Type: application/json" -d '{}'
   
   # Or for specific ID
   curl -X POST http://localhost:3009/generate-embeddings-examples -H "Content-Type: application/json" -d '{"id": 15}'
   ```

## 📝 License

[Add your license information here]

## 🙏 Acknowledgments

- Built on LangGraph for agent orchestration
- Uses PostgreSQL pgvector for efficient vector search
- Original architecture from `ship-ai` project
