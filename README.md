# Ship RAG AI - Agentic SQL Chat with LangGraph

A modern AI-powered application that converts natural language questions into SQL queries using **LangGraph** and **Agentic RAG**. This project re-architects the original `ship-ai` to use:

- **LangGraph** instead of LangChain for agent orchestration
- **FAISS vector store** for storing example queries and business rules
- **Conditional tool calling** - LLM decides when to retrieve examples vs. execute queries
- **Reduced token consumption** by moving long prompts to vector store

## 🎯 Key Features

- **Multi-Provider LLM Support**: Supports both OpenAI and Groq (configurable via environment variable)
- **Free Embeddings**: Uses Hugging Face embeddings (no API key required!)
- **Agentic RAG**: LLM conditionally retrieves examples from FAISS when needed
- **Hybrid Storage**: FAISS for examples, PostgreSQL for data execution
- **Token Optimization**: Long prompts moved to vector store, retrieved only when needed
- **LangGraph Workflow**: State-based agent with conditional tool calling
- **PostgreSQL Integration**: Full SQL query generation and execution
- **Vector Store Management**: API endpoint to reload vector stores when examples are updated

## 🏗️ Architecture

```
User Question
    ↓
FastAPI /chat endpoint
    ↓
LangGraph Agent
    ↓
[Conditional Decision]
    ├─→ get_few_shot_examples (FAISS) → Retrieve examples
    └─→ execute_db_query (PostgreSQL) → Execute SQL
    ↓
Natural Language Answer + SQL Query
```

## 📁 Project Structure

```
ship-RAG-ai/
├── main.py                 # FastAPI application
├── agent_graph.py          # LangGraph agent implementation
├── agent_tools.py          # Tool definitions (FAISS + PostgreSQL)
├── vector_store.py         # FAISS vector store manager
├── examples_data.py        # Example queries and business rules
├── prompts.py              # Concise system prompts
├── db.py                   # PostgreSQL connection
├── llm_model.py            # LLM model wrapper
├── models.py               # Pydantic models
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## 🚀 Setup & Installation

### Prerequisites

- Python 3.12+
- PostgreSQL database
- OpenAI API key **OR** Groq API key (choose one for LLM)
- **No API key needed for embeddings!** Uses free Hugging Face models

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables

Create a `.env` file in the project root:

```env
# Database Configuration
HOST="your_postgresql_host"
PORT="5432"
DATABASE="your_database_name"
USER="your_postgresql_username"
PASSWORD="your_postgresql_password"
SSL_MODE="prefer"

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
# You can use other models like:
# - "sentence-transformers/all-mpnet-base-v2" (better quality, slower)
# - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (multilingual)
EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
```

### Step 3: Start the Application

```bash
# Using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 3009

# Or with auto-reload for development
uvicorn main:app --host 0.0.0.0 --port 3009 --reload
```

The API will be available at: `http://localhost:3009`

### Step 4: Switching LLM Providers

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
# Note: OpenAI API key is still required for embeddings (vector store)
API_KEY="your_openai_api_key"
```

**Available Groq Models:**
- `llama-3.3-70b-versatile` (default) - Best for complex queries
- `llama-3.1-8b-versatile` - Faster, lighter model
- `mixtral-8x7b-32768` - Alternative option

**Embeddings:** The application uses **Hugging Face embeddings** (no API key required!) for the vector store. The default model is `sentence-transformers/all-MiniLM-L6-v2`, which provides good quality semantic search without any API costs.

You can customize the model by modifying `llm_model.py` or passing a `model` parameter to `get_llm_model()`.

## 📖 Usage

### API Endpoint: `POST /chat`

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
  "sql_query": "SELECT COUNT(*) as device_count FROM Device_Details_Table D INNER JOIN user_device_assignment ud ON D.SNo = ud.device_id LEFT JOIN incoming_message_history_K ik ON D.latest_incoming_message_id = ik.SNo WHERE ud.user_id = '27' AND ik.temperature > 10",
  "debug": "Received 0 messages in history"
}
```

### API Endpoint: `POST /reload-vector-store`

Reload the vector store from `examples_data.py`. Useful when:
- You've updated `examples_data.py` with new examples
- You want to switch embedding models
- Vector stores need to be rebuilt

**Request:**
```bash
curl -X POST http://localhost:3009/reload-vector-store
```

**Response:**
```json
{
  "status": "success",
  "message": "Vector stores reloaded successfully",
  "examples_count": 10,
  "extra_prompts_count": 12
}
```

### Interactive API Documentation

- **Swagger UI**: `http://localhost:3009/docs`
- **ReDoc**: `http://localhost:3009/redoc`

## 🔄 How It Works

### 1. Vector Store Initialization

On first startup, the application:
- Loads example queries from `examples_data.py`
- Loads business rules and schema info
- Uses Hugging Face embeddings (no API key needed!)
- Creates FAISS indexes (saved to `./faiss_index/`)
- On subsequent startups, loads existing indexes

**To reload vector stores after updating `examples_data.py`:**
```bash
POST /reload-vector-store
```
This clears existing indexes and rebuilds them with the latest data.

### 2. Agent Workflow

1. **User asks a question** → FastAPI receives request
2. **LangGraph agent decides** → Should I retrieve examples?
3. **Conditional tool calling**:
   - If complex → Call `get_few_shot_examples` (FAISS search)
   - Generate SQL → Call `execute_db_query` (PostgreSQL)
4. **Format answer** → Natural language response

### 3. Token Optimization

- **Before**: ~15,000-20,000 chars sent to LLM every call (with examples)
- **After**: ~3,000-5,000 chars base prompt, examples retrieved only when needed

## 🛠️ Key Components

### `agent_graph.py`
- LangGraph state machine
- Conditional tool calling logic
- State management for conversation flow

### `agent_tools.py`
- `get_few_shot_examples`: FAISS semantic search
- `execute_db_query`: PostgreSQL query execution

### `vector_store.py`
- FAISS index management
- Startup initialization with existence check
- Separate indexes for examples and business rules

### `examples_data.py`
- All example queries (moved from prompt)
- Business rules and schema descriptions
- Metadata for better retrieval

## 🔒 Security

- User-based data access control (same as original)
- Token-based API authentication
- SQL injection prevention via parameterized queries
- Environment variable management

## 📊 Differences from Original ship-ai

| Feature | ship-ai | ship-RAG-ai |
|---------|---------|------------|
| Framework | LangChain | LangGraph |
| Examples | In prompt | FAISS vector store |
| Token Usage | High (~20k chars) | Optimized (~5k base) |
| Tool Calling | Fixed sequence | Conditional |
| State Management | Chain-based | Graph-based |

## 🐛 Troubleshooting

### FAISS Index Not Found
- First run creates indexes automatically
- Check `./faiss_index/` directory exists
- Ensure write permissions

### Database Connection Issues
- Verify `.env` file has correct credentials
- Check PostgreSQL server is running
- Test connection: `python -c "from db import sync_engine; print('OK')"`

### OpenAI API Errors
- Verify `API_KEY` in `.env`
- Check API quota/credits
- Ensure network connectivity

## 📝 License

[Add your license information here]

## 🙏 Acknowledgments

- Built on LangGraph for agent orchestration
- Uses FAISS for efficient vector search
- Original architecture from `ship-ai` project

