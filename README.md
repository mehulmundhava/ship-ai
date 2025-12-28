# Ship RAG AI - Agentic SQL Chat with LangGraph

A modern AI-powered application that converts natural language questions into SQL queries using **LangGraph** and **Agentic RAG**. This project re-architects the original `ship-ai` to use:

- **LangGraph** instead of LangChain for agent orchestration
- **FAISS vector store** for storing example queries and business rules
- **Conditional tool calling** - LLM decides when to retrieve examples vs. execute queries
- **Reduced token consumption** by moving long prompts to vector store

## 🎯 Key Features

- **Agentic RAG**: LLM conditionally retrieves examples from FAISS when needed
- **Hybrid Storage**: FAISS for examples, PostgreSQL for data execution
- **Token Optimization**: Long prompts moved to vector store, retrieved only when needed
- **LangGraph Workflow**: State-based agent with conditional tool calling
- **PostgreSQL Integration**: Full SQL query generation and execution

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
- OpenAI API key

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

# OpenAI API Configuration
API_KEY="your_openai_api_key"
OPENAI_API_KEY="your_openai_api_key"
```

### Step 3: Start the Application

```bash
# Using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 3009

# Or with auto-reload for development
uvicorn main:app --host 0.0.0.0 --port 3009 --reload
```

The API will be available at: `http://localhost:3009`

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

### Interactive API Documentation

- **Swagger UI**: `http://localhost:3009/docs`
- **ReDoc**: `http://localhost:3009/redoc`

## 🔄 How It Works

### 1. Vector Store Initialization

On first startup, the application:
- Loads example queries from `examples_data.py`
- Loads business rules and schema info
- Creates FAISS indexes (saved to `./faiss_index/`)
- On subsequent startups, loads existing indexes

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

