# Dual Database Connection Setup

## Overview

The application uses two separate database connections for enhanced security:

1. **Read-Only Connection** (`sync_engine`) - Uses `USER` and `PASSWORD` from `.env`
   - Used for: Health checks, Chat endpoints (query execution)
   - Purpose: Prevents accidental data modification in normal operations

2. **Update Connection** (`sync_engine_update`) - Uses `UPDATE_USER` and `UPDATE_PASSWORD` from `.env`
   - Used for: Embedding generation routes (UPDATE operations)
   - Purpose: Allows controlled write access only where needed

## Configuration

Add these variables to your `.env` file:

```env
# Read-only user (required)
USER="readonly_username"
PASSWORD="readonly_password"

# Update user (optional but recommended)
UPDATE_USER="update_username"
UPDATE_PASSWORD="update_password"
```

**Note:** If `UPDATE_USER` and `UPDATE_PASSWORD` are not set, the application will fall back to using the read-only connection for embedding routes (not recommended for production).

## Usage in Code

### Read-Only Connection

**Import:**
```python
from app.config.database import sync_engine
```

**Used in:**
- `app/controllers/health_controller.py` - Health check endpoint (`/health`)
- `app/controllers/chat_controller.py` - Chat endpoint (`/chat`) - SQL query execution
- `app/services/vector_store_service.py` - Vector search operations (read-only)
- `app/main.py` - SQLDatabase wrapper initialization for LangChain agent

**Example Usage:**
```python
from app.config.database import sync_engine
from sqlalchemy import text

with sync_engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM devices"))
    count = result.fetchone()[0]
```

### Update Connection

**Import:**
```python
from app.config.database import sync_engine_update
```

**Used in:**
- `app/controllers/embeddings_controller.py` - Embedding generation routes
  - `POST /generate-embeddings-examples` - Updates `ai_vector_examples` table
  - `POST /generate-embeddings-extra-prompts` - Updates `ai_vector_extra_prompts` table
- `scripts/load_examples_data.py` - Data loading script (INSERT/UPDATE operations)

**Example Usage:**
```python
from app.config.database import sync_engine_update
from sqlalchemy import text

with sync_engine_update.connect() as conn:
    conn.execute(text("UPDATE ai_vector_examples SET minilm_embedding = :embedding WHERE id = :id"), 
                 {"embedding": embedding_vector, "id": record_id})
    conn.commit()
```

## Security Benefits

1. **Principle of Least Privilege**: Regular endpoints only have read access
2. **Attack Surface Reduction**: Even if a read-only connection is compromised, data cannot be modified
3. **Controlled Updates**: Only specific routes (embedding generation) can perform writes
4. **Audit Trail**: Update operations are clearly separated and can be monitored

## Database Setup Recommendations

### PostgreSQL User Creation

```sql
-- Create read-only user
CREATE USER readonly_user WITH PASSWORD 'readonly_password';
GRANT CONNECT ON DATABASE your_database TO readonly_user;
GRANT USAGE ON SCHEMA public TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO readonly_user;

-- Create update user (for embeddings)
CREATE USER update_user WITH PASSWORD 'update_password';
GRANT CONNECT ON DATABASE your_database TO update_user;
GRANT USAGE ON SCHEMA public TO update_user;
GRANT SELECT, UPDATE ON ALL TABLES IN SCHEMA public TO update_user;
GRANT SELECT, UPDATE ON ai_vector_examples TO update_user;
GRANT SELECT, UPDATE ON ai_vector_extra_prompts TO update_user;
-- Grant only necessary permissions, not full write access
```

## Connection Verification

When the application starts, you should see output like:

```
📄 Config: Loaded .env from C:\xampp\htdocs\ship-ai-dev\ship-RAG-ai\.env
🔗 Read-Only Database Connection URL: postgresql+psycopg2://readonly_user:***@host:port/database
✅ Read-Only Database SQLAlchemy engine created successfully
🔗 Update Database Connection URL: postgresql+psycopg2://update_user:***@host:port/database
✅ Update Database SQLAlchemy engine created successfully
✅ Using separate UPDATE_USER connection for embedding routes
🚀 Starting application...
✅ LLM model initialized
✅ Vector stores initialized
✅ PostgreSQL database connection initialized
✅ Database connection test successful
✅ Application ready to serve requests
```

If `UPDATE_USER` is not configured:
```
⚠️  UPDATE_USER not configured - using readonly connection for embedding routes (not recommended)
```

## Implementation Details

### Connection Creation

The dual connection setup is implemented in `app/config/database.py`:

1. **Read-Only Engine** (`sync_engine`):
   - Created using `USER` and `PASSWORD` from environment variables
   - Used for all read operations

2. **Update Engine** (`sync_engine_update`):
   - Created using `UPDATE_USER` and `UPDATE_PASSWORD` if provided
   - Falls back to read-only connection if update credentials are not set
   - Used only for write operations (embedding generation, data loading)

### Connection Pooling

Both engines use SQLAlchemy connection pooling:
- `pool_size=10`: Maintains 10 connections in the pool
- `max_overflow=20`: Allows up to 20 additional connections
- `pool_pre_ping=True`: Validates connections before use

### SSL Support

Both connections support SSL modes:
- `prefer` (default): Try SSL, fallback to non-SSL
- `require`: Require SSL connection
- `disable`: Disable SSL
- Other PostgreSQL SSL modes are also supported

