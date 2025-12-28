"""
Database Connection Module for PostgreSQL

This module handles all database connection logic for the application.
It loads environment variables from .env file and creates a SQLAlchemy engine
for PostgreSQL database connections with connection pooling and SSL support.
"""

import os
import urllib
from pathlib import Path
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables from .env file
# Try multiple locations to find .env file
base_path = Path(__file__).parent
env_paths = [
    base_path / ".env",           # Current directory (ship-RAG-ai/.env)
    base_path.parent / ".env",    # Parent directory (ship-ai-dev/.env)
    base_path.parent / "ship-ai" / ".env",  # Original ship-ai/.env
]

env_path = None
for path in env_paths:
    if path.exists():
        env_path = path
        print(f"📄 Loading .env file from: {env_path}")
        break

# Load .env file explicitly, or use default behavior
if env_path:
    load_dotenv(dotenv_path=env_path, override=True)
else:
    # Fallback to default behavior (current working directory)
    print("⚠️  No .env file found in expected locations, using default load_dotenv()")
    load_dotenv(override=True)

# ============================================================================
# ENVIRONMENT VARIABLE LOADING
# ============================================================================

HOST = os.environ.get("HOST")
DATABASE = os.environ.get("DATABASE")
DBUSER = os.environ.get("USER") or os.environ.get("DBUSER")
PASSWORD = os.environ.get("PASSWORD")
PORT = os.environ.get("PORT", "5432")
SSL_MODE = os.environ.get("SSL_MODE", "prefer")

# Debug: Print loaded environment variables (mask password for security)
masked_password = "***" if PASSWORD else "NOT SET"
print(f"🔍 Database Config Loaded:")
print(f"   HOST: {HOST or 'NOT SET'}")
print(f"   PORT: {PORT}")
print(f"   DATABASE: {DATABASE or 'NOT SET'}")
print(f"   USER: {DBUSER or 'NOT SET'}")
print(f"   PASSWORD: {masked_password}")
print(f"   SSL_MODE: {SSL_MODE}")

if not HOST or not DATABASE or not DBUSER or not PASSWORD:
    missing = []
    if not HOST: missing.append("HOST")
    if not DATABASE: missing.append("DATABASE")
    if not DBUSER: missing.append("USER/DBUSER")
    if not PASSWORD: missing.append("PASSWORD")
    print(f"⚠️  Warning: Missing environment variables: {', '.join(missing)}")
else:
    print("✅ All required database environment variables are set")

# ============================================================================
# ENVIRONMENT VARIABLE VALIDATION
# ============================================================================

if not all([HOST, DATABASE, DBUSER, PASSWORD]):
    missing_vars = [var for var, val in [("HOST", HOST), ("DATABASE", DATABASE), ("USER", DBUSER), ("PASSWORD", PASSWORD)] if not val]
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}. Please set them in your .env file or environment.")

# ============================================================================
# PASSWORD ENCODING
# ============================================================================

encoded_password = urllib.parse.quote_plus(PASSWORD)

# ============================================================================
# CONNECTION CONFIGURATION
# ============================================================================

# PORT and SSL_MODE are already defined above with the other env vars

# ============================================================================
# DATABASE CONNECTION URL CONSTRUCTION
# ============================================================================

if SSL_MODE != "disable":
    SYNC_DATABASE_URL = f"postgresql+psycopg2://{DBUSER}:{encoded_password}@{HOST}:{PORT}/{DATABASE}?sslmode={SSL_MODE}"
else:
    SYNC_DATABASE_URL = f"postgresql+psycopg2://{DBUSER}:{encoded_password}@{HOST}:{PORT}/{DATABASE}"

# ============================================================================
# SQLALCHEMY ENGINE CREATION
# ============================================================================

connect_args = {}
if SSL_MODE != "disable":
    connect_args["sslmode"] = SSL_MODE

# Create connection URL for logging (mask password)
log_url = SYNC_DATABASE_URL.replace(encoded_password, "***") if encoded_password else SYNC_DATABASE_URL
print(f"🔗 Database Connection URL: {log_url}")

try:
    sync_engine = create_engine(
        SYNC_DATABASE_URL,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        echo=False,
        connect_args=connect_args
    )
    print("✅ SQLAlchemy engine created successfully")
except Exception as e:
    print(f"❌ Error creating SQLAlchemy engine: {e}")
    raise

