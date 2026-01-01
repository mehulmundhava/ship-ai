"""
Database Connection Module for PostgreSQL

This module handles all database connection logic for the application.
It creates two separate SQLAlchemy engines:
1. sync_engine - Read-only connection (for health, chat endpoints)
2. sync_engine_update - Update connection (for embedding generation routes)
"""

import urllib.parse
from sqlalchemy import create_engine
from app.config.settings import settings


# ============================================================================
# READ-ONLY DATABASE CONNECTION (USER, PASSWORD)
# ============================================================================

def _create_engine(user: str, password: str, connection_name: str):
    """Helper function to create a database engine."""
    encoded_password = urllib.parse.quote_plus(password)
    
    # Construct connection URL
    if settings.db_ssl_mode != "disable":
        database_url = (
            f"postgresql+psycopg2://{user}:{encoded_password}"
            f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
            f"?sslmode={settings.db_ssl_mode}"
        )
    else:
        database_url = (
            f"postgresql+psycopg2://{user}:{encoded_password}"
            f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
        )
    
    # Connection arguments
    connect_args = {}
    if settings.db_ssl_mode != "disable":
        connect_args["sslmode"] = settings.db_ssl_mode
    
    # Create connection URL for logging (mask password)
    log_url = database_url.replace(encoded_password, "***") if encoded_password else database_url
    print(f"🔗 {connection_name} Connection URL: {log_url}")
    
    try:
        engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            echo=False,
            connect_args=connect_args
        )
        print(f"✅ {connection_name} SQLAlchemy engine created successfully")
        return engine
    except Exception as e:
        print(f"❌ Error creating {connection_name} SQLAlchemy engine: {e}")
        raise


# ============================================================================
# READ-ONLY ENGINE (for health, chat endpoints)
# ============================================================================

sync_engine = _create_engine(
    user=settings.db_user,
    password=settings.db_password,
    connection_name="Read-Only Database"
)

# ============================================================================
# UPDATE ENGINE (for embedding generation routes)
# ============================================================================

# Use UPDATE_USER/UPDATE_PASSWORD if provided, otherwise fall back to readonly user
update_user = settings.UPDATE_USER if settings.UPDATE_USER else settings.db_user
update_password = settings.UPDATE_PASSWORD if settings.UPDATE_PASSWORD else settings.db_password

if settings.UPDATE_USER and settings.UPDATE_PASSWORD:
    sync_engine_update = _create_engine(
        user=update_user,
        password=update_password,
        connection_name="Update Database"
    )
    print("✅ Using separate UPDATE_USER connection for embedding routes")
else:
    # Fallback to readonly connection if UPDATE_USER not configured
    sync_engine_update = sync_engine
    print("⚠️  UPDATE_USER not configured - using readonly connection for embedding routes (not recommended)")

