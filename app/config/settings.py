"""
Application Settings and Configuration

Centralizes all environment variable loading and configuration.
Uses Pydantic Settings for validation.
"""

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv


def load_env_file():
    """Load .env file from multiple possible locations."""
    base_path = Path(__file__).parent.parent.parent
    env_paths = [
        base_path / ".env",
        base_path.parent / ".env",
        base_path.parent / "ship-ai" / ".env",
    ]
    
    env_path = None
    for path in env_paths:
        if path.exists():
            env_path = path
            print(f"📄 Config: Loaded .env from {env_path}")
            break
    
    if env_path:
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv(override=True)
        print("⚠️  Config: Using default .env loading")


# Load environment variables first
load_env_file()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Use environment variable names directly as field names
    # Database Configuration - Read-only user (for health, chat endpoints)
    HOST: str = ""
    PORT: str = "5432"
    DATABASE: str = ""
    USER: str = ""
    PASSWORD: str = ""
    SSL_MODE: str = "prefer"
    
    # Database Configuration - Update user (for embedding generation routes)
    # Optional: If not provided, falls back to readonly user (not recommended)
    UPDATE_USER: str = ""
    UPDATE_PASSWORD: str = ""
    
    # LLM Configuration
    LLM_PROVIDER: str = "OPENAI"
    API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    # When True, skip Groq and use fallback (e.g. OpenAI) directly. Use when Groq key is expired/invalid to avoid latency.
    GROQ_DISABLED: bool = False
    
    # Embedding Configuration
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    HUGGING_FACE_MODEL: Optional[str] = None
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    LOG_DIR: str = "logs"
    LOG_ROTATION_INTERVAL_HOURS: int = 24
    LOG_RETENTION_DAYS: int = 30
    LOG_TO_CONSOLE: bool = True
    LOG_TO_FILE: bool = True
    
    # API base URL for CSV download links (e.g. http://localhost:3009 or http://54.183.26.153/postgre)
    API_BASE_URL: str = "http://localhost:3009"

    # Vector Cache Configuration
    VECTOR_CACHE_ENABLED: bool = True
    VECTOR_CACHE_SIMILARITY_THRESHOLD: float = 0.80
    VECTOR_CACHE_AUTO_SAVE: bool = True
    VECTOR_CACHE_DETERMINISTIC_ONLY: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields from environment
    )
    
    def model_post_init(self, __context):
        """Post-initialization to handle fallbacks and normalization."""
        import os
        
        # Handle USER vs DBUSER fallback
        if not self.USER:
            self.USER = os.environ.get("DBUSER", "")
        
        # Handle API_KEY vs OPENAI_API_KEY fallback
        if not self.API_KEY:
            self.API_KEY = os.environ.get("OPENAI_API_KEY")
        
        # Normalize LLM provider
        if self.LLM_PROVIDER:
            self.LLM_PROVIDER = self.LLM_PROVIDER.upper()
        
        # Use HUGGING_FACE_MODEL if provided, otherwise use EMBEDDING_MODEL_NAME
        if not self.HUGGING_FACE_MODEL:
            self.HUGGING_FACE_MODEL = self.EMBEDDING_MODEL_NAME
    
    # Properties for backward compatibility with db_* naming
    @property
    def db_host(self) -> str:
        return self.HOST
    
    @property
    def db_port(self) -> str:
        return self.PORT
    
    @property
    def db_name(self) -> str:
        return self.DATABASE
    
    @property
    def db_user(self) -> str:
        return self.USER
    
    @property
    def db_password(self) -> str:
        return self.PASSWORD
    
    @property
    def db_ssl_mode(self) -> str:
        return self.SSL_MODE
    
    @property
    def llm_provider(self) -> str:
        return self.LLM_PROVIDER
    
    @property
    def openai_api_key(self) -> Optional[str]:
        return self.API_KEY
    
    @property
    def groq_api_key(self) -> Optional[str]:
        return self.GROQ_API_KEY
    
    def get_api_base_url(self) -> str:
        """Base URL for API (e.g. CSV download links). No trailing slash."""
        return (self.API_BASE_URL or "http://localhost:3009").rstrip("/")

    @property
    def embedding_model_name(self) -> str:
        """Get the embedding model name from HUGGING_FACE_MODEL or fallback to EMBEDDING_MODEL_NAME."""
        return self.HUGGING_FACE_MODEL or self.EMBEDDING_MODEL_NAME
    
    def get_embedding_field_name(self) -> str:
        """
        Get the database field name for embeddings based on the model.
        
        Returns:
            'minilm_embedding' for all-MiniLM-L6-v2 models
            'bge_large_embedding' for BAAI/bge-large-en-v1.5 models
            'minilm_embedding' as default fallback
        """
        model_name = self.embedding_model_name.lower()
        
        if "bge-large" in model_name or "bge_large" in model_name:
            return "bge_large_embedding"
        elif "minilm" in model_name:
            return "minilm_embedding"
        else:
            # Default fallback
            return "minilm_embedding"
    
    def validate_database_config(self):
        """Validate that all required database config is present."""
        missing = []
        if not self.HOST:
            missing.append("HOST")
        if not self.DATABASE:
            missing.append("DATABASE")
        if not self.USER:
            missing.append("USER/DBUSER")
        if not self.PASSWORD:
            missing.append("PASSWORD")
        
        # UPDATE_USER and UPDATE_PASSWORD are optional but recommended for embedding routes
        if not self.UPDATE_USER:
            print("⚠️  Warning: UPDATE_USER not set. Embedding routes will use readonly connection.")
        if not self.UPDATE_PASSWORD:
            print("⚠️  Warning: UPDATE_PASSWORD not set. Embedding routes will use readonly connection.")
        
        if missing:
            raise ValueError(
                f"Missing required database environment variables: {', '.join(missing)}. "
                "Please set them in your .env file or environment."
            )
    
    def validate_llm_config(self):
        """Validate that LLM configuration is correct."""
        if self.LLM_PROVIDER == "OPENAI" and not self.API_KEY:
            raise ValueError(
                "OpenAI API key not found. "
                "Set API_KEY or OPENAI_API_KEY environment variable."
            )
        elif self.LLM_PROVIDER == "GROQ" and not self.GROQ_API_KEY:
            raise ValueError(
                "Groq API key not found. "
                "Set GROQ_API_KEY environment variable."
            )
        elif self.LLM_PROVIDER not in ["OPENAI", "GROQ"]:
            raise ValueError(
                f"Invalid LLM_PROVIDER: {self.LLM_PROVIDER}. "
                "Must be 'OPENAI' or 'GROQ'"
            )


# Global settings instance
settings = Settings()

# Validate on import
settings.validate_database_config()
settings.validate_llm_config()

# Debug output
masked_password = "***" if settings.PASSWORD else "NOT SET"
masked_update_password = "***" if settings.UPDATE_PASSWORD else "NOT SET"
print(f"🔍 Database Config Loaded:")
print(f"   HOST: {settings.HOST or 'NOT SET'}")
print(f"   PORT: {settings.PORT}")
print(f"   DATABASE: {settings.DATABASE or 'NOT SET'}")
print(f"   Read-Only USER: {settings.USER or 'NOT SET'}")
print(f"   Read-Only PASSWORD: {masked_password}")
if settings.UPDATE_USER:
    print(f"   Update USER: {settings.UPDATE_USER}")
    print(f"   Update PASSWORD: {masked_update_password}")
else:
    print(f"   Update USER: NOT SET (will use read-only user)")
print(f"   SSL_MODE: {settings.SSL_MODE}")
print(f"🔧 LLM Provider: {settings.LLM_PROVIDER}")

