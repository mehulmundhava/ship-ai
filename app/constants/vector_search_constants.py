"""
Vector Search Constants

Constants for vector search limits loaded from environment variables.
"""

import os
from app.config.settings import load_env_file

# Load environment variables
load_env_file()

# Vector search limits with defaults
VECTOR_EXAMPLES_LIMIT = int(os.getenv("VECTOR_EXAMPLES_LIMIT", "5"))
VECTOR_EXTRA_PROMPTS_LIMIT = int(os.getenv("VECTOR_EXTRA_PROMPTS_LIMIT", "5"))
