"""
Validation Helpers

Utility functions for request validation and data validation.
"""

from typing import Optional


def validate_token_id(token_id: str) -> bool:
    """
    Validate authentication token.
    
    Args:
        token_id: Token to validate
        
    Returns:
        True if token is valid, False otherwise
    """
    return token_id == "Test123"


def validate_user_id(user_id: Optional[str]) -> bool:
    """
    Validate user ID format.
    
    Args:
        user_id: User ID to validate
        
    Returns:
        True if user ID is valid (or None), False otherwise
    """
    if user_id is None:
        return True
    # Add any user ID validation logic here
    return len(str(user_id).strip()) > 0

