"""
LLM Model Wrapper Module

This module provides a unified interface for interacting with OpenAI LLM.
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


class LLMModel:
    """
    Wrapper class for Large Language Model interactions.
    """
    
    def __init__(self):
        """Initialize the LLM model wrapper."""
        pass
    
    def openai_llm_model(self, temperature=0, model="gpt-4o"):
        """
        Create an OpenAI LLM instance.
        
        Args:
            temperature (float): Controls randomness in responses. Default: 0
            model (str): OpenAI model name. Default: "gpt-4o"
        
        Returns:
            ChatOpenAI: Configured OpenAI LLM instance
        """
        openai_cred_llm = ChatOpenAI(
            api_key=os.environ.get('API_KEY'),
            model=model,
            temperature=temperature
        )
        return openai_cred_llm

