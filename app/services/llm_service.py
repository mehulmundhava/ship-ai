"""
LLM Service Module

This module provides a unified interface for interacting with OpenAI and Groq LLMs.
Supports switching between providers via LLM_PROVIDER environment variable.
"""

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from app.config.settings import settings

# Optional Groq import - only needed if LLM_PROVIDER=GROQ
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    ChatGroq = None


class LLMService:
    """
    Service class for Large Language Model interactions.
    Supports both OpenAI and Groq providers.
    """
    
    def __init__(self):
        """Initialize the LLM service."""
        self.provider = settings.llm_provider
        print(f"🔧 LLM Provider: {self.provider}")
    
    def get_provider(self) -> str:
        """
        Get the active LLM provider from settings.
        
        Returns:
            str: 'OPENAI' or 'GROQ' (default: 'OPENAI')
        """
        return self.provider
    
    def get_llm_model(
        self, 
        temperature: float = 0, 
        model: str = None
    ) -> BaseChatModel:
        """
        Get LLM instance based on LLM_PROVIDER setting.
        
        Args:
            temperature (float): Controls randomness in responses. Default: 0
            model (str): Model name. If None, uses provider-specific default.
        
        Returns:
            BaseChatModel: Configured LLM instance (OpenAI or Groq)
        
        Raises:
            ValueError: If LLM_PROVIDER is not 'OPENAI' or 'GROQ'
        """
        provider = self.get_provider()
        
        if provider == 'OPENAI':
            return self.openai_llm_model(
                temperature=temperature,
                model=model or "gpt-4o"
            )
        elif provider == 'GROQ':
            if not GROQ_AVAILABLE:
                raise ImportError(
                    "Groq is not installed. "
                    "Install it with: pip install langchain-groq"
                )
            return self.groq_llm_model(
                temperature=temperature,
                model=model or "llama-3.3-70b-versatile"
            )
        else:
            raise ValueError(
                f"Invalid LLM_PROVIDER: {provider}. "
                "Must be 'OPENAI' or 'GROQ'"
            )
    
    def openai_llm_model(self, temperature=0, model="gpt-4o"):
        """
        Create an OpenAI LLM instance.
        
        Args:
            temperature (float): Controls randomness in responses. Default: 0
            model (str): OpenAI model name. Default: "gpt-4o"
        
        Returns:
            ChatOpenAI: Configured OpenAI LLM instance
        """
        api_key = settings.openai_api_key
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set API_KEY or OPENAI_API_KEY environment variable."
            )
        
        openai_cred_llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature
        )
        return openai_cred_llm
    
    def groq_llm_model(self, temperature=0, model="llama-3.3-70b-versatile"):
        """
        Create a Groq LLM instance.
        
        Args:
            temperature (float): Controls randomness in responses. Default: 0
            model (str): Groq model name. Default: "llama-3.3-70b-versatile"
                        Other options: "llama-3.1-8b-versatile", "mixtral-8x7b-32768", 
                        "llama-3.1-70b-versatile" (deprecated)
        
        Returns:
            ChatGroq: Configured Groq LLM instance
        """
        api_key = settings.groq_api_key
        if not api_key:
            raise ValueError(
                "Groq API key not found. "
                "Set GROQ_API_KEY environment variable."
            )
        
        groq_llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model,
            temperature=temperature
        )
        return groq_llm
    
    def get_fallback_llm_model(
        self, 
        temperature: float = 0, 
        model: str = None
    ) -> BaseChatModel:
        """
        Get fallback LLM instance (always OpenAI/ChatGPT).
        Used when primary provider (Groq) fails.
        
        Args:
            temperature (float): Controls randomness in responses. Default: 0
            model (str): Model name. If None, uses OpenAI default.
        
        Returns:
            BaseChatModel: Configured OpenAI LLM instance
        """
        return self.openai_llm_model(
            temperature=temperature,
            model=model or "gpt-4o"
        )

