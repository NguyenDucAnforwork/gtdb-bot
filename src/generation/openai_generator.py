# src/generation/openai_generator.py
from openai import OpenAI
from config import settings
import os

def get_llm():
    """Initializes and returns the OpenAI client."""
    # Set API key in environment if available
    if settings.OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    
    return OpenAI(api_key=settings.OPENAI_API_KEY)

def get_llm_params():
    """Returns default LLM parameters for OpenAI calls."""
    return {
        "model": settings.OPENAI_MODEL_NAME,
        "temperature": 0,
        "max_completion_tokens": 3000
    }
