# src/generation/openai_generator.py
from langchain_openai import ChatOpenAI
from config import settings
def get_llm():
    """Initializes and returns the OpenAI LLM."""
    return ChatOpenAI(
        model_name=settings.OPENAI_MODEL_NAME,
        openai_api_key=settings.OPENAI_API_KEY,
        temperature=0
    )