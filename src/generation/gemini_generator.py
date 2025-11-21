# src/generation/gemini_generator.py
from langchain_google_genai import ChatGoogleGenerativeAI
from config import settings

def get_llm():
    """Initializes and returns the Gemini LLM."""
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL_NAME,
        google_api_key=settings.GOOGLE_API_KEY,
        convert_system_message_to_human=True # For compatibility
    )
