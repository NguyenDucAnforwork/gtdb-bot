# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM and Embedding Models ---
GEMINI_MODEL_NAME = "gemini-2.0-flash"
OPENAI_MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "text-embedding-3-small" # Example, can be changed

# --- Vector Store ---
QDRANT_URL = os.getenv("QDRANT_URL", "localhost")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "gtdb-1"
QDRANT_RETURN_DOCS = 3

# --- APIs ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Reranker ---
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2" # Example for Cohere
TOP_N_RERANK = 3

# --- Web Search ---
TAVILY_MAX_RESULTS = 2

# --- Caching ---
CACHE_DB_PATH = "cache.db"

# --- Cache Model ---
CACHE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Example embedding model for cache
