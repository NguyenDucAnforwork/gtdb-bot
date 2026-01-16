# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

# Base directory của project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

# ===========================
# HIPPORAG CONFIGURATION
# ===========================
# Customized for Vietnamese Traffic Law (từ notebook section 3)

# LLM config - Dùng model rẻ cho extraction
HIPPORAG_LLM_NAME = "gpt-4o-mini"
HIPPORAG_MAX_NEW_TOKENS = 2048
HIPPORAG_TEMPERATURE = 0

# Embedding config - Vietnamese model
HIPPORAG_EMBEDDING_MODEL = "Transformers/AITeamVN/Vietnamese_Embedding"
HIPPORAG_EMBEDDING_BATCH_SIZE = 32
HIPPORAG_EMBEDDING_RETURN_NORMALIZED = True
HIPPORAG_EMBEDDING_MAX_SEQ_LEN = 512

# Preprocessing config - Phù hợp với điều khoản luật
HIPPORAG_CHUNK_MAX_TOKEN_SIZE = 800  # Phù hợp 400-800 tokens cho điều khoản luật
HIPPORAG_CHUNK_OVERLAP_TOKEN_SIZE = 10  # Giảm overlap để tránh duplicate
HIPPORAG_CHUNK_FUNC = 'by_token'

# Graph construction config
HIPPORAG_SYNONYMY_EDGE_TOPK = 1024  # Giảm để tăng tốc
HIPPORAG_SYNONYMY_EDGE_SIM_THRESHOLD = 0.85  # Tăng threshold để chỉ giữ synonyms chắc chắn
HIPPORAG_IS_DIRECTED_GRAPH = False

# Retrieval config - Tối ưu cho luật
HIPPORAG_LINKING_TOP_K = 10  # Tăng từ 5 để bắt nhiều entity liên quan
HIPPORAG_RETRIEVAL_TOP_K = 100  # Giảm từ 200 để nhanh hơn
HIPPORAG_PASSAGE_NODE_WEIGHT = 0.1  # Tăng trọng số passage (luật cần văn bản chính xác)
HIPPORAG_DAMPING = 0.5

# QA config
HIPPORAG_MAX_QA_STEPS = 1
HIPPORAG_QA_TOP_K = 5  # Top 5 điều khoản để đọc

# Storage config
# Path đến thư mục chứa data đã được index (PHẢI LÀ ABSOLUTE PATH)
_hipporag_rel_path = 'kaggle/working/HippoRAG/content/HippoRAG/outputs/vietnamese_traffic_law'
HIPPORAG_SAVE_DIR = os.path.join(BASE_DIR, _hipporag_rel_path)
HIPPORAG_SAVE_OPENIE = True  # Lưu kết quả OpenIE để tái sử dụng
HIPPORAG_FORCE_INDEX_FROM_SCRATCH = False  # Set True nếu muốn build lại từ đầu
HIPPORAG_FORCE_OPENIE_FROM_SCRATCH = False
