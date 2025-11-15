# src/cache/gpt_cache_manager.py

import langchain
from langchain_community.cache import GPTCache
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.processor import pre
from gptcache.embedding.sbert import SBERT
from config import settings
import torch

def safe_last_content(data, **kwargs):
    """A safer pre-embedding function that avoids NoneType errors."""
    if not isinstance(data, dict):
        return str(data)
    msgs = data.get("messages")
    if not msgs or not isinstance(msgs, list):
        return str(data)
    return msgs[-1].get("content", str(data))

def init_cache(cache_model_name: str = "all-MiniLM-L6-v2"):
    """
    Initialize GPTCache using SBERT embedding.  
    cache_model_name: tên model SBERT nhẹ — mặc định là all-MiniLM-L6-v2.
    """
    # Khởi SBERT embedding model (có sử dụng PyTorch / transformers)
    sbert = SBERT(model=cache_model_name)

    # Tạo cache base và vector base
    cache_base = CacheBase("sqlite", sql_url=f"sqlite:///{settings.CACHE_DB_PATH}")
    vector_base = VectorBase("faiss", dimension=sbert.dimension)
    data_manager = get_data_manager(cache_base=cache_base, vector_base=vector_base)

    def init_gptcache(cache_obj, llm_str):
        cache_obj.init(
            pre_embedding_func=safe_last_content,
            data_manager=data_manager,
            embedding_func=sbert.to_embeddings,
        )

    langchain.llm_cache = GPTCache(init_func=init_gptcache)
    print(f"✅ GPTCache initialized using SBERT embedding ({cache_model_name}).")

    # Nếu dùng GPU, optional: đẩy model SBERT sang GPU nếu có thể
    try:
        if torch.cuda.is_available():
            # Access underlying SentenceTransformer model if possible
            model = getattr(sbert, "model", None)
            if model is not None:
                model.to("cuda")
                print("ℹ️ SBERT model moved to GPU.")
    except Exception as e:
        print("⚠️ Không thể di chuyển SBERT lên GPU:", e)
