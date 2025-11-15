# src/retrieval/reranker.py
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from config import settings

def create_reranker(base_retriever):
    # chọn model cross-encoder mà bạn chắc có trên HuggingFace
    # ví dụ một cross-encoder đa ngôn ngữ như "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # hoặc nếu bạn tìm được cross-encoder tiếng Việt
    cross = HuggingFaceCrossEncoder(
        model_name=settings.RERANKER_MODEL_NAME,
        model_kwargs={"device": "cuda"}  # hoặc "cuda" nếu có GPU
    )
    compressor = CrossEncoderReranker(
        model=cross,
        top_n=settings.TOP_N_RERANK
    )
    print("✅ Using reranker:", settings.RERANKER_MODEL_NAME)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    return compression_retriever
