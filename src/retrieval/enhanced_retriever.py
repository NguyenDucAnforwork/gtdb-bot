# src/retrieval/enhanced_retriever.py
from langchain.retrievers import MergerRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
import re

from langchain_community.retrievers.tavily_search_api import TavilySearchAPIRetriever

from config import settings
from src.retrieval.web_search import get_web_search_tool
from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore
from typing import List, Dict, Any, Set
from pydantic import Field
from langchain_core.retrievers import BaseRetriever
import qdrant_client
from time import time
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range, IntegerIndexParams, IntegerIndexType, PayloadSchemaType

# Mapping văn bản có lỗi font encoding (giống HippoRAG)
DOCUMENT_MAPPING = {
    "Ngh nh 168-2024-N-CP": "Nghị định 168/2024/NĐ-CP",
    "Ngh nh 03-2021-N-CP": "Nghị định 03/2021/NĐ-CP",
    "Ngh nh 100-2019-N-CP": "Nghị định 100/2019/NĐ-CP",
    "Ngh nh 123-2021-N-CP": "Nghị định 123/2021/NĐ-CP",
    "Lu t 35-2024-QH15": "Luật 35/2024/QH15",
    "Lu t 36-2024-QH15": "Luật 36/2024/QH15",
}

def format_qdrant_citation(metadata: Dict[str, Any]) -> str:
    """
    Format citation từ Qdrant metadata
    
    Args:
        metadata: Document metadata với law_id, article_id, clause_id
    
    Returns:
        Formatted citation string
    """
    citation_parts = []
    
    law_id = metadata.get("law_id")
    if law_id:
        # Fix encoding nếu cần
        law_id = DOCUMENT_MAPPING.get(law_id, law_id)
        citation_parts.append(law_id)
    
    article_id = metadata.get("article_id")
    if article_id:
        citation_parts.append(f"Điều {article_id}")
    
    clause_id = metadata.get("clause_id")
    if clause_id:
        citation_parts.append(f"Khoản {clause_id}")
    
    point_id = metadata.get("point_id")
    if point_id:
        citation_parts.append(f"Điểm {point_id}")
    
    if citation_parts:
        return ", ".join(citation_parts)
    else:
        return "Không xác định nguồn"

def get_qdrant_retriever(embeddings):
    client = qdrant_client.QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        prefer_grpc=True
    )

    # Khởi tạo vector store Qdrant (dense / hybrid / sparse tùy config)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        embedding=embeddings,
        content_payload_key="text",
        # metadata_payload_key="metadata",
        # nếu bạn muốn hỗn hợp, bạn có thể thêm sparse_embedding hoặc retrieval_mode
        # ví dụ: retrieval_mode="hybrid" hoặc RetrievalMode.HYBRID
    )

    # Dùng as_retriever() với search_kwargs để control parameters
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": settings.QDRANT_RETURN_DOCS,  # số lượng documents trả về
            # Không set score_threshold để lấy tất cả kết quả
        }
    )
    return retriever

class EnhancedQdrantRetriever(BaseRetriever):
    """Custom Qdrant retriever with metadata handling và recursive search"""
    
    # Declare Pydantic fields
    embeddings: Any = Field(description="Embedding model")
    collection_name: str = Field(description="Qdrant collection name")
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold")
    client: Any = Field(default=None, description="Qdrant client")
    
    def __init__(self, embeddings, similarity_threshold=0.7, **kwargs):
        # Initialize with proper Pydantic field assignment
        super().__init__(
            embeddings=embeddings,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            similarity_threshold=similarity_threshold,
            client=qdrant_client.QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                prefer_grpc=True
            ),
            **kwargs
        )
        print("✅ EnhancedQdrantRetriever initialized")
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Required method from BaseRetriever"""
        return self._search_documents(query)
    
    def _search_documents(self, query: str) -> List[Document]:
        """Main retrieval method with proper metadata extraction"""
        try:
            print(f"🔍 EnhancedQdrantRetriever searching for: {query}")
            
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search trong Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=7,
                with_payload=True,
                score_threshold=self.similarity_threshold
            )
            
            documents = []
            
            for point in search_results:
                # 🔧 Lấy content từ field "text"
                content = point.payload.get("text", "")
                
                if not content or not content.strip():
                    print(f"⚠️ Empty content for point {point.id}")
                    continue
                
                # 🔧 Build metadata từ tất cả fields khác (trừ "text")
                metadata = {
                    "_id": point.id,
                    "_score": point.score,
                    "_collection": self.collection_name,
                }
                
                # Thêm tất cả fields khác làm metadata
                for key, value in point.payload.items():
                    if key != "text":  # Không include field content
                        metadata[key] = value
                
                # Tạo Document
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)
            
            print(f"✅ Found {len(documents)} documents with content")
            
            # Debug first document
            if documents:
                first_doc = documents[0]
                print(f"📄 First doc preview:")
                print(f"   Content length: {len(first_doc.page_content)}")
                print(f"   Metadata keys: {list(first_doc.metadata.keys())}")
                print(f"   Content preview: {first_doc.page_content[:100]}...")
            
            return documents
            
        except Exception as e:
            print(f"❌ EnhancedQdrantRetriever failed: {e}")
            import traceback
            traceback.print_exc()
            return []

class RecursiveQdrantRetriever(BaseRetriever):
    """Recursive retriever với metadata filtering"""
    
    # Declare Pydantic fields
    embeddings: Any = Field(description="Embedding model")  
    collection_name: str = Field(description="Qdrant collection name")
    similarity_threshold: float = Field(default=0.28, description="Similarity threshold")
    max_depth: int = Field(default=2, description="Maximum recursion depth")
    client: Any = Field(default=None, description="Qdrant client")
    
    def __init__(self, embeddings, max_depth=2, similarity_threshold=0.28, **kwargs):
        # Initialize with proper Pydantic field assignment
        super().__init__(
            embeddings=embeddings,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            max_depth=max_depth,
            similarity_threshold=similarity_threshold,
            client=qdrant_client.QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                prefer_grpc=True
            ),
            **kwargs
        )

        # --- tạo index cho các trường metadata cần filter ---
        try:
            # law_id là string → dùng keyword
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="law_id",
                field_schema=PayloadSchemaType.KEYWORD
            )
            # article_id là int → dùng IntegerIndexParams với lookup & range = True
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="article_id",
                field_schema=IntegerIndexParams(
                    type=IntegerIndexType.INTEGER,
                    lookup=True,
                    range=True
                )
            )
            print(f"✅ Payload indexes ensured for fields law_id + article_id")
        except Exception as e:
            print(f"⚠️ Could not create payload indexes (maybe already exist): {e}")
            
        print(f"✅ RecursiveQdrantRetriever initialized (max_depth={max_depth})")
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Required method from BaseRetriever"""
        return self._recursive_search(query)
    
    def _recursive_search(self, query: str) -> List[Document]:
        """Main method với recursive search"""
        # 1. Base search
        base_docs = self._base_search(query)
        if not base_docs:
            return []
            
        # 2. Recursive search: bắt đầu với base_docs và depth=0
        seen_ids = {doc.metadata.get("_id") for doc in base_docs if doc.metadata.get("_id")}
        all_docs = base_docs.copy()
        
        # Bắt đầu đệ quy
        self._search_related_recursively(base_docs, current_depth=0, seen_ids=seen_ids, all_accumulated_docs=all_docs)

        print(f"🎯 Total unique documents after recursion: {len(all_docs)}")
        return all_docs
    
    def _base_search(self, query: str) -> List[Document]:
        """Base search method"""
        try:
            print(f"🔍 RecursiveQdrantRetriever base search for: {query}")
            print(f"   Using similarity_threshold: {self.similarity_threshold}")
            
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search trong Qdrant - KHÔNG dùng score_threshold để lấy hết kết quả
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=15,
                with_payload=True,
                # Bỏ score_threshold để lấy tất cả kết quả như get_qdrant_retriever
                # score_threshold=self.similarity_threshold
            )
            print(f"✅ Base search in _base_search function found {len(search_results)} documents")
            
            documents = []
            
            for point in search_results:
                # 🔧 Lấy content từ field "text"
                content = point.payload.get("text", "")
                
                if not content or not content.strip():
                    continue
                
                # Filter by similarity threshold manually (sau khi lấy kết quả)
                if point.score < self.similarity_threshold:
                    # print(f"⚠️ Point {point.id} filtered out (score {point.score:.4f} < threshold {self.similarity_threshold})")
                    continue
                
                # 🔧 Build metadata từ tất cả fields khác (trừ "text")
                metadata = {
                    "_id": point.id,
                    "_score": point.score,
                    "_collection": self.collection_name,
                    "_depth": 0  # Base search depth
                }
                
                # Thêm tất cả fields khác làm metadata
                for key, value in point.payload.items():
                    if key != "text":  # Không include field content
                        metadata[key] = value
                
                # Tạo Document
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)
            
            print(f"✅ Returned {len(documents)} documents after filtering")
            return documents
            
        except Exception as e:
            print(f"❌ Base search failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _search_related_by_metadata(self, base_docs: List[Document], depth: int) -> List[Document]:
        next_depth = depth + 1
        if next_depth > self.max_depth:
            return []

        related_docs = []
        for doc in base_docs:
            law_id = doc.metadata.get("law_id")
            article_id = doc.metadata.get("article_id")
            if law_id is None or article_id is None:
                continue

            # Build filter: law_id = string, article_id = int
            filt = Filter(
                must=[
                    FieldCondition(key="law_id", match=MatchValue(value=str(law_id))),
                    FieldCondition(key="article_id", match=MatchValue(value=int(article_id)))
                ]
            )
            try:
                q_emb = self.embeddings.embed_query(doc.page_content[:200])
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=q_emb,
                    limit=5,
                    with_payload=True,
                    query_filter=filt
                )
                print(f"🔍 Related search for law_id={law_id}, article_id={article_id} at depth {next_depth}, found {len(results)} results")
                for pt in results:
                    content = pt.payload.get("text", "").strip()
                    if not content:
                        continue
                    metadata = {
                        "_id": pt.id,
                        "_score": pt.score,
                        "_depth": next_depth
                    }
                    for k, v in pt.payload.items():
                        if k != "text":
                            metadata[k] = v
                    related_docs.append(Document(page_content=content, metadata=metadata))
            except Exception as e:
                print(f"⚠️ Related search failed for law_id={law_id}, article_id={article_id}: {e}")

        print(f"🔄 Depth {next_depth}: found {len(related_docs)} related docs")
        return related_docs
    
    def _search_related_recursively(self, docs_to_process: List[Document], current_depth: int, seen_ids: Set, all_accumulated_docs: List[Document]):
        """
        Hàm đệ quy tìm kiếm các tài liệu liên quan dựa trên trích xuất text.
        """
        next_depth = current_depth + 1
        if next_depth > self.max_depth:
            return

        new_docs = []
        
        for doc in docs_to_process:
            content = doc.page_content.lower()
            law_id = doc.metadata.get("law_id")
            article_id = doc.metadata.get("article_id")

            # Chỉ xử lý nếu có đủ law_id và article_id để tham chiếu
            if not law_id or not article_id:
                continue

            # Regex để tìm pattern "khoản X Điều này"
            # \b là ranh giới từ, \d+ là một hoặc nhiều chữ số
            # Pattern này có thể cần tinh chỉnh tùy vào dữ liệu thực tế của bạn
            matches = re.findall(r"khoản\s+(\d+)\s+điều\s+này", content)
            print(f"Tìm thấy khoản tham chiếu trong doc {doc.metadata.get('_id')}: {matches}")
            
            if not matches:
                continue

            # Deduplicate các clause_id tìm thấy trong cùng 1 doc
            referenced_clause_ids = set(matches)
            
            for clause_id_str in referenced_clause_ids:
                try:
                     # Giả sử clause_id trong DB là integer. Nếu là string thì bỏ int()
                    target_clause_id = int(clause_id_str)
                    
                    # Tìm kiếm chính xác trong Qdrant bằng Filter
                    # Lưu ý: Cần đảm bảo 'clause_id' đã được index trong Qdrant
                    filt = Filter(
                        must=[
                            FieldCondition(key="law_id", match=MatchValue(value=str(law_id))),
                            FieldCondition(key="article_id", match=MatchValue(value=int(article_id))),
                            FieldCondition(key="clause_id", match=MatchValue(value=target_clause_id))
                        ]
                    )
                    
                    # Dùng scroll để lấy tất cả chunk khớp (thường một khoản có thể bị chia thành nhiều chunk)
                    # Limit có thể điều chỉnh tùy độ dài trung bình của khoản
                    points, _ = self.client.scroll(
                        collection_name=self.collection_name,
                        scroll_filter=filt,
                        limit=5, 
                        with_payload=True,
                        with_vectors=False # Không cần vector
                    )
                    
                    for point in points:
                        if point.id in seen_ids:
                            continue
                            
                        seen_ids.add(point.id)
                        
                        # Tạo document mới
                        content = point.payload.get("text", "")
                        if content:
                            metadata = {"_id": point.id, "_depth": next_depth, "_source_type": "referenced"}
                            for k, v in point.payload.items():
                                if k != "text":
                                    metadata[k] = v
                                    
                            new_doc = Document(page_content=content, metadata=metadata)
                            new_docs.append(new_doc)
                            all_accumulated_docs.append(new_doc)
                            print(f"  -> Found referenced: Law {law_id}, Art {article_id}, Clause {target_clause_id} (from doc {doc.metadata.get('_id')})")

                except ValueError:
                    # Trường hợp clause_id không phải là số hợp lệ
                    continue
                except Exception as e:
                    print(f"⚠️ Error searching for referenced clause {clause_id_str}: {e}")

        if new_docs:
            print(f"🔄 Depth {next_depth}: found {len(new_docs)} new related docs. Continuing recursion...")
            # Gọi đệ quy với các tài liệu mới tìm được
            self._search_related_recursively(new_docs, next_depth, seen_ids, all_accumulated_docs)
        else:
            print(f"⏹️ Depth {next_depth}: No new related docs found. Stopping recursion branch.")

def create_enhanced_retriever(embeddings):
    """
    Creates a combined retriever that merges results from a vector store and web search.
    """
    # 1. Vector Store Retriever (simple retriever without score filtering)
    qdrant_retriever = get_qdrant_retriever(embeddings)
    
    # 2. Recursive Qdrant Retriever with lower threshold (0.4 instead of 0.7)
    recursive_qdrant_retriever = RecursiveQdrantRetriever(
        embeddings=embeddings,
        max_depth=2,
        similarity_threshold=0.4  # Lower threshold to get more results
    )

    # 3. Merge the retrievers
    tavily_retriever = TavilySearchAPIRetriever(api_key=settings.TAVILY_API_KEY, k=settings.TAVILY_MAX_RESULTS)
    print("✅ Tavily Search Retriever Created.", type(tavily_retriever))
    lotr = MergerRetriever(retrievers=[recursive_qdrant_retriever, tavily_retriever])
    print("✅ Enhanced Retriever Created.")
 
    return lotr

def create_vector_only_retriever(embeddings):
    """
    Tạo retriever chỉ dùng Qdrant vector store (không web search)
    
    Returns:
        RecursiveQdrantRetriever với citations từ metadata
    """
    print("🔍 Creating Vector-Only Retriever (Qdrant)...")
    
    retriever = RecursiveQdrantRetriever(
        embeddings=embeddings,
        max_depth=2,
        similarity_threshold=0.28
    )
    
    print("✅ Vector-Only Retriever Created")
    return retriever

def create_hipporag_only_retriever():
    """
    Tạo retriever chỉ dùng HippoRAG knowledge graph
    
    Returns:
        HippoRAGRetriever
    """
    print("🧠 Creating HippoRAG-Only Retriever...")
    
    from src.retrieval.hipporag_retriever import get_hipporag_retriever
    
    retriever = get_hipporag_retriever(max_docs_per_query=3)
    
    print("✅ HippoRAG-Only Retriever Created")
    return retriever

def create_combined_retriever(embeddings):
    """
    Tạo retriever kết hợp HippoRAG + Qdrant
    
    Combine results từ:
    - HippoRAG: Knowledge graph với citations từ doc titles
    - Qdrant: Vector store với citations từ metadata
    
    Returns:
        MergerRetriever combining both
    """
    print("🔄 Creating Combined Retriever (HippoRAG + Qdrant)...")
    
    # Get individual retrievers
    vector_retriever = create_vector_only_retriever(embeddings)
    hipporag_retriever = create_hipporag_only_retriever()
    
    # Merge them
    combined = MergerRetriever(retrievers=[hipporag_retriever, vector_retriever])
    
    print("✅ Combined Retriever Created (HippoRAG + Qdrant)")
    return combined
