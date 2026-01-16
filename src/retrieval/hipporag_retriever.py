# src/retrieval/hipporag_retriever.py
"""
HippoRAG Knowledge Graph Retriever
Sử dụng HippoRAG API để retrieve documents với knowledge graph
Customized cho Vietnamese Traffic Law
"""

import os
import sys
from typing import List, Dict, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import Document
from pydantic import Field
import re

# Add HippoRAG directory to path
hipporag_path = os.path.join(os.path.dirname(__file__), '..', '..', 'HippoRAG', 'src')
if hipporag_path not in sys.path:
    sys.path.insert(0, hipporag_path)

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from config import settings
from src.retrieval.vietnamese_law_prompts import get_vietnamese_law_prompts

# Mapping các văn bản có lỗi font encoding
DOCUMENT_MAPPING = {
    "Ngh nh 168-2024-N-CP": "Nghị định 168/2024/NĐ-CP",
    "Ngh nh 03-2021-N-CP": "Nghị định 03/2021/NĐ-CP", 
    "Ngh nh 100-2019-N-CP": "Nghị định 100/2019/NĐ-CP",
    "Ngh nh 123-2021-N-CP": "Nghị định 123/2021/NĐ-CP",
    "Lu t 35-2024-QH15": "Luật 35/2024/QH15",
    "Lu t 36-2024-QH15": "Luật 36/2024/QH15",
}

def _override_hipporag_prompts():
    """
    Override HippoRAG default prompts với Vietnamese Traffic Law prompts
    PHẢI GỌI TRƯỚC khi khởi tạo HippoRAG (như trong notebook section 5)
    """
    try:
        # Import prompt templates từ HippoRAG
        from hipporag.prompts.templates import ner, triple_extraction, rag_qa_musique
        
        # Get Vietnamese law prompts
        vn_prompts = get_vietnamese_law_prompts()
        
        # Override NER prompt (giống y hệt notebook)
        ner.ner_system = vn_prompts['ner']['system']
        ner.one_shot_ner_paragraph = vn_prompts['ner']['example_input']
        ner.one_shot_ner_output = vn_prompts['ner']['example_output']
        ner.prompt_template = vn_prompts['ner']['prompt_template']
        
        # Override Triple Extraction prompt
        triple_extraction.ner_conditioned_re_system = vn_prompts['triple_extraction']['system']
        triple_extraction.ner_conditioned_re_output = vn_prompts['triple_extraction']['example_output']
        
        # Override QA prompt
        rag_qa_musique.rag_qa_system = vn_prompts['qa']['system']
        
        print("✅ Prompts overridden successfully:")
        print("   - NER: Vietnamese Traffic Law specific")
        print("   - Triple extraction: Optimized for law relationships")
        print("   - QA: Legal citation enforced")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not override prompts: {e}")
        print("   Using default HippoRAG prompts instead")


class HippoRAGRetriever(BaseRetriever):
    """
    HippoRAG-based retriever sử dụng knowledge graph
    Customized cho Vietnamese Traffic Law với:
    - Custom NER prompts cho entities trong luật giao thông
    - Custom Triple Extraction cho relationships trong điều khoản
    - Custom QA prompts với citation enforcement
    
    Flow (giống y hệt notebook):
    1. Override prompts TRƯỚC khi init HippoRAG
    2. Initialize HippoRAG với custom config
    3. Use hipporag.retrieve() để retrieve documents
    4. Format thành LangChain Documents
    """
    
    hipporag: Any = Field(default=None, description="HippoRAG instance")
    max_docs_per_query: int = Field(default=3, description="Max documents per query")
    
    def __init__(self, max_docs_per_query: int = 3, **kwargs):
        """Initialize HippoRAG retriever với Vietnamese Traffic Law customization"""
        
        # Load OpenAI API key từ settings hoặc environment
        openai_api_key = getattr(settings, 'OPENAI_API_KEY', None) or os.getenv('OPENAI_API_KEY')
        
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in settings or environment variables")
        
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # ⚠️ QUAN TRỌNG: Override prompts TRƯỚC KHI tạo config (như notebook)
        print("📝 Overriding prompts with Vietnamese Traffic Law templates...")
        _override_hipporag_prompts()
        
        # Debug: Print path being used
        print(f"📁 HIPPORAG_SAVE_DIR: {settings.HIPPORAG_SAVE_DIR}")
        print(f"   Path exists: {os.path.exists(settings.HIPPORAG_SAVE_DIR)}")
        if os.path.exists(settings.HIPPORAG_SAVE_DIR):
            print(f"   Contents: {os.listdir(settings.HIPPORAG_SAVE_DIR)}")
        
        # Create custom config cho Vietnamese Traffic Law (từ notebook section 3)
        print("⚙️ Creating custom HippoRAG config for Vietnamese Traffic Law...")
        config = BaseConfig(
            # LLM config
            llm_name=settings.HIPPORAG_LLM_NAME,
            max_new_tokens=settings.HIPPORAG_MAX_NEW_TOKENS,
            temperature=settings.HIPPORAG_TEMPERATURE,

            # Embedding config - Vietnamese model
            embedding_model_name=settings.HIPPORAG_EMBEDDING_MODEL,
            embedding_batch_size=settings.HIPPORAG_EMBEDDING_BATCH_SIZE,
            embedding_return_as_normalized=settings.HIPPORAG_EMBEDDING_RETURN_NORMALIZED,
            embedding_max_seq_len=settings.HIPPORAG_EMBEDDING_MAX_SEQ_LEN,

            # Preprocessing config
            preprocess_chunk_max_token_size=settings.HIPPORAG_CHUNK_MAX_TOKEN_SIZE,
            preprocess_chunk_overlap_token_size=settings.HIPPORAG_CHUNK_OVERLAP_TOKEN_SIZE,
            preprocess_chunk_func=settings.HIPPORAG_CHUNK_FUNC,

            # Graph construction config
            synonymy_edge_topk=settings.HIPPORAG_SYNONYMY_EDGE_TOPK,
            synonymy_edge_sim_threshold=settings.HIPPORAG_SYNONYMY_EDGE_SIM_THRESHOLD,
            is_directed_graph=settings.HIPPORAG_IS_DIRECTED_GRAPH,

            # Retrieval config
            linking_top_k=settings.HIPPORAG_LINKING_TOP_K,
            retrieval_top_k=settings.HIPPORAG_RETRIEVAL_TOP_K,
            passage_node_weight=settings.HIPPORAG_PASSAGE_NODE_WEIGHT,
            damping=settings.HIPPORAG_DAMPING,

            # QA config
            max_qa_steps=settings.HIPPORAG_MAX_QA_STEPS,
            qa_top_k=settings.HIPPORAG_QA_TOP_K,

            # Storage config - dùng save_dir từ settings
            save_dir=settings.HIPPORAG_SAVE_DIR,
            save_openie=settings.HIPPORAG_SAVE_OPENIE,
            force_index_from_scratch=settings.HIPPORAG_FORCE_INDEX_FROM_SCRATCH,
        )
        
        print("✅ Custom config created:")
        print(f"   - LLM: {config.llm_name}")
        print(f"   - Embedding: {config.embedding_model_name}")
        print(f"   - Save dir: {config.save_dir}")
        print(f"   - Chunk size: {config.preprocess_chunk_max_token_size} tokens")
        print(f"   - Retrieval top-k: {config.retrieval_top_k}")
        
        # Initialize HippoRAG với custom config
        print("🧠 Initializing HippoRAG Knowledge Graph...")
        hipporag_instance = HippoRAG(
            global_config=config,
            save_dir=settings.HIPPORAG_SAVE_DIR,
            llm_model_name=config.llm_name,
            embedding_model_name=config.embedding_model_name
        )

        if hasattr(hipporag_instance, 'graph') and hipporag_instance.graph is not None:
            print("Knowledge Graph Statistics:")
            print(f"  - Total nodes: {hipporag_instance.graph.vcount()}")
            print(f"  - Total edges: {hipporag_instance.graph.ecount()}")
            print(f"  - Average degree: {2 * hipporag_instance.graph.ecount() / hipporag_instance.graph.vcount():.2f}")
            # Kiểm tra một số node mẫu
            if hipporag_instance.graph.vcount() > 0:
                print("\nSample nodes:")
                for i in range(min(5, hipporag_instance.graph.vcount())):
                    node_name = hipporag_instance.graph.vs[i]['name'] if 'name' in hipporag_instance.graph.vs.attributes() else f"Node {i}"
                    print(f"  - {node_name}")
        else:
            print("Graph not yet initialized. Run indexing first.")
        
        super().__init__(
            hipporag=hipporag_instance,
            max_docs_per_query=max_docs_per_query,
            **kwargs
        )
        
        print("✅ HippoRAG Retriever initialized successfully with Vietnamese Law customization")
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Required method from BaseRetriever"""
        return self._hipporag_search(query)
    
    def _hipporag_search(self, query: str) -> List[Document]:
        """
        Sử dụng HippoRAG để search với knowledge graph
        
        Returns:
            List[Document]: Documents với enhanced citations
        """
        try:
            print(f"🧠 HippoRAG searching for: {query}")
            
            # Call HippoRAG API
            queries = [query]
            rag_results = self.hipporag.rag_qa(queries=queries)
            
            if not rag_results or not rag_results[0]:
                print("⚠️ No results from HippoRAG")
                return []
            
            documents = []
            query_solution = rag_results[0][0]  # First query's first solution
            
            # Extract answer
            answer = query_solution.answer
            
            # Process each document trong results
            for doc_idx, doc_text in enumerate(query_solution.docs[:self.max_docs_per_query]):
                # Parse citation từ document title
                citation_info = self._parse_citation(doc_text)
                
                # Create metadata
                metadata = {
                    "_source": "hipporag",
                    "_query": query,
                    "_answer": answer,
                    "legal_citation": citation_info["formatted_citation"],
                    "law_id": citation_info["law_id"],
                    "article_id": citation_info["article_id"],
                    "clause_id": citation_info.get("clause_id"),
                    "has_legal_citation": True,
                    "_retrieval_method": "hipporag_knowledge_graph"
                }
                
                # Clean document content (remove citation header)
                clean_content = self._clean_document(doc_text)
                
                # Create Document
                doc = Document(
                    page_content=clean_content,
                    metadata=metadata
                )
                documents.append(doc)
                
                print(f"📄 HippoRAG Doc {doc_idx+1}: {citation_info['formatted_citation']}")
            
            print(f"✅ HippoRAG retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            print(f"❌ HippoRAG search failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _parse_citation(self, doc_text: str) -> Dict[str, Any]:
        """
        Parse citation từ document text
        
        Format: [Nghị định 168-2024-NĐ-CP] Điều 32 Mục 16
        """
        citation_info = {
            "law_id": None,
            "article_id": None,
            "clause_id": None,
            "formatted_citation": "Không xác định nguồn"
        }
        
        try:
            # Extract title line (first line)
            lines = doc_text.split('\n')
            if not lines:
                return citation_info
            
            title_line = lines[0].strip()
            
            # Parse law ID từ brackets [...]
            law_match = re.search(r'\[(.*?)\]', title_line)
            if law_match:
                law_id_raw = law_match.group(1)
                # Fix encoding issues
                law_id = DOCUMENT_MAPPING.get(law_id_raw, law_id_raw)
                citation_info["law_id"] = law_id
            
            # Parse article (Điều X)
            article_match = re.search(r'Điều\s+(\d+)', title_line)
            if article_match:
                citation_info["article_id"] = article_match.group(1)
            
            # Parse clause (Mục/Khoản X)
            clause_match = re.search(r'(?:Mục|Khoản)\s+(\d+)', title_line)
            if clause_match:
                citation_info["clause_id"] = clause_match.group(1)
            
            # Build formatted citation
            parts = []
            if citation_info["law_id"]:
                parts.append(citation_info["law_id"])
            if citation_info["article_id"]:
                parts.append(f"Điều {citation_info['article_id']}")
            if citation_info["clause_id"]:
                parts.append(f"Khoản {citation_info['clause_id']}")
            
            if parts:
                citation_info["formatted_citation"] = ", ".join(parts)
            
        except Exception as e:
            print(f"⚠️ Citation parsing failed: {e}")
        
        return citation_info
    
    def _clean_document(self, doc_text: str) -> str:
        """Remove citation header và clean document"""
        lines = doc_text.split('\n')
        
        # Remove first line (citation header)
        if lines and re.match(r'\[.*?\]', lines[0]):
            lines = lines[1:]
        
        # Join and clean
        cleaned = '\n'.join(lines).strip()
        return cleaned


def get_hipporag_retriever(max_docs_per_query: int = 3) -> HippoRAGRetriever:
    """
    Factory function để tạo HippoRAG retriever
    
    Args:
        max_docs_per_query: Số documents tối đa cho mỗi query
    
    Returns:
        HippoRAGRetriever instance
    """
    return HippoRAGRetriever(max_docs_per_query=max_docs_per_query)
