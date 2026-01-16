# src/chatbot_core.py
from typing import Dict, Any, List, Optional
from enum import Enum
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from operator import itemgetter

# Core imports
from src.generation.openai_generator import get_llm
from src.retrieval.enhanced_retriever import (
    create_vector_only_retriever,
    create_hipporag_only_retriever, 
    format_qdrant_citation
)
from src.retrieval.vietnamese_law_prompts import get_vietnamese_law_prompts
from src.guardrails.injection_detector import is_prompt_injection
from src.guardrails.content_filter import is_sensitive_content
from config import settings

# Query Types Enum for better classification
class QueryType(Enum):
    GREETING = "greeting"
    SIMPLE_LEGAL = "simple_legal"
    COMPLEX_LEGAL = "complex_legal"
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"
    WEB_SEARCH = "web_search"


# STRICT_QA_SYSTEM_PROMPT - 5-section structured response format with citation enforcement
STRICT_QA_SYSTEM_PROMPT = """Bạn là trợ lý pháp luật chuyên về luật giao thông Việt Nam. Hãy trả lời câu hỏi CHỈ DỰA TRÊN các tài liệu được cung cấp.

QUY TẮC BẮT BUỘC:
1. CHỈ sử dụng thông tin từ các tài liệu được cung cấp
2. PHẢI trích dẫn nguồn cụ thể (Nghị định, Điều, Khoản, Điểm)
3. Nếu không tìm thấy thông tin trong tài liệu, nói rõ "Không tìm thấy thông tin trong cơ sở dữ liệu"
4. KHÔNG bịa ra thông tin hoặc dùng kiến thức bên ngoài

ĐỊNH DẠNG TRẢ LỜI (5 PHẦN):

## 1. TÓM TẮT NHANH
[1-2 câu trả lời trực tiếp câu hỏi]

## 2. CHI TIẾT QUY ĐỊNH
[Liệt kê các quy định liên quan với mức phạt/điều kiện cụ thể]

## 3. TRÍCH DẪN NGUỒN
[Ghi rõ: Nghị định số..., Điều..., Khoản..., Điểm...]

## 4. LƯU Ý QUAN TRỌNG
[Các trường hợp đặc biệt hoặc ngoại lệ nếu có]

## 5. CÂU HỎI LIÊN QUAN
[1-2 câu hỏi gợi ý người dùng có thể quan tâm]
"""


class ChatbotCore:
    def __init__(self, use_memory: bool = False):
        """
        Initialize Optimized Chatbot Core
        
        Args:
            use_memory: Nếu True, sử dụng memory để lưu và truy xuất conversation history.
                       Nếu False, tắt tất cả memory functionality (default: False)
        """
        print("🚀 Initializing Optimized Chatbot Core...")
        print(f"   Memory: {'ENABLED' if use_memory else 'DISABLED'}")
        
        # Store memory flag
        self.use_memory = use_memory
        
        # Core models - OpenAI client (not langchain)
        self.openai_client = get_llm()
        self.embeddings = HuggingFaceEmbeddings(model_name="AITeamVN/Vietnamese_Embedding")
        
        # Initialize retrievers (selective initialization)
        print("📚 Initializing retrievers...")
        self.vector_retriever = create_vector_only_retriever(self.embeddings)
        self.hipporag_retriever = create_hipporag_only_retriever()
        
        # Default retriever based on query type
        self.current_retriever = self.vector_retriever
        
        # Build optimized chains for different query types
        self.chains = self._build_specialized_chains()
        
        # Simple memory for follow-ups (only if memory is enabled)
        if self.use_memory:
            self.conversation_context = {}
        else:
            self.conversation_context = None
        
        print("✅ Chatbot Core initialized!")
    
    def _call_openai(self, messages: list, max_tokens: int = 3000) -> str:
        """
        Helper method to call OpenAI API directly
        Replacement for langchain's LLM invocation
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ OpenAI API call failed: {e}")
            return f"Lỗi khi gọi API: {str(e)}"
    
    def _build_specialized_chains(self):
        """Build specialized chains for different query types."""
        chains = {}
        
        # Greeting chain (no retrieval needed)
        chains[QueryType.GREETING] = self._build_greeting_chain()
        
        # HippoRAG chain - PRIMARY chain cho ALL legal queries (retrieve 60 docs + strict prompt)
        hipporag_chain = self._build_hipporag_chain()
        chains[QueryType.SIMPLE_LEGAL] = hipporag_chain
        chains[QueryType.COMPLEX_LEGAL] = hipporag_chain
        
        # Conversational chain (minimal retrieval)
        chains[QueryType.CONVERSATIONAL] = self._build_conversational_chain()
        
        # Web search chain (external search)
        chains[QueryType.WEB_SEARCH] = self._build_web_search_chain()
        
        return chains
    
    def _build_greeting_chain(self):
        """Build chain for greetings and simple interactions."""
        def greet(inputs):
            """Simple greeting response using OpenAI"""
            question = inputs.get("question", "")
            
            greeting_prompt = f"""Bạn là trợ lý AI thân thiện về luật giao thông Việt Nam.
            
Người dùng: {question}
            
Hãy trả lời CỰC KỲ NGẮN GỌN (1-2 câu), thân thiện. 
KHÔNG liệt kê quy định.
KHÔNG đưa ra "câu hỏi gợi ý".
CHỈ chào lại hoặc cảm ơn đơn giản."""
            
            messages = [{"role": "user", "content": greeting_prompt}]
            return self._call_openai(messages, max_tokens=100)
        
        return RunnableLambda(greet)
    
    def _build_hipporag_chain(self):
        """
        Build HippoRAG chain - PRIMARY chain for all legal queries.
        Uses HippoRAG retriever (num_to_retrieve=60) + STRICT_QA_SYSTEM_PROMPT
        """
        def hipporag_qa(inputs):
            """HippoRAG-based QA with strict citation requirements"""
            question = inputs.get("question", "")
            
            print(f"🦛 HippoRAG Chain processing: {question[:50]}...")
            
            # Step 1: Retrieve documents using HippoRAG (retrieves 60 docs internally)
            try:
                docs = self.hipporag_retriever.invoke(question)
                print(f"   Retrieved {len(docs)} documents from HippoRAG")
            except Exception as e:
                print(f"❌ HippoRAG retrieval failed: {e}")
                return f"Lỗi truy xuất dữ liệu: {str(e)}"
            
            if not docs:
                return "Không tìm thấy thông tin trong cơ sở dữ liệu về câu hỏi này."
            
            # Step 2: Format context from top 30 documents
            top_docs = docs[:30]
            context_parts = []
            for i, doc in enumerate(top_docs, 1):
                content = doc.page_content
                metadata = doc.metadata
                
                # Format citation from metadata
                citation = format_qdrant_citation(metadata)
                context_parts.append(f"[Tài liệu {i}]\n{content}\n[Nguồn: {citation}]")
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Step 3: Generate answer using OpenAI with STRICT_QA_SYSTEM_PROMPT
            messages = [
                {"role": "system", "content": STRICT_QA_SYSTEM_PROMPT},
                {"role": "user", "content": f"""TÀI LIỆU THAM KHẢO:
{context}

CÂU HỎI: {question}

Hãy trả lời theo đúng định dạng 5 phần đã quy định."""}
            ]
            
            response = self._call_openai(messages, max_tokens=3000)
            print(f"✅ HippoRAG Chain completed")
            
            return response
        
        return RunnableLambda(hipporag_qa)
    
    def _build_conversational_chain(self):
        """Build chain for conversational queries."""
        def conversational_response(inputs):
            question = inputs.get("question", "")
            previous_context = inputs.get("previous_context", "")
            
            conv_prompt = f"""Bạn là trợ lý AI thân thiện chuyên về luật giao thông Việt Nam.
            
Ngữ cảnh trước: {previous_context}
            
Người dùng: {question}
            
Hãy trả lời tự nhiên, thân thiện như cuộc trò chuyện bình thường. Chỉ trả lời chung chung, ngắn gọn, không cần trích dẫn điều luật."""
            
            messages = [{"role": "user", "content": conv_prompt}]
            return self._call_openai(messages, max_tokens=500)
        
        return RunnableLambda(conversational_response)
    
    def _build_web_search_chain(self):
        """Build chain for web search queries."""
        def web_search_response(inputs):
            question = inputs.get("question", "")
            
            try:
                from src.retrieval.web_search import get_web_search_tool
                web_tool = get_web_search_tool()
                results = web_tool.invoke(question)
                search_results = "\n".join([f"- {r.get('content', '')[:200]}..." for r in results[:3]])
            except Exception as e:
                search_results = f"Lỗi tìm kiếm: {e}"
            
            web_prompt = f"""Bạn là trợ lý AI tìm kiếm thông tin mới nhất về luật giao thông Việt Nam.
            
Kết quả tìm kiếm:
{search_results}
            
Câu hỏi: {question}
            
Hãy:
- LIỆT KÊ NGẮN GỌN các quy định mới nhất (dạng bullet points)
- Trích dẫn nguồn rõ ràng
- KHÔNG giải thích chi tiết
- Chỉ thông tin quan trọng nhất"""
            
            messages = [{"role": "user", "content": web_prompt}]
            return self._call_openai(messages, max_tokens=1000)
        
        return RunnableLambda(web_search_response)
    
    def process_query(self, question: str, chat_history: list = None, user_id: str = None):
        """
        Process query using HippoRAG chain ONLY.
        No classification - all queries go through HippoRAG.
        """
        print(f"🟢 process_query: {question=}")
        
        try:
            # Check for safety
            if is_prompt_injection(question):
                return "Xin lỗi, tôi không thể xử lý yêu cầu này."
            
            if is_sensitive_content(question):
                return "Xin lỗi, câu hỏi này nằm ngoài phạm vi hỗ trợ của tôi."
            
            # ALWAYS use HippoRAG chain for all queries
            print("🦛 Using HippoRAG chain for query processing")
            response = self.chains[QueryType.SIMPLE_LEGAL].invoke({"question": question})
            
            print("✅ Query processed successfully")
            return response
            
        except Exception as e:
            import traceback
            print("❌ ERROR in process_query:")
            traceback.print_exc()
            return f"Xin lỗi, có lỗi xảy ra: {e}"
    
    def clear_context(self, user_id: str):
        """Clear conversation context for user."""
        if self.conversation_context and user_id in self.conversation_context:
            del self.conversation_context[user_id]
    
    def get_conversation_stats(self) -> Dict[str, int]:
        """Get conversation statistics."""
        return {
            "total_sessions": len(self.conversation_context) if self.conversation_context else 0,
            "query_types_supported": len(QueryType)
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get chatbot system information."""
        return {
            "query_types": [qt.value for qt in QueryType],
            "active_chains": list(self.chains.keys()),
            "conversation_sessions": len(self.conversation_context) if self.conversation_context else 0,
            "primary_retriever": "HippoRAG",
            "memory_enabled": self.use_memory
        }
