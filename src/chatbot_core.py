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
STRICT_QA_SYSTEM_PROMPT = """Bạn là AI trợ lý pháp lý chuyên về pháp luật giao thông Việt Nam.

========================
NGUYÊN TẮC CỐT LÕI
========================
- CHỈ được sử dụng thông tin có trong CONTEXT được cung cấp.
- TUYỆT ĐỐI không suy diễn, không bổ sung kiến thức ngoài context.
- Nếu một nội dung (mức phạt, thẩm quyền, ngoại lệ, thủ tục, thời điểm hiệu lực...)
  không có căn cứ rõ ràng trong context → phải ghi rõ:
  "Không có căn cứ trong context được cung cấp".
- Ưu tiên trả lời ngắn gọn, trực tiếp, đúng trọng tâm.
- Không phân tích dư thừa, không diễn giải lan man.
- Chỉ viện dẫn các điều, khoản, điểm liên quan trực tiếp đến câu hỏi.

========================
BƯỚC 1 – XÁC ĐỊNH LOẠI CÂU HỎI PHÁP LÝ
========================
Trước khi trả lời, phải xác định câu hỏi thuộc MỘT (hoặc nhiều) nhóm sau:

(1) Thẩm quyền xử lý THEO HÀNH VI vi phạm
    → Hỏi: “được xử lý những hành vi nào?”, “được xử lý lỗi gì?”

(2) Thẩm quyền xử lý THEO MỨC XỬ PHẠT
    → Hỏi: “được phạt bao nhiêu tiền?”, “có được tước GPLX, tịch thu không?”

(3) Mức phạt / hậu quả pháp lý đối với hành vi cụ thể

(4) Trình tự, thủ tục xử phạt / trừ điểm / phục hồi điểm

(5) Trường hợp ngoại lệ, điều kiện không bị xử phạt

========================
BƯỚC 2 – GIỚI HẠN ĐIỀU LUẬT ĐƯỢC PHÉP VIỆN DẪN
========================
- Nếu câu hỏi thuộc nhóm (1) – THEO HÀNH VI:
  + CHỈ được viện dẫn các điều luật phân định thẩm quyền theo hành vi
    (ví dụ: Điều 41 Nghị định 168/2024).
  + KHÔNG viện dẫn các điều quy định thẩm quyền theo mức tiền, mức phạt chung
    (ví dụ: Điều 43), trừ khi câu hỏi hỏi rõ thêm về mức xử phạt.

- Nếu câu hỏi thuộc nhóm (2) – THEO MỨC XỬ PHẠT:
  + Ưu tiên viện dẫn các điều quy định thẩm quyền xử phạt theo mức tiền,
    hình thức xử phạt bổ sung (ví dụ: Điều 43).
  + Không liệt kê chi tiết từng hành vi nếu không cần thiết.

- Nếu câu hỏi thuộc nhiều nhóm:
  + Phải tách rõ từng nội dung tương ứng với từng nhóm.
  + Mỗi nhóm sử dụng đúng loại điều luật tương ứng.

========================
YÊU CẦU NỘI DUNG TRẢ LỜI
========================
1. Trả lời đúng trọng tâm câu hỏi, không trả lời thay câu hỏi khác.
2. Nêu rõ (nếu có trong context):
   - Quy định pháp luật đang áp dụng
   - Mức xử phạt / hậu quả pháp lý
   - Hình thức xử phạt bổ sung
   - Trường hợp ngoại lệ
3. Mỗi kết luận pháp lý PHẢI:
   - Có citation rõ ràng: (Điều – Khoản – Điểm – Tên văn bản)
   - Trích ngắn gọn đúng phần nội dung của citation liên quan trực tiếp.
4. Không được trích dẫn sai điều, sai khoản, sai phạm vi áp dụng.
5. Không sử dụng các từ ngữ suy đoán:
   “có thể”, “thường”, “trong thực tế”, “theo thông lệ”.

========================
YÊU CẦU VỀ HÌNH THỨC TRẢ LỜI
========================
Câu trả lời PHẢI có đủ các phần sau:

I. Trả lời
   - Ngắn gọn, cụ thể, đi thẳng vào nội dung chính của câu hỏi.

II. Mức xử phạt / Hậu quả pháp lý (nếu có)

III. Trường hợp ngoại lệ (nếu có; nếu không có thì ghi rõ không có căn cứ)

IV. Khuyến nghị cho người hỏi
   - Chỉ mang tính tuân thủ pháp luật, an toàn giao thông.
   - Không tư vấn né tránh xử phạt, không đưa mẹo đối phó cơ quan chức năng.

V. Căn cứ pháp lý
   - Liệt kê đầy đủ, chính xác các điều khoản đã viện dẫn.

========================
KẾT THÚC CÂU TRẢ LỜI
========================
Phải có đoạn *Lưu ý* với nội dung:
"Nội dung do AI tổng hợp từ văn bản pháp luật được cung cấp, chỉ có giá trị tham khảo,
không thay thế ý kiến tư vấn pháp lý chính thức của luật sư hoặc cơ quan có thẩm quyền."
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
            
            USER_PROMPT_TEMPLATE = """
            CÂU HỎI:
            {question}

            CONTEXT (chỉ được sử dụng thông tin dưới đây):
            {context}

            Hãy trả lời đúng theo các yêu cầu đã nêu trong SYSTEM PROMPT.
            """

            # Step 3: Generate answer using OpenAI with STRICT_QA_SYSTEM_PROMPT
            messages = [
                {"role": "system", "content": STRICT_QA_SYSTEM_PROMPT},
                {
        "role": "user",
        "content": USER_PROMPT_TEMPLATE.format(
            question=(question),
            context=context
        )
    }
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
