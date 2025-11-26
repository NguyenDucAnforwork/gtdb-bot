# src/chatbot_core.py
from typing import Dict, Any, List, Optional
from enum import Enum
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
from src.retrieval.query_transformer import create_query_transformer
from src.retrieval.reranker import create_reranker
from src.guardrails.injection_detector import is_prompt_injection
from src.guardrails.content_filter import is_sensitive_content
from src.routing import QueryRouter
from src.memory import MemoryManager
from src.persona.prompts import get_chat_prompt_template, SYSTEM_PROMPTS
from src.persona.admin_bot import AdminBot
from config import settings

# Query Types Enum for better classification
class QueryType(Enum):
    GREETING = "greeting"
    SIMPLE_LEGAL = "simple_legal"
    COMPLEX_LEGAL = "complex_legal"
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"
    WEB_SEARCH = "web_search"


class ChatbotCore:
    def __init__(self):
        print("🚀 Initializing Optimized Chatbot Core...")
        
        # Core models
        self.llm = get_llm()
        self.embeddings = HuggingFaceEmbeddings(model_name="AITeamVN/Vietnamese_Embedding")
        
        # Memory manager for legal queries only
        self.memory_manager = MemoryManager()
        
        # CSGT Bot for specialized commands (lazy loading)
        self._csgt_bot = None
        
        # Admin Bot for document management (EP-03)
        self.admin_bot = AdminBot(chatbot_core=self)
        
        # Query classification
        self.query_classifier = self._build_query_classifier()
        
        # Initialize retrievers (selective initialization)
        print("📚 Initializing retrievers...")
        self.vector_retriever = create_vector_only_retriever(self.embeddings)
        self.hipporag_retriever = create_hipporag_only_retriever()
        
        # Default retriever based on query type
        self.current_retriever = self.vector_retriever
        
        # Build optimized chains for different query types
        self.chains = self._build_specialized_chains()
        
        # Simple memory for follow-ups (no complex caching)
        self.conversation_context = {}
        
        print("✅ Chatbot Core initialized!")

    @property
    def csgt_bot(self):
        """Lazy loading cho CSGTBot"""
        if self._csgt_bot is None:
            from src.persona.csgt_bot import CSGTBot
            self._csgt_bot = CSGTBot(self)
        return self._csgt_bot

    def _build_memory_validator(self):
        """Build LLM chain to validate if memory context can answer the question."""
        validator_prompt = ChatPromptTemplate.from_template(
            """Bạn là một chuyên gia đánh giá chất lượng câu trả lời.

NGỮ CẢNH TỪ BỘ NHỚ CUỘC TRÒ CHUYỆN:
{memory_context}

CÂU HỎI CỦA NGƯỜI DÙNG: {question}

NHIỆM VỤ: Đánh giá xem ngữ cảnh trên có ĐỦ THÔNG TIN để trả lời CỤ THỂ và CHÍNH XÁC câu hỏi không?

TIÊU CHÍ ĐÁNH GIÁ:
✅ CÓ ĐỦ THÔNG TIN nếu:
- Ngữ cảnh chứa TRỰC TIẾP câu trả lời với SỐ LIỆU CỤ THỂ (mức phạt, tốc độ, v.v.)
- Câu hỏi là "tổng cộng/cộng lại" và ngữ cảnh có ĐẦY ĐỦ các con số cần tính
- Câu hỏi hỏi lại về điều đã được nói trước đó trong ngữ cảnh

❌ KHÔNG ĐỦ THÔNG TIN nếu:
- Ngữ cảnh chỉ có câu hỏi, KHÔNG có câu trả lời
- Ngữ cảnh trả lời MƠ HỒ, không có số liệu cụ thể
- Câu hỏi yêu cầu thông tin MỚI hoàn toàn không có trong ngữ cảnh
- Ngữ cảnh có câu trả lời nhưng KHÔNG LIÊN QUAN đến câu hỏi hiện tại

VÍ DỤ:
1. Ngữ cảnh: "Mức phạt không đội mũ bảo hiểm là 400k-600k"
   Câu hỏi: "Mức phạt không đội mũ bảo hiểm là bao nhiêu?"
   → CÓ ĐỦ (trực tiếp trả lời với số cụ thể)

2. Ngữ cảnh: "Mức phạt không đội mũ bảo hiểm?" (chỉ là câu hỏi)
   Câu hỏi: "Mức phạt là bao nhiêu?"
   → KHÔNG ĐỦ (ngữ cảnh không có câu trả lời)

3. Ngữ cảnh: "Phạt mũ: 400k-600k\nPhạt bằng lái: 4M-6M"
   Câu hỏi: "Vậy tổng cộng bao nhiêu?"
   → CÓ ĐỦ (có đủ số liệu để tính tổng)

4. Ngữ cảnh: "Phạt không đội mũ bảo hiểm"
   Câu hỏi: "Còn nếu không có bằng lái thì sao?"
   → KHÔNG ĐỦ (câu hỏi về thông tin mới không có trong ngữ cảnh)

CHỈ TRẢ LỜI BẰNG MỘT TỪ: "YES" (có đủ) hoặc "NO" (không đủ)"""
        )
        
        return validator_prompt | self.llm | StrOutputParser()

    def _build_query_classifier(self):
        """Build smart query classifier."""
        classifier_prompt = ChatPromptTemplate.from_template(
            """Phân loại câu hỏi sau vào một trong các loại:
            
1. GREETING: CHỈ những câu chào hỏi thuần túy như "xin chào", "hello", "hi", "cảm ơn", "thank you". KHÔNG bao gồm câu có ý định hỏi thông tin.
2. SIMPLE_LEGAL: Hỏi pháp luật đơn giản, CỤ THỂ (ví dụ: "mức phạt không đội mũ bảo hiểm là bao nhiêu?", "tốc độ tối đa trong khu dân cư?")
3. COMPLEX_LEGAL: Hỏi pháp luật PHỨC TẠP (so sánh giữa các năm, nhiều điều kiện kết hợp, phân tích sâu)
4. CONVERSATIONAL: Trò chuyện chung chung, KHÔNG CỤ THỂ, KHÔNG hỏi về quy định/mức phạt cụ thể
5. WEB_SEARCH: Hỏi về "quy định MỚI NHẤT", "thay đổi gần đây", "tin tức", "hiện nay"

QUY TẮC QUAN TRỌNG:
- Nếu câu hỏi có "mức phạt", "quy định", "luật" + nội dung CỤ THỂ -> SIMPLE_LEGAL
- Nếu hỏi về "quy định mới nhất", "thay đổi gì" NHƯNG KHÔNG có nội dung cụ thể -> WEB_SEARCH
- "Xin chào, tôi muốn hỏi về X" -> CONVERSATIONAL (vì chỉ có ý định hỏi chung chung)
- "Tôi đang quan tâm về quy định mới nhất" -> WEB_SEARCH (vì chung chung, cần tìm kiếm)

Câu hỏi: "{question}"

Chỉ trả lời bằng MỘT TỪ: GREETING, SIMPLE_LEGAL, COMPLEX_LEGAL, CONVERSATIONAL, WEB_SEARCH"""
        )
        
        return classifier_prompt | self.llm | StrOutputParser()
    
    def _build_specialized_chains(self):
        """Build specialized chains for different query types."""
        chains = {}
        
        # Greeting chain (no retrieval needed)
        chains[QueryType.GREETING] = self._build_greeting_chain()
        
        # Simple legal chain (MEMORY FIRST -> vector search -> web search -> hipporag)
        chains[QueryType.SIMPLE_LEGAL] = self._build_simple_legal_chain()
        
        # Complex legal chain (HippoRAG + vector)
        chains[QueryType.COMPLEX_LEGAL] = self._build_complex_legal_chain()
        
        # Conversational chain (minimal retrieval)
        chains[QueryType.CONVERSATIONAL] = self._build_conversational_chain()
        
        # Web search chain (external search)
        chains[QueryType.WEB_SEARCH] = self._build_web_search_chain()
        
        return chains
    
    def _build_greeting_chain(self):
        """Build chain for greetings and simple interactions with persona support."""
        def persona_greeting(inputs):
            question = inputs["question"]
            persona_key = inputs.get("persona_key", "default")
            
            # Get system prompt from prompts.py
            sys_content = SYSTEM_PROMPTS.get(persona_key, SYSTEM_PROMPTS["default"])
            
            greeting_prompt = ChatPromptTemplate.from_template(
                f"""{sys_content}

Người dùng: {{question}}

Trả lời ngắn gọn, phù hợp với vai trò."""
            )
            
            chain = greeting_prompt | self.llm | StrOutputParser()
            return chain.invoke({"question": question})
        
        return RunnableLambda(persona_greeting)
    
    def _build_simple_legal_chain(self):
        simple_legal_prompt = ChatPromptTemplate.from_template(
            """Bạn là chuyên gia luật giao thông Việt Nam. Trả lời CHÍNH XÁC dựa trên thông tin có sẵn.
            
THÔNG TIN:
{context}
            
CÂU HỎI: {question}
            
HÃY TRẢ LỜI:
- Nếu thông tin đủ: Đưa ra câu trả lời CỤ THỂ và RÕ RÀNG (số tiền phạt, tốc độ, v.v.)
- Nếu cần TÍNH TOÁN (như "tổng cộng"): Hãy TÍNH TOÁN và đưa ra kết quả cụ thể
- Trích dẫn nguồn pháp lý nếu có (Nghị định, Điều, Khoản)
- Giọng như cuộc trò chuyện tự nhiên, NGẮN GỌN

LƯU Ý: Nếu thông tin KHÔNG đủ, hãy nói thẳng "Tôi không tìm thấy thông tin cụ thể về {question} trong cơ sở dữ liệu."""
        )
        """Build chain with MEMORY as FIRST priority retriever and persona support."""
        def get_persona_prompt(persona_key: str) -> ChatPromptTemplate:
            # Use get_chat_prompt_template from prompts.py
            return get_chat_prompt_template(persona_key)
        
        def persona_legal_response(inputs):
            question = inputs["question"]
            context = inputs["context"]
            persona_key = inputs.get("persona_key", "default")
            
            # Get persona-specific prompt
            prompt_template = get_persona_prompt(persona_key)
            
            # Apply the prompt with context and question
            chain = prompt_template | self.llm | StrOutputParser()
            return chain.invoke({"context": context, "question": question})
        
#         def smart_retrieval_with_memory_first(inputs):
#             """MEMORY FIRST retrieval strategy with strict validation."""
#             question = inputs["question"]
#             user_id = inputs.get("user_id", "")
            
#             # ✅ STEP 0: Try MEMORY FIRST (highest priority)
#             if user_id:
#                 print("🧠 Checking if MEMORY can answer the question...")
                
#                 try:
#                     memory_context = self.memory_manager.get_context(user_id, limit=5)
                    
#                     if memory_context and memory_context.strip():
#                         # Use memory manager's validator
#                         is_sufficient = self.memory_manager.validate_memory_sufficiency(
#                             memory_context, question
#                         )
                        
#                         if is_sufficient:
#                             print("✅ MEMORY has sufficient information! Using memory directly.")
                            
#                             # Generate answer from memory using LLM
#                             memory_answer_prompt = ChatPromptTemplate.from_template(
#                                 """Dựa vào NGỮ CẢNH CUỘC TRÒ CHUYỆN dưới đây, hãy trả lời câu hỏi một cách TỰ NHIÊN và CỤ THỂ.

# NGỮ CẢNH:
# {memory_context}

# CÂU HỎI: {question}

# YÊU CẦU:
# - Nếu cần TÍNH TOÁN (như "tổng cộng"), hãy TÍNH và đưa ra KẾT QUẢ CỤ THỂ
# - Trả lời NGẮN GỌN, giọng điệu TỰ NHIÊN như đang trò chuyện
# - KHÔNG cần trích dẫn nguồn vì đây là thông tin từ cuộc trò chuyện trước
# - CHỈ trả lời dựa trên thông tin có trong ngữ cảnh

# TRẢ LỜI:"""
#                             )
                            
#                             answer_chain = memory_answer_prompt | self.llm | StrOutputParser()
#                             memory_answer = answer_chain.invoke({
#                                 "memory_context": memory_context,
#                                 "question": question
#                             })
                            
#                             return f"[Từ cuộc trò chuyện trước]\n{memory_answer}"
#                         else:
#                             print("⚠️ MEMORY validation failed - information insufficient or irrelevant")
#                     else:
#                         print("ℹ️ No memory context available")
                        
#                 except Exception as e:
#                     print(f"❌ Memory retrieval error: {e}")
#             else:
#                 print("ℹ️ No user_id provided, skipping memory retrieval")
            
#             # Helper function to check if response is insufficient
#             def is_insufficient_response(response_text):
#                 insufficient_indicators = [
#                     "không thể xác định", "không có thông tin", "không tìm thấy",
#                     "dựa trên tài liệu", "do không có thông tin trong tài liệu",
#                     "tôi không thể trích dẫn", "không thể nêu rõ", "không có điều khoản", "rất tiếc",
#                     "bạn cần tham khảo", "không đề cập", "thông tin bạn cung cấp không",
#                     "các văn bản quy phạm pháp luật khác", "chỉ quy định chung"
#                 ]
#                 response_lower = response_text.lower()
                
#                 has_indicator = any(indicator in response_lower for indicator in insufficient_indicators)
#                 has_specific_info = any(char.isdigit() for char in response_text)
                
#                 return has_indicator or not has_specific_info
            
#             # ✅ STEP 1: Try Vector retriever
#             try:
#                 query_transformer = create_query_transformer(self.vector_retriever, self.llm)
#                 reranker = create_reranker(query_transformer)
#                 vector_docs = reranker.invoke(question)
                
#                 if vector_docs and len(vector_docs) > 0:
#                     print("📚 Trying vector retriever")
#                     vector_context = []
#                     for doc in vector_docs[:3]:
#                         content = doc.page_content
#                         metadata = doc.metadata
#                         citation = format_qdrant_citation(metadata)
#                         vector_context.append(f"{content}\n[Nguồn: {citation}]")
                    
#                     return "\n\n".join(vector_context)
                
#                 # check if vector_context is sufficient
#                 vector_formatted = "\n\n".join(vector_context)
#                 test_response = self.llm.invoke(
#                     simple_legal_prompt.format(context=vector_formatted, question=question)
#                 )
#             except Exception as e:
#                 print(f"❌ Vector retriever error: {e}")
            
#             return "Xin lỗi, không tìm thấy thông tin chính xác về câu hỏi này."
        
#         return (
#             {
#                 "question": itemgetter("question"),
#                 "user_id": itemgetter("user_id"),
#                 "context": RunnableLambda(smart_retrieval_with_memory_first)
#             }
#             | RunnableLambda(lambda inputs: {
#                 "context": inputs["context"],
#                 "question": inputs["question"],
#                 "persona_key": inputs.get("persona_key", "default")
#             })
#             | RunnableLambda(persona_legal_response)
#         )
        
        def smart_retrieval_with_memory_first(inputs):
            """MEMORY FIRST retrieval strategy with strict validation."""
            question = inputs["question"]
            user_id = inputs.get("user_id", "")
            
            # ✅ STEP 0: Try MEMORY FIRST (highest priority)
            if user_id:
                print("🧠 Checking if MEMORY can answer the question...")
                
                try:
                    memory_context = self.memory_manager.get_context(user_id, limit=5)
                    print(f"🧠 Retrieved memory context: {memory_context}")

                    if memory_context and memory_context.strip():
                        # Use memory manager's validator
                        is_sufficient = self.memory_manager.validate_memory_sufficiency(
                            memory_context, question
                        )
                        
                        if is_sufficient:
                            print("✅ MEMORY has sufficient information! Using memory directly.")
                            
                            # Generate answer from memory using LLM
                            memory_answer_prompt = ChatPromptTemplate.from_template(
                                """Dựa vào NGỮ CẢNH CUỘC TRÒ CHUYỆN dưới đây, hãy trả lời câu hỏi một cách TỰ NHIÊN và CỤ THỂ.

NGỮ CẢNH:
{memory_context}

CÂU HỎI: {question}

YÊU CẦU:
- Nếu cần TÍNH TOÁN (như "tổng cộng"), hãy TÍNH và đưa ra KẾT QUẢ CỤ THỂ
- Trả lời NGẮN GỌN, giọng điệu TỰ NHIÊN như đang trò chuyện
- KHÔNG cần trích dẫn nguồn vì đây là thông tin từ cuộc trò chuyện trước
- CHỈ trả lời dựa trên thông tin có trong ngữ cảnh

TRẢ LỜI:"""
                            )
                            
                            answer_chain = memory_answer_prompt | self.llm | StrOutputParser()
                            memory_answer = answer_chain.invoke({
                                "memory_context": memory_context,
                                "question": question
                            })
                            
                            return f"[Từ cuộc trò chuyện trước]\n{memory_answer}"
                        else:
                            print("⚠️ MEMORY validation failed - information insufficient or irrelevant")
                    else:
                        print("ℹ️ No memory context available")
                        
                except Exception as e:
                    print(f"❌ Memory retrieval error: {e}")
            else:
                print("ℹ️ No user_id provided, skipping memory retrieval")
            
            # Helper function to check if response is insufficient
            def is_insufficient_response(response_text):
                insufficient_indicators = [
                    "không thể xác định", "không có thông tin", "không tìm thấy",
                    "dựa trên tài liệu", "do không có thông tin trong tài liệu",
                    "tôi không thể trích dẫn", "không thể nêu rõ", "không có điều khoản", "rất tiếc",
                    "bạn cần tham khảo", "không đề cập", "thông tin bạn cung cấp không",
                    "các văn bản quy phạm pháp luật khác", "chỉ quy định chung"
                ]
                response_lower = response_text.lower()
                
                has_indicator = any(indicator in response_lower for indicator in insufficient_indicators)
                has_specific_info = any(char.isdigit() for char in response_text)
                
                return has_indicator or not has_specific_info
            
            # ✅ STEP 1: Try Vector retriever
            try:
                query_transformer = create_query_transformer(self.vector_retriever, self.llm)
                reranker = create_reranker(query_transformer)
                vector_docs = reranker.invoke(question)
                
                if vector_docs and len(vector_docs) > 0:
                    print("📚 Trying vector retriever")
                    vector_context = []
                    for doc in vector_docs[:3]:
                        content = doc.page_content
                        metadata = doc.metadata
                        citation = format_qdrant_citation(metadata)
                        vector_context.append(f"{content}\n[Nguồn: {citation}]")
                    
                    vector_formatted = "\n\n".join(vector_context)
                    
                    test_response = self.llm.invoke(
                        simple_legal_prompt.format(context=vector_formatted, question=question)
                    )
                    
                    response_text = test_response.content if hasattr(test_response, 'content') else str(test_response)
                    
                    if not is_insufficient_response(response_text):
                        print("✅ Vector retriever provided sufficient answer")
                        return vector_formatted
                    else:
                        print("⚠️ Vector answer insufficient, trying web search")
            except Exception as e:
                print(f"❌ Vector retriever error: {e}")
            
            # ✅ STEP 2: Try Web search
            try:
                print("🌐 Falling back to web search")
                from src.retrieval.web_search import get_web_search_tool
                web_tool = get_web_search_tool()
                web_results = web_tool.invoke(question)
                
                if web_results:
                    web_content = "\n".join([f"- {r.get('content', '')[:300]}" for r in web_results[:3]])
                    web_context = f"Thông tin tìm kiếm trên web:\n{web_content}"
                    
                    test_response = self.llm.invoke(
                        simple_legal_prompt.format(context=web_context, question=question)
                    )
                    
                    response_text = test_response.content if hasattr(test_response, 'content') else str(test_response)
                    
                    if not is_insufficient_response(response_text):
                        print("✅ Web search provided sufficient answer")
                        return web_context
                    else:
                        print("⚠️ Web search answer insufficient, trying HippoRAG")
            except Exception as e:
                print(f"❌ Web search error: {e}")
            
            # ✅ STEP 3: Try HippoRAG as last resort
            try:
                print("🦄 Falling back to HippoRAG")
                hippo_docs = self.hipporag_retriever.invoke(question)
                if hippo_docs:
                    hippo_context = "\n".join([doc.page_content for doc in hippo_docs[:2]])
                    print("✅ Using HippoRAG as final fallback")
                    return hippo_context
            except Exception as e:
                print(f"❌ HippoRAG error: {e}")
            
            return "Xin lỗi, không tìm thấy thông tin chính xác về câu hỏi này."
        
        return (
            {
                "question": itemgetter("question"),
                "user_id": itemgetter("user_id"),
                "context": RunnableLambda(smart_retrieval_with_memory_first)
            }
            | simple_legal_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _build_complex_legal_chain(self):
        """Build chain for complex legal queries using HippoRAG + vector with memory support."""
        complex_legal_prompt = ChatPromptTemplate.from_template(
            """Bạn là chuyên gia luật cao cấp, giỏi phân tích phức tạp về luật giao thông Việt Nam.
            
THÔNG TIN TỪ KIẾN THỨC GRAPH:
{hippo_context}
            
THÔNG TIN TỪ TÀI LIỆU:
{vector_context}
            
CÂU HỎI PHỨC TẠP: {question}
            
HÃY:
1. KẾT HỢP thông tin từ cả hai nguồn
2. PHÂN TÍCH so sánh, liên kết
3. ĐƯA RA kết luận rõ ràng
4. TRÍCH DẪN đầy đủ căn cứ pháp lý
            
TRẢ LỜI CHI TIẾT:"""
        )
        
        vector_transformer = create_query_transformer(self.vector_retriever, self.llm)
        vector_reranker = create_reranker(vector_transformer)
        
        def format_contexts(inputs):
            question = inputs["question"]
            user_id = inputs.get("user_id", "")
            
            # Check memory first for complex queries too
            if user_id:
                try:
                    memory_context = self.memory_manager.get_context(user_id, limit=5)
                    if memory_context and self.memory_manager.validate_memory_sufficiency(memory_context, question):
                        print("✅ MEMORY sufficient for complex query")
                        return {
                            "question": question,
                            "vector_context": f"[Từ cuộc trò chuyện trước]\n{memory_context}",
                            "hippo_context": ""
                        }
                except Exception as e:
                    print(f"❌ Memory error in complex chain: {e}")
            
            vector_docs = vector_reranker.invoke(question)
            vector_context = "\n".join([doc.page_content for doc in vector_docs[:3]])
            
            try:
                hippo_docs = self.hipporag_retriever.invoke(question)
                hippo_context = "\n".join([doc.page_content for doc in hippo_docs[:3]])
            except Exception as e:
                print(f"HippoRAG error: {e}")
                hippo_context = "Không có thông tin từ knowledge graph"
            
            return {
                "question": question,
                "vector_context": vector_context,
                "hippo_context": hippo_context
            }
        
        return (
            RunnableLambda(format_contexts)
            | complex_legal_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _build_conversational_chain(self):
        """Build chain for conversational queries."""
        conv_prompt = ChatPromptTemplate.from_template(
            """Bạn là trợ lý AI thân thiện chuyên về luật giao thông Việt Nam.
            
Ngữ cảnh trước: {previous_context}
            
Người dùng: {question}
            
Hãy trả lời tự nhiên, thân thiện như cuộc trò chuyện bình thường. Chỉ trả lời chung chung, ngắn gọn, không cần trích dẫn điều luật."""
        )
        
        return conv_prompt | self.llm | StrOutputParser()
    
    def _build_web_search_chain(self):
        """Build chain for web search queries."""
        web_prompt = ChatPromptTemplate.from_template(
            """Bạn là trợ lý AI tìm kiếm thông tin mới nhất về luật giao thông Việt Nam.
            
Kết quả tìm kiếm:
{search_results}
            
Câu hỏi: {question}
            
Hãy:
- LIỆT KÊ NGẮN GỌN các quy định mới nhất (dạng bullet points)
- Trích dẫn nguồn rõ ràng
- KHÔNG giải thích chi tiết
- Chỉ thông tin quan trọng nhất"""
        )
        
        def web_search_and_format(inputs):
            question = inputs["question"]
            try:
                # Web search functionality - simplified for now
                search_results = f"Tìm kiếm web cho: {question}\n(Chức năng web search sẽ được triển khai sau)"
                return {"question": question, "search_results": search_results}
            except Exception as e:
                return {"question": question, "search_results": f"Lỗi tìm kiếm: {e}"}
        
        return RunnableLambda(web_search_and_format) | web_prompt | self.llm | StrOutputParser()
    
    def process_query(self, question: str, user_id: str = None, persona_key: str = "default", force_hipporag: bool = False):
        """Enhanced query processing with MEMORY and CSGT commands support."""
        print(f"🟢 process_query: {question=}, {user_id=}, {persona_key=}, {force_hipporag=}")
        
        try:
            # Step 0: Auto-enable HippoRAG for hipporag persona
            if persona_key == "hipporag":
                force_hipporag = True
                print("🦄 HippoRAG persona detected - automatically enabling force_hipporag=True")
            
            # Step 0: Force HippoRAG mode - bypass everything else, only use HippoRAG
            if force_hipporag:
                print("🎯 Force HippoRAG mode - using HippoRAG retriever only, no vector search")
                
                # Get documents from HippoRAG only
                hipporag_docs = self.hipporag_retriever.invoke(question)
                
                # Format HippoRAG results
                if hipporag_docs:
                    formatted_context = "\n\n".join([
                        f"📄 Tài liệu {i+1}: {format_qdrant_citation(doc.metadata)}\n{doc.page_content}"
                        for i, doc in enumerate(hipporag_docs)
                    ])
                else:
                    formatted_context = "Không tìm thấy tài liệu phù hợp."
                
                # Use default persona prompt for HippoRAG-only response
                prompt_template = get_chat_prompt_template("default")
                chain = prompt_template | self.llm | StrOutputParser()
                
                response = chain.invoke({
                    "question": question,
                    "context": formatted_context
                })
                
                return response
            
            # Step 1: Check for CSGT special commands first
            if persona_key == "csgt" and question.strip().startswith("/"):
                return self._handle_csgt_commands(question.strip(), user_id)
            
            # Step 2: Classify query
            query_type = self._classify_query(question, [])
            print(f"🎯 Query classified as: {query_type.value}")
            
            # Step 2: Process with appropriate chain
            if query_type in self.chains:
                if query_type == QueryType.CONVERSATIONAL:
                    response = self.chains[query_type].invoke({
                        "question": question,
                        "previous_context": "",
                        "persona_key": persona_key
                    })
                elif query_type in [QueryType.SIMPLE_LEGAL, QueryType.COMPLEX_LEGAL]:
                    # ✅ Pass user_id and persona_key to legal chains
                    response = self.chains[query_type].invoke({
                        "question": question,
                        "user_id": user_id or "",
                        "persona_key": persona_key
                    })
                else:
                    response = self.chains[query_type].invoke({
                        "question": question,
                        "persona_key": persona_key
                    })
            else:
                response = "Xin lỗi, tôi không hiểu câu hỏi của bạn."
            
            # Step 3: Add follow-up suggestions
            final_response = self._create_response(response, query_type, question)
            
            print("✅ Query processed successfully")
            return final_response
            
        except Exception as e:
            import traceback
            print("❌ ERROR in process_query:")
            traceback.print_exc()
            return f"Xin lỗi, có lỗi xảy ra: {e}"

    def _handle_csgt_commands(self, command: str, user_id: str = None) -> str:
        """Xử lý các lệnh đặc biệt của CSGT (AC1, AC2)"""
        try:
            if command.startswith("/lookup "):
                keyword = command[8:].strip()  # Bỏ "/lookup "
                return self.csgt_bot.fast_lookup(keyword)
            
            elif command.startswith("/checklist "):
                violation = command[11:].strip()  # Bỏ "/checklist "
                return self.csgt_bot.generate_checklist(violation)
            
            elif command.startswith("/quick "):
                code = command[7:].strip()  # Bỏ "/quick "
                return self.csgt_bot.quick_penalty_check(code)
            
            elif command.startswith("/draft"):
                # Phân tích thông tin từ command (đơn giản hóa)
                if ":" in command:
                    # Format: /draft name:X,plate:Y,violation:Z,...
                    parts = command[6:].strip().split(",")
                    info = {}
                    for part in parts:
                        if ":" in part:
                            key, value = part.split(":", 1)
                            info[key.strip()] = value.strip()
                    return self.csgt_bot.draft_report(info)
                else:
                    return """📄 Để soạn biên bản, vui lòng cung cấp thông tin:
/draft name:Nguyễn Văn A,plate:29A-123.45,violation:Vượt đèn đỏ,time:10:30 21/11/2025,location:Ngã tư Sở"""
            
            elif command == "/help":
                return """👮 LỆNH CSGT KHÁC NHAU:
/lookup [từ khóa] - Tra cứu nhanh mức phạt (≤10s)
/checklist [lỗi] - Tạo checklist lập biên bản  
/quick [mã] - Tra offline (VDR,QTS,KMB,NCN...)
/draft [info] - Soạn mẫu biên bản
/help - Hiển thị menu này

VÍ DỤ:
/lookup vượt đèn đỏ xe máy
/checklist nồng độ cồn
/quick VDR"""
            
            else:
                return f"❌ Lệnh không hợp lệ: {command}\nGõ /help để xem danh sách lệnh."
                
        except Exception as e:
            return f"❌ Lỗi xử lý lệnh CSGT: {e}"
    
    def _handle_admin_commands(self, command: str, user_id: str = None) -> str:
        """Xử lý các lệnh admin quản trị văn bản (EP-03)"""
        try:
            # US1: Document approval workflow
            if command == "/admin pending":
                return self.admin_bot.list_pending_documents()
            
            elif command.startswith("/admin approve "):
                filename = command[15:].strip()
                return self.admin_bot.approve_document(filename, approved_by=user_id or "admin")
            
            elif command.startswith("/admin reject "):
                parts = command[14:].split(maxsplit=1)
                filename = parts[0]
                reason = parts[1] if len(parts) > 1 else ""
                return self.admin_bot.reject_document(filename, reason)
            
            # US2/AC2: Version diff & status management
            elif command.startswith("/admin diff "):
                # Format: /admin diff <doc_name> <v1> <v2>
                parts = command[12:].split()
                if len(parts) >= 3:
                    return self.admin_bot.show_diff(parts[0], parts[1], parts[2])
                return "❌ Format: /admin diff <tên> <phiên_bản_cũ> <phiên_bản_mới>"
            
            elif command.startswith("/admin status "):
                # Format: /admin status <doc_name> <status> [effective_date]
                parts = command[14:].split()
                if len(parts) >= 2:
                    doc_name = parts[0]
                    status = parts[1]
                    effective_date = parts[2] if len(parts) > 2 else None
                    return self.admin_bot.update_status(doc_name, status, effective_date)
                return "❌ Format: /admin status <tên> <trạng_thái> [ngày_hiệu_lực]"
            
            # AC1/AC3: Coverage & Indexing
            elif command == "/admin coverage":
                return self.admin_bot.check_coverage()
            
            elif command.startswith("/admin index "):
                filename = command[13:].strip()
                return self.admin_bot.trigger_indexing(filename)
            
            elif command == "/admin index-all":
                return self.admin_bot.trigger_indexing()
            
            # Stats & Help
            elif command == "/admin stats":
                return self.admin_bot.get_admin_stats()
            
            elif command in ["/admin help", "/admin"]:
                return self.admin_bot.get_help()
            
            else:
                return f"❌ Lệnh admin không hợp lệ: {command}\nGõ /admin help để xem danh sách lệnh."
                
        except Exception as e:
            return f"❌ Lỗi xử lý lệnh admin: {e}"
    
    def _classify_query(self, question: str, chat_history: list) -> QueryType:
        """Classify query using LLM."""
        try:
            classification = self.query_classifier.invoke({"question": question})
            classification = classification.strip().upper()
            
            type_mapping = {
                "GREETING": QueryType.GREETING,
                "SIMPLE_LEGAL": QueryType.SIMPLE_LEGAL,
                "COMPLEX_LEGAL": QueryType.COMPLEX_LEGAL,
                "CONVERSATIONAL": QueryType.CONVERSATIONAL,
                "TECHNICAL": QueryType.CONVERSATIONAL,
                "WEB_SEARCH": QueryType.WEB_SEARCH
            }
            
            return type_mapping.get(classification, QueryType.CONVERSATIONAL)
        except Exception as e:
            print(f"⚠️ Classification error: {e}")
            return QueryType.CONVERSATIONAL
    
    def _create_response(self, content: str, query_type: QueryType, original_question: str = "") -> str:
        """Create formatted response with natural follow-up suggestions."""
        # Skip follow-ups if answer came from memory (starts with "[Từ cuộc trò chuyện trước]")
        if content.startswith("[Từ cuộc trò chuyện trước]"):
            return content
        
        if query_type in [QueryType.SIMPLE_LEGAL, QueryType.COMPLEX_LEGAL]:
            has_suggestions = any(phrase in content.lower() for phrase in ['câu hỏi gợi ý', 'follow-up', 'bạn có thể', 'nếu bạn'])
            is_insufficient = any(phrase in content.lower() for phrase in ['không tìm thấy', 'không có thông tin', 'xin lỗi'])
            is_too_short = len(content) < 200
            
            if not has_suggestions and not is_insufficient and not is_too_short:
                follow_ups = self._generate_follow_ups(query_type, original_question)
                if follow_ups:
                    return f"{content}\n\n{follow_ups}"
        return content
    
    def _generate_follow_ups(self, query_type: QueryType, original_question: str = "") -> str:
        """Generate natural follow-up suggestions using LLM."""
        followup_prompt = ChatPromptTemplate.from_template(
            """Dựa trên câu hỏi gốc, hãy tạo thêm một gợi ý tiếp theo tự nhiên và hấp dẫn.
            
Câu hỏi gốc: "{original_question}"
Loại: {query_type}
            
Tạo các câu gợi ý như:
- "Nếu bạn muốn biết thêm về X, tôi có thể giải thích chi tiết hơn"
- "Bạn cũng có thể hỏi về Y nếu quan tâm" 
- "Tôi cũng có thể hướng dẫn về Z nếu bạn cần"
            
Hãy viết tự nhiên, thân thiện."""
        )
        
        try:
            followup_chain = followup_prompt | self.llm | StrOutputParser()
            follow_ups = followup_chain.invoke({
                "query_type": query_type.value,
                "original_question": original_question
            })
            return follow_ups.strip()
        except Exception as e:
            print(f"⚠️ Follow-up generation error: {e}")
            return "- Nếu bạn có thắc mắc gì khác về luật giao thông, tôi sẵn sàng giải đáp!"
    
    def clear_context(self, user_id: str):
        """Clear conversation context for user."""
        if user_id in self.conversation_context:
            del self.conversation_context[user_id]
    
    def get_conversation_stats(self) -> Dict[str, int]:
        """Get conversation statistics."""
        return {
            "total_sessions": len(self.conversation_context),
            "query_types_supported": len(QueryType)
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get chatbot system information."""
        return {
            "query_types": [qt.value for qt in QueryType],
            "active_chains": list(self.chains.keys()),
            "conversation_sessions": len(self.conversation_context),
            "retrievers": {
                "memory": "Priority 0 - Conversation history",
                "vector": "Priority 1 - Qdrant vector store",
                "web": "Priority 2 - Web search",
                "hipporag": "Priority 3 - Knowledge graph"
            }
        }