# src/memory/memory_manager.py
import os
from typing import List, Dict, Any, Optional
from mem0 import Memory
from dotenv import load_dotenv

load_dotenv()


class MemoryManager:
    """Manages conversation memory using Mem0 with Qdrant Cloud."""
    
    def __init__(self):
        """Initialize Memory with Qdrant Cloud configuration."""
        QDRANT_URL = os.getenv("QDRANT_URL")
        QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "temperature": 0,
                    "api_key": OPENAI_API_KEY
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "api_key": OPENAI_API_KEY
                }
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "mem0_chatbot",
                    "url": QDRANT_URL,
                    "api_key": QDRANT_API_KEY,
                }
            },
            "version": "v1.1"
        }
        
        print("🧠 Initializing MemoryManager with Qdrant Cloud...")
        self.memory = Memory.from_config(config)
    
    def get_memories(self, user_id: str, limit: int = 15) -> List[str]:
        """
        Get all memories for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of memory strings
        """
        try:
            res = self.memory.get_all(user_id=user_id, limit=limit)
            all_memories = res.get('results', [])
            
            # Extract just the 'memory' text field
            memories = [mem['memory'] for mem in all_memories]
            
            print(f"🧠 Retrieved {len(memories)} memories for user {user_id}")
            return memories
            
        except Exception as e:
            print(f"⚠️ Error retrieving memories: {e}")
            return []
    
    def get_recent_context(self, user_id: str, n: int = 15) -> str:
        """
        Get recent conversation context formatted for LLM.
        
        Args:
            user_id: User identifier
            n: Number of recent memories to include
            
        Returns:
            Formatted context string
        """
        memories = self.get_memories(user_id, limit=n)
        
        if not memories:
            return ""
        
        # Use last n memories as recent context
        recent_memories = memories[-n:] if len(memories) > n else memories
        context_str = "\n".join([f"- {mem}" for mem in recent_memories])
        
        return context_str
    
    def save_conversation(
        self, 
        user_id: str, 
        query: str, 
        response: str,
        should_save_fn: Optional[callable] = None
    ) -> bool:
        """
        Save conversation to memory with optional validation.
        
        Args:
            user_id: User identifier
            query: User query
            response: Bot response
            should_save_fn: Optional function to validate if should save
            
        Returns:
            True if saved, False otherwise
        """
        # Check if should save using provided function
        if should_save_fn and not should_save_fn(query, response):
            print("❌ Not saved to memory (validation failed)")
            return False
        
        try:
            messages_to_save = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response}
            ]
            
            # Use infer=False to store raw conversation
            self.memory.add(messages_to_save, user_id=user_id, infer=False)
            
            print(f"💾 Saved to Mem0 memory:")
            print(f"   User: {query[:80]}...")
            print(f"   Bot: {response[:80]}...")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error saving to memory: {e}")
            return False
    
    def clear_user_memories(self, user_id: str) -> bool:
        """
        Clear all memories for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.memory.delete_all(user_id=user_id)
            print(f"🧹 Cleared all memories for user {user_id}")
            return True
        except Exception as e:
            print(f"⚠️ Error clearing memories: {e}")
            return False
    
    def add_conversation(self, query: str, response: str, user_id: str) -> bool:
        """
        Add a conversation to memory.
        
        Args:
            query: User query
            response: Bot response
            user_id: User identifier
            
        Returns:
            True if saved successfully
        """
        try:
            messages_to_save = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response}
            ]
            
            # Critical: Use infer=False to store raw conversation
            self.memory.add(messages_to_save, user_id=user_id, infer=False)
            
            print(f"💾 Saved to memory for user {user_id}")
            return True
            
        except Exception as e:
            print(f"⚠️ Error adding conversation: {e}")
            return False
    
    def get_context(self, user_id: str, limit: int = 15) -> str:
        """
        Get formatted conversation context for a user.
        
        Args:
            user_id: User identifier
            limit: Number of recent memories to retrieve
            
        Returns:
            Formatted context string
        """
        return self.get_recent_context(user_id, n=limit)
    
    def validate_memory_sufficiency(self, memory_context: str, question: str) -> bool:
        """
        Use LLM to validate if memory context is sufficient to answer the question.
        
        Args:
            memory_context: Formatted memory context
            question: User question
            
        Returns:
            True if memory has sufficient information
        """
        if not memory_context or not memory_context.strip():
            return False
        
        try:
            from src.generation.openai_generator import get_llm
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            
            llm = get_llm()
            validator_prompt = ChatPromptTemplate.from_template(
                """Bạn là một chuyên gia đánh giá chất lượng câu trả lời.

NGỮ CẢNH TỪ BỘ NHỚ CUỘC TRÒ CHUYỆN:
{memory_context}

CÂU HỎI CỦA NGƯỜI DÙNG: {question}

NHIỆM VỤ: Đánh giá xem ngữ cảnh trên có ĐỦ THÔNG TIN để trả lời CỤ THỂ và CHÍNH XÁC câu hỏi không?

TIÊU CHÍ ĐÁNH GIÁ:
✅ CÓ ĐỦ THÔNG TIN nếu:
- Ngữ cảnh chứa TRỰC TIẾP câu trả lời với SỐ LIỆU CỤ THỂ
- Câu hỏi là "tổng cộng/cộng lại" và ngữ cảnh có ĐẦY ĐỦ các con số cần tính
- Câu hỏi hỏi lại về điều đã được nói trước đó trong ngữ cảnh

❌ KHÔNG ĐỦ THÔNG TIN nếu:
- Ngữ cảnh chỉ có câu hỏi, KHÔNG có câu trả lời
- Ngữ cảnh trả lời MƠ HỒ, không có số liệu cụ thể
- Câu hỏi yêu cầu thông tin MỚI hoàn toàn không có trong ngữ cảnh
- Ngữ cảnh có câu trả lời nhưng KHÔNG LIÊN QUAN đến câu hỏi hiện tại

CHỈ TRẢ LỜI BẰNG MỘT TỪ: "YES" (có đủ) hoặc "NO" (không đủ)"""
            )
            
            chain = validator_prompt | llm | StrOutputParser()
            result = chain.invoke({
                "memory_context": memory_context,
                "question": question
            }).strip().upper()
            
            return result == "YES"
            
        except Exception as e:
            print(f"⚠️ Memory validation error: {e}")
            return False
    
    def should_save_memory(self, query: str, response: str) -> bool:
        """
        Use LLM to decide whether to save conversation to memory.
        
        Args:
            query: User query
            response: Bot response
            
        Returns:
            True if conversation should be saved
        """
        try:
            from src.generation.openai_generator import get_llm
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            
            llm = get_llm()
            memory_prompt = ChatPromptTemplate.from_template(
                """Phân tích cuộc trò chuyện và quyết định có nên lưu vào bộ nhớ dài hạn không?
            
Câu hỏi: "{query}"
Trả lời: "{response}"
            
Trả lời "YES" nếu:
- Chứa thông tin pháp lý quan trọng (mức phạt, quy định, điều luật cụ thể)
- Chứa thông tin cá nhân của người dùng (tên, nghề nghiệp, địa điểm)
- Là cuộc trò chuyện có giá trị tham khảo lâu dài
- Người dùng đang hỏi về một chủ đề cụ thể cần nhớ
            
Trả lời "NO" nếu:
- Chỉ là chào hỏi đơn giản ("xin chào", "cảm ơn", "hi", "hello")
- Trả lời mơ hồ, không cụ thể, không có số liệu
- Không có giá trị tham khảo
            
Chỉ trả lời "YES" hoặc "NO"."""
            )
            
            chain = memory_prompt | llm | StrOutputParser()
            result = chain.invoke({"query": query, "response": response}).strip().upper()
            return result == "YES"
            
        except Exception as e:
            print(f"⚠️ Memory save decision error: {e}")
            # Fallback to simple heuristic
            legal_keywords = ['phạt', 'luật', 'quy định', 'nghị định', 'tôi tên', 'tôi là', 'tôi đang']
            return any(word in query.lower() for word in legal_keywords) and len(response) > 50
    
    def format_context(self, memories: List[str]) -> str:
        """
        Format a list of memories into a context string.
        
        Args:
            memories: List of memory strings
            
        Returns:
            Formatted context string
        """
        if not memories:
            return ""
        return "\n".join([f"- {mem}" for mem in memories])
