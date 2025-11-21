"""
Debug utilities for RAG Chatbot system
Separate debugging tools without modifying original source code
"""

import time
import json
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document


class RetrievalDebugger:
    """Debug utility to trace retrieval pipeline without modifying source code"""
    
    def __init__(self, chatbot_core):
        self.chatbot = chatbot_core
        self.debug_enabled = False
        self.last_debug_info = {}
    
    def enable_debug(self):
        """Enable detailed debugging"""
        self.debug_enabled = True
        print("🔍 Retrieval debugging enabled")
    
    def disable_debug(self):
        """Disable debugging"""
        self.debug_enabled = False
        print("❌ Retrieval debugging disabled")
    
    def debug_retrieval_pipeline(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Debug the entire retrieval pipeline step by step
        Returns detailed information about each step
        """
        debug_info = {
            "question": question,
            "timestamp": time.time(),
            "steps": []
        }
        
        print(f"🔍 DEBUGGING RETRIEVAL PIPELINE")
        print(f"Question: {question}")
        print("=" * 60)
        
        # Step 1: Test base retriever
        print("\n📊 STEP 1: Base Retriever (Vector Search)")
        try:
            base_docs = self.chatbot.base_retriever.invoke(question)
            debug_info["steps"].append({
                "step": "base_retriever",
                "documents_count": len(base_docs),
                "documents": [self._doc_to_dict(doc) for doc in base_docs[:3]]  # Top 3 only
            })
            
            print(f"✅ Found {len(base_docs)} documents from vector search")
            for i, doc in enumerate(base_docs[:3]):
                print(f"  Doc {i+1}: {doc.page_content[:100]}...")
                if hasattr(doc, 'metadata'):
                    print(f"    Metadata: {doc.metadata}")
                print()
        except Exception as e:
            print(f"❌ Base retriever failed: {e}")
            debug_info["steps"].append({"step": "base_retriever", "error": str(e)})
        
        # # Step 2: Test query transformer
        # print("\n🔄 STEP 2: Query Transformer")
        # try:
        #     transformed_docs = self.chatbot.query_transform_retriever.invoke(question)
        #     debug_info["steps"].append({
        #         "step": "query_transformer", 
        #         "documents_count": len(transformed_docs),
        #         "documents": [self._doc_to_dict(doc) for doc in transformed_docs[:3]]
        #     })
            
        #     print(f"✅ Query transformer returned {len(transformed_docs)} documents")
        #     for i, doc in enumerate(transformed_docs[:3]):
        #         print(f"  Doc {i+1}: {doc.page_content[:100]}...")
        #         print()
        # except Exception as e:
        #     print(f"❌ Query transformer failed: {e}")
        #     debug_info["steps"].append({"step": "query_transformer", "error": str(e)})
        
        # # Step 3: Test reranker
        # print("\n🎯 STEP 3: Reranker (Final Results)")
        # try:
        #     final_docs = self.chatbot.reranking_retriever.invoke(question)
        #     debug_info["steps"].append({
        #         "step": "reranker",
        #         "documents_count": len(final_docs), 
        #         "documents": [self._doc_to_dict(doc) for doc in final_docs]
        #     })
            
        #     print(f"✅ Reranker returned {len(final_docs)} final documents")
        #     for i, doc in enumerate(final_docs):
        #         print(f"  📄 FINAL Doc {i+1}:")
        #         print(f"    Content: {doc.page_content[:200]}...")
        #         if hasattr(doc, 'metadata'):
        #             print(f"    Metadata: {doc.metadata}")
        #         print()
        # except Exception as e:
        #     print(f"❌ Reranker failed: {e}")
        #     debug_info["steps"].append({"step": "reranker", "error": str(e)})
        
        self.last_debug_info = debug_info
        return debug_info, base_docs
    
    def debug_full_query(self, question: str, chat_history: List = None) -> Dict[str, Any]:
        """
        Debug the complete query processing including context formatting
        """
        chat_history = chat_history or []
        
        print(f"🔍 DEBUGGING FULL QUERY PROCESSING")
        print(f"Question: {question}")
        print("=" * 60)
        
        debug_info = {
            "question": question,
            "chat_history_length": len(chat_history),
            "steps": []
        }
        
        # Step 1: Check semantic cache
        print("\n💾 STEP 1: Semantic Cache Check")
        cached_response = self.chatbot.semantic_cache.get(question)
        if cached_response:
            print(f"✅ Cache HIT: {cached_response[:100]}...")
            debug_info["cache_hit"] = True
            return debug_info
        else:
            print("❌ Cache MISS")
            debug_info["cache_hit"] = False
        
        # Step 2: Debug retrieval
        print("\n📚 STEP 2: Document Retrieval")
        retrieval_debug = self.debug_retrieval_pipeline(question)
        debug_info["retrieval"] = retrieval_debug
        
        # Step 3: Test context formatting
        print("\n📝 STEP 3: Context Formatting")
        try:
            # Get documents from reranker
            docs = self.chatbot.reranking_retriever.invoke(question)
            
            # Format context like in chatbot_core
            formatted_context = "\n\n".join(
                getattr(doc, "page_content", str(doc)) for doc in docs
            )
            
            print(f"✅ Formatted context ({len(formatted_context)} characters):")
            print("📄 CONTEXT:")
            print("-" * 40)
            print(formatted_context[:500])
            if len(formatted_context) > 500:
                print("... (truncated)")
            print("-" * 40)
            
            debug_info["context"] = {
                "length": len(formatted_context),
                "preview": formatted_context[:500],
                "full_context": formatted_context
            }
        except Exception as e:
            print(f"❌ Context formatting failed: {e}")
            debug_info["context"] = {"error": str(e)}
        
        # Step 4: Test with legal persona prompt
        print("\n⚖️ STEP 4: Legal Persona Prompt")
        try:
            legal_prompt = self.chatbot.persona_manager.get_system_prompt("legal")
            print(f"Legal persona prompt: {legal_prompt[:200]}...")
            debug_info["legal_prompt"] = legal_prompt[:200]
        except Exception as e:
            print(f"❌ Legal persona failed: {e}")
            debug_info["legal_prompt_error"] = str(e)
        
        self.last_debug_info = debug_info
        return debug_info
    
    def test_specific_keywords(self, question: str, expected_keywords: List[str]) -> Dict[str, Any]:
        """
        Test if retrieval finds documents containing specific keywords
        """
        print(f"🔍 TESTING KEYWORD RETRIEVAL")
        print(f"Question: {question}")
        print(f"Expected keywords: {expected_keywords}")
        print("=" * 60)
        
        # Get documents
        docs = self.chatbot.reranking_retriever.invoke(question)
        
        results = {
            "question": question,
            "expected_keywords": expected_keywords,
            "documents_found": len(docs),
            "keyword_matches": {}
        }
        
        # Check each keyword
        for keyword in expected_keywords:
            matches = []
            for i, doc in enumerate(docs):
                content = getattr(doc, "page_content", str(doc)).lower()
                if keyword.lower() in content:
                    matches.append({
                        "doc_index": i,
                        "preview": content[content.find(keyword.lower())-50:content.find(keyword.lower())+100]
                    })
            
            results["keyword_matches"][keyword] = matches
            print(f"🔍 Keyword '{keyword}': {len(matches)} matches")
            
            for match in matches[:2]:  # Show top 2 matches
                print(f"  📄 Doc {match['doc_index']}: ...{match['preview']}...")
        
        return results
    
    def generate_improved_response(self, question: str, context: str) -> str:
        """
        Generate response with improved legal prompt
        """
        print("\n⚖️ GENERATING IMPROVED LEGAL RESPONSE")
        
        # Enhanced legal prompt for traffic violations
        enhanced_prompt = """Bạn là chuyên gia tư vấn pháp luật giao thông Việt Nam. 
Khi trả lời về vi phạm giao thông, BẮT BUỘC phải:

1. Nêu chính xác mức phạt tiền (VD: 400.000 - 600.000 đồng)
2. Trích dẫn đầy đủ điều khoản pháp lý (VD: Điểm o khoản 4 Điều 2 Nghị định 123/2021)
3. Giải thích rõ phạm vi áp dụng và ngoại lệ (VD: trẻ dưới 6 tuổi)
4. Đưa ví dụ minh họa nếu cần

Dựa trên ngữ cảnh sau đây để trả lời:

{context}

Câu hỏi: {question}

Trả lời chi tiết với trích dẫn pháp lý:"""
        
        formatted_prompt = enhanced_prompt.format(context=context, question=question)
        
        try:
            response = self.chatbot.llm.invoke(formatted_prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            print(f"✅ Enhanced response generated ({len(content)} characters)")
            return content
        except Exception as e:
            print(f"❌ Response generation failed: {e}")
            return f"Error generating response: {e}"
    
    def _doc_to_dict(self, doc: Document) -> Dict[str, Any]:
        """Convert document to dictionary for JSON serialization"""
        return {
            "content": getattr(doc, "page_content", str(doc))[:200],
            "metadata": getattr(doc, "metadata", {})
        }
    
    def save_debug_results(self, filename: str = "debug_results.json"):
        """Save last debug results to file"""
        if self.last_debug_info:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.last_debug_info, f, ensure_ascii=False, indent=2)
            print(f"💾 Debug results saved to {filename}")
        else:
            print("❌ No debug results to save")


class ChatbotTester:
    """Test specific scenarios and compare results"""
    
    def __init__(self, chatbot_core):
        self.chatbot = chatbot_core
        self.debugger = RetrievalDebugger(chatbot_core)
    
    def test_traffic_violation_case(self):
        """Test the specific traffic violation case"""
        question = "Chở con 8 tuổi không đội mũ bảo hiểm khi đi xe máy bị phạt bao nhiêu?"
        
        expected_keywords = [
            "400.000", "600.000", "đồng", 
            "nghị định 123/2021", "điều 2", "khoản 4",
            "mũ bảo hiểm", "trẻ em", "6 tuổi"
        ]
        
        print("🧪 TESTING TRAFFIC VIOLATION CASE")
        print("=" * 60)
        
        # Test current response
        print("\n1. Current Response:")
        current_response = self.chatbot.process_query(question, [])
        print(current_response)
        
        # Debug retrieval
        print("\n2. Debug Retrieval:")
        debug_info = self.debugger.debug_full_query(question)
        
        # Test keyword retrieval
        print("\n3. Keyword Testing:")
        keyword_results = self.debugger.test_specific_keywords(question, expected_keywords)
        
        # Generate improved response
        if "context" in debug_info and "full_context" in debug_info["context"]:
            print("\n4. Improved Response:")
            improved_response = self.debugger.generate_improved_response(
                question, debug_info["context"]["full_context"]
            )
            print(improved_response)
        
        return {
            "current_response": current_response,
            "debug_info": debug_info,
            "keyword_results": keyword_results,
            "improved_response": improved_response if 'improved_response' in locals() else None
        }