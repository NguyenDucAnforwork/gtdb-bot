#!/usr/bin/env python3
"""
Test HippoRAG persona switching functionality
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chatbot_core import ChatbotCore
from memory.memory_manager import MemoryManager

def test_hipporag_persona():
    """Test HippoRAG persona auto-switching"""
    print("🧪 Testing HippoRAG persona functionality...")
    
    # Initialize components
    memory_manager = MemoryManager()
    chatbot = ChatbotCore(memory_manager=memory_manager)
    
    # Test 1: Default persona (should not use HippoRAG)
    print("\n1. Testing default persona:")
    response1 = chatbot.process_query(
        question="What is Vietnam's legal system?",
        user_id="test_user",
        persona_key="default"
    )
    print(f"Response: {response1[:100]}...")
    
    # Test 2: HippoRAG persona (should auto-enable force_hipporag)
    print("\n2. Testing HippoRAG persona:")
    response2 = chatbot.process_query(
        question="What is Vietnam's legal system?",
        user_id="test_user", 
        persona_key="hipporag"
    )
    print(f"Response: {response2[:100]}...")
    
    # Test 3: Explicit force_hipporag=True
    print("\n3. Testing explicit force_hipporag=True:")
    response3 = chatbot.process_query(
        question="What is Vietnam's legal system?",
        user_id="test_user",
        persona_key="default",
        force_hipporag=True
    )
    print(f"Response: {response3[:100]}...")
    
    print("\n✅ HippoRAG persona test completed!")

if __name__ == "__main__":
    test_hipporag_persona()