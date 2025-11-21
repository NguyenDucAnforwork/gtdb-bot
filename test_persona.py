#!/usr/bin/env python3
# test_persona.py - Test persona system

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.chatbot_core import ChatbotCore
from src.persona.prompts import SYSTEM_PROMPTS

def test_persona_system():
    """Test persona switching and responses."""
    print("🧪 Testing Persona System...")
    
    # Initialize chatbot
    chatbot = ChatbotCore()
    
    # Test question
    test_question = "Mức phạt không đội mũ bảo hiểm là bao nhiêu?"
    
    print(f"\n📝 Câu hỏi test: {test_question}")
    print("\n" + "="*60)
    
    # Test each persona
    personas = ["default", "csgt", "lawyer"]
    
    for persona in personas:
        print(f"\n🎭 Testing persona: {persona}")
        print(f"📋 System prompt: {SYSTEM_PROMPTS[persona][:100]}...")
        
        try:
            response = chatbot.process_query(
                question=test_question,
                user_id="test_user", 
                persona_key=persona
            )
            
            print(f"✅ Response ({len(response)} chars):")
            print(f"   {response[:200]}...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 40)
    
    print("\n✅ Persona test completed!")

def test_redis_fallback():
    """Test Redis fallback functionality."""
    print("\n🧪 Testing Redis fallback...")
    
    # Test persona functions without Redis
    from app import get_user_persona, set_user_persona
    
    test_user = "test_user_123"
    
    # Should return "default" when Redis is not available
    persona = get_user_persona(test_user)
    print(f"✅ Default persona returned: {persona}")
    
    # Should not crash when setting persona without Redis
    set_user_persona(test_user, "csgt")
    print("✅ Set persona function handled gracefully")

if __name__ == "__main__":
    test_persona_system()
    test_redis_fallback()