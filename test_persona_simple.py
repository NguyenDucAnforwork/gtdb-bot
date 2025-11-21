# test_persona_simple.py
"""Test persona system functionality"""

import sys
import os
sys.path.append('/workspace/gtdb-bot')

from src.chatbot_core import ChatbotCore
from src.persona.prompts import SYSTEM_PROMPTS

def test_persona_system():
    """Test basic persona functionality"""
    print("🧪 Testing Persona System...")
    
    # Test 1: Check if SYSTEM_PROMPTS are loaded
    print("\n1️⃣ Testing SYSTEM_PROMPTS:")
    for persona, prompt in SYSTEM_PROMPTS.items():
        print(f"   ✅ {persona}: {prompt[:50]}...")
    
    # Test 2: Initialize ChatbotCore
    print("\n2️⃣ Testing ChatbotCore initialization:")
    try:
        chatbot = ChatbotCore()
        print("   ✅ ChatbotCore initialized successfully")
    except Exception as e:
        print(f"   ❌ ChatbotCore initialization failed: {e}")
        return
    
    # Test 3: Test persona parameter
    print("\n3️⃣ Testing persona parameter:")
    test_query = "Xin chào"
    
    for persona in ["default", "csgt", "lawyer"]:
        try:
            response = chatbot.process_query(
                question=test_query, 
                user_id="test_user", 
                persona_key=persona
            )
            print(f"   ✅ {persona}: {response[:100]}...")
        except Exception as e:
            print(f"   ❌ {persona} failed: {e}")
    
    print("\n✅ Persona system test completed!")

if __name__ == "__main__":
    test_persona_system()