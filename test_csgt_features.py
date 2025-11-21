# test_csgt_features.py
"""
Test các tính năng CSGT theo EPIC EP-02
AC1: Tìm căn cứ & khung phạt ≤10s
AC2: Checklist & mẫu biên bản  
AC3: Offline bundle
"""

import time
from src.chatbot_core import ChatbotCore

def test_csgt_mode():
    print("🧪 TESTING CSGT FEATURES - EPIC EP-02")
    print("=" * 50)
    
    # Khởi tạo chatbot
    bot = ChatbotCore()
    
    # Test AC1: Tra cứu nhanh ≤10s
    print("\n📊 AC1 TEST: Fast Lookup (≤10s)")
    test_queries = [
        "vượt đèn đỏ xe máy",
        "không mũ bảo hiểm",
        "nồng độ cồn",
        "quá tốc độ 20km"
    ]
    
    for query in test_queries:
        start_time = time.time()
        response = bot.process_query(f"/lookup {query}", user_id="test_csgt", persona_key="csgt")
        elapsed = time.time() - start_time
        
        status = "✅ PASS" if elapsed <= 10 else f"❌ FAIL ({elapsed:.2f}s > 10s)"
        print(f"{status} - {query}: {elapsed:.2f}s")
        print(f"Response: {response[:100]}...\n")
    
    # Test AC2: Checklist
    print("\n📋 AC2 TEST: Checklist Generation")
    checklist_response = bot.process_query("/checklist nồng độ cồn xe ô tô", 
                                         user_id="test_csgt", 
                                         persona_key="csgt")
    print("Checklist Response:")
    print(checklist_response[:500], "...\n")
    
    # Test AC3: Offline Quick Codes
    print("\n💾 AC3 TEST: Offline Bundle")
    offline_codes = ["VDR", "QTS", "KMB", "NCN"]
    for code in offline_codes:
        response = bot.process_query(f"/quick {code}", user_id="test_csgt", persona_key="csgt")
        print(f"Code {code}: {response}\n")
    
    # Test Help Command
    print("\n❓ HELP TEST:")
    help_response = bot.process_query("/help", user_id="test_csgt", persona_key="csgt")
    print(help_response)
    
    print("\n🎉 CSGT TESTING COMPLETED!")

if __name__ == "__main__":
    test_csgt_mode()