#!/usr/bin/env python3
"""
Script to test the fixes for chatbot issues.
Run this to verify all improvements are working correctly.
"""

import asyncio
from app import process_user_query, memory
from src.chatbot_core import ChatbotCore

# Test user ID
TEST_USER = "test_user_fixes_001"

async def test_greeting_concise():
    """Test 1: Greeting should be concise"""
    print("\n" + "="*60)
    print("TEST 1: Greeting Conciseness")
    print("="*60)
    
    query = "Xin chào, tôi muốn hỏi về luật giao thông"
    response = await process_user_query(TEST_USER, query)
    
    print(f"Query: {query}")
    print(f"Response length: {len(response)} chars")
    print(f"Response:\n{response}\n")
    
    # Check if response is short (< 300 chars for greeting)
    if len(response) < 300:
        print("✅ PASS: Response is concise")
    else:
        print("❌ FAIL: Response is too long")
    
    return len(response) < 300


async def test_memory_follow_up():
    """Test 2: Memory should work for follow-up questions"""
    print("\n" + "="*60)
    print("TEST 2: Memory Follow-up (Test 4.1)")
    print("="*60)
    
    # Clear previous memory
    try:
        all_mems = memory.get_all(user_id=TEST_USER)
        for mem in all_mems:
            if 'id' in mem:
                memory.delete(mem['id'])
    except:
        pass
    
    queries = [
        "Mức phạt không đội mũ bảo hiểm là bao nhiêu?",
        "Còn nếu không có bằng lái thì sao?",
        "Vậy tổng cộng sẽ bị phạt bao nhiêu?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i}: {query}")
        response = await process_user_query(TEST_USER, query)
        await asyncio.sleep(2)  # Wait between queries
    
    # Check if final response mentions both violations
    last_response = response.lower()
    has_context = any(keyword in last_response for keyword in ['mũ bảo hiểm', 'bằng lái', 'tổng'])
    
    if has_context:
        print("\n✅ PASS: Bot remembered context from previous queries")
    else:
        print("\n❌ FAIL: Bot did not remember context")
    
    return has_context


async def test_speed_limit_fallback():
    """Test 3: Speed limit question should trigger fallback"""
    print("\n" + "="*60)
    print("TEST 3: Web Search Fallback for Speed Limit")
    print("="*60)
    
    query = "Tốc độ tối đa trong khu dân cư là bao nhiêu?"
    response = await process_user_query(TEST_USER, query)
    
    print(f"Query: {query}")
    print(f"Response:\n{response}\n")
    
    # Check if response contains specific speed number
    has_specific_answer = any(char.isdigit() for char in response)
    
    if has_specific_answer:
        print("✅ PASS: Response contains specific speed information")
    else:
        print("❌ FAIL: Response lacks specific speed information")
    
    return has_specific_answer


async def test_web_search_classification():
    """Test 4: "Quy định mới nhất" should be classified as WEB_SEARCH"""
    print("\n" + "="*60)
    print("TEST 4: Web Search Classification")
    print("="*60)
    
    query = "Tôi đang quan tâm về các quy định mới nhất"
    
    # Test classification
    chatbot = ChatbotCore()
    query_type = chatbot._classify_query(query, [])
    
    print(f"Query: {query}")
    print(f"Classified as: {query_type.value}")
    
    response = await process_user_query(TEST_USER, query)
    print(f"Response length: {len(response)} chars")
    print(f"Response:\n{response}\n")
    
    # Should be web_search and response should be concise (< 800 chars)
    is_correct = query_type.value == "web_search" and len(response) < 800
    
    if is_correct:
        print("✅ PASS: Correctly classified as WEB_SEARCH with concise response")
    else:
        print(f"❌ FAIL: Classification={query_type.value}, length={len(response)}")
    
    return is_correct


async def test_user_info_memory():
    """Test 5: User info should be saved to memory"""
    print("\n" + "="*60)
    print("TEST 5: User Info Memory")
    print("="*60)
    
    query = "Tôi tên An, tôi đang học luật tại Đại học Quốc gia"
    response = await process_user_query(TEST_USER, query)
    
    print(f"Query: {query}")
    print(f"Response: {response}\n")
    
    # Wait for memory to save
    await asyncio.sleep(2)
    
    # Check if info was saved
    all_memories = memory.get_all(user_id=TEST_USER)
    print(f"Total memories saved: {len(all_memories)}")
    
    # Check if user info is in memory
    has_user_info = any('An' in str(mem) or 'luật' in str(mem) for mem in all_memories)
    
    if has_user_info:
        print("✅ PASS: User info saved to memory")
        for mem in all_memories[:3]:
            print(f"  Memory: {str(mem)[:100]}...")
    else:
        print("❌ FAIL: User info not found in memory")
    
    return has_user_info


async def run_all_tests():
    """Run all tests"""
    print("\n" + "🧪 "*20)
    print("CHATBOT FIX VERIFICATION TESTS")
    print("🧪 "*20)
    
    results = []
    
    # Run tests
    # results.append(("Greeting Conciseness", await test_greeting_concise()))
    results.append(("Memory Follow-up", await test_memory_follow_up()))
    # results.append(("Speed Limit Fallback", await test_speed_limit_fallback()))
    # results.append(("Web Search Classification", await test_web_search_classification()))
    # results.append(("User Info Memory", await test_user_info_memory()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed*100//total}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review logs above.")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
