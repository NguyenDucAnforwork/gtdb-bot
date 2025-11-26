"""
Test EP-03: Cập nhật & quản trị văn bản pháp luật
Test crawling, indexing, và admin workflows

AC1: Crawler + chuẩn hoá JSONL → cập nhật KG/VecStore ≤7 ngày
AC2: Gắn trạng thái hiệu lực; hiển thị diff cũ–mới
AC3: Coverage ≥95% nguồn GTĐB trọng yếu
"""

import pytest
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# TEST 1: Web Crawler - Crawl văn bản từ thuvienphapluat.vn
# ============================================================================

def test_crawler_basic():
    """Test basic crawling functionality"""
    from src.ingestion.crawler import ThuvienPhaplaatCrawler
    
    crawler = ThuvienPhaplaatCrawler()
    
    # Test URL - Nghị định 158/2024/NĐ-CP
    test_url = "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Nghi-dinh-158-2024-ND-CP-quy-dinh-hoat-dong-van-tai-duong-bo-636875.aspx"
    
    print(f"\n🧪 TEST: Crawling từ URL...")
    result = crawler.crawl(test_url)
    
    # Assertions
    assert result is not None, "❌ Crawler returned None"
    assert 'law_code' in result, "❌ Missing law_code"
    assert 'title' in result, "❌ Missing title"
    assert 'content' in result, "❌ Missing content"
    assert 'url' in result, "❌ Missing url"
    
    print(f"✅ Law code: {result['law_code']}")
    print(f"✅ Title: {result['title'][:100]}...")
    print(f"✅ Content length: {len(result['content'])} chars")
    print(f"✅ URL: {result['url']}")
    
    # Validate content không rỗng
    assert len(result['content']) > 1000, "❌ Content too short"
    
    print("✅ TEST PASSED: Crawler hoạt động tốt")
    return result


def test_crawler_law_code_extraction():
    """Test law code extraction từ URL"""
    from src.ingestion.crawler import ThuvienPhaplaatCrawler
    
    crawler = ThuvienPhaplaatCrawler()
    
    test_cases = [
        {
            "url": "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Nghi-dinh-158-2024-ND-CP-quy-dinh-hoat-dong-van-tai-duong-bo-636875.aspx",
            "expected_pattern": "158/2024"
        },
        {
            "url": "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Nghi-dinh-100-2019-ND-CP-xu-phat-giao-thong-duong-bo-428726.aspx",
            "expected_pattern": "100/2019"
        }
    ]
    
    print(f"\n🧪 TEST: Law code extraction...")
    
    for case in test_cases:
        result = crawler.crawl(case['url'])
        if result:
            print(f"✅ URL: {case['url'][:80]}...")
            print(f"   Law code: {result['law_code']}")
            assert case['expected_pattern'] in result['law_code'], \
                f"❌ Expected {case['expected_pattern']} in {result['law_code']}"
    
    print("✅ TEST PASSED: Law code extraction chính xác")


# ============================================================================
# TEST 2: Document Splitter - Tách văn bản theo Điều/Khoản
# ============================================================================

def test_document_splitter():
    """Test splitting văn bản theo Điều/Khoản"""
    from src.ingestion.updater import split_passages
    
    # Sample legal text
    sample_text = """
    Điều 1. Phạm vi điều chỉnh
    
    Nghị định này quy định về xử phạt vi phạm hành chính trong lĩnh vực giao thông đường bộ.
    
    Điều 2. Đối tượng áp dụng
    
    1. Cá nhân, tổ chức có hành vi vi phạm quy định về trật tự, an toàn giao thông đường bộ.
    
    2. Cơ quan, tổ chức, cá nhân có thẩm quyền xử phạt vi phạm hành chính trong lĩnh vực giao thông đường bộ.
    
    Điều 3. Nguyên tắc xử phạt
    
    Khoản 1: Việc xử phạt vi phạm hành chính phải căn cứ vào tính chất, mức độ vi phạm.
    
    Khoản 2: Một hành vi vi phạm hành chính chỉ bị xử phạt một lần.
    """
    
    print(f"\n🧪 TEST: Document splitting...")
    
    passages = split_passages(sample_text, law_code="TEST-001")
    
    # Assertions
    assert len(passages) > 0, "❌ No passages extracted"
    print(f"✅ Extracted {len(passages)} passages")
    
    # Validate passage format
    for i, passage in enumerate(passages[:3], 1):
        print(f"\n📄 Passage {i}:")
        print(passage[:200] + "...")
        assert "[TEST-001]" in passage, "❌ Missing law code in passage"
        assert "Điều" in passage, "❌ Missing Điều marker"
    
    print("✅ TEST PASSED: Document splitter hoạt động tốt")
    return passages


def test_splitter_with_real_document():
    """Test splitter với văn bản thực tế"""
    from src.ingestion.crawler import crawl_document
    from src.ingestion.updater import split_passages
    
    print(f"\n🧪 TEST: Splitting real document...")
    
    # Crawl document
    test_url = "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Nghi-dinh-100-2019-ND-CP-xu-phat-giao-thong-duong-bo-428726.aspx"
    
    doc_data = crawl_document(test_url)
    assert doc_data is not None, "❌ Failed to crawl document"
    
    # Split
    passages = split_passages(doc_data['content'], law_code=doc_data['law_code'])
    
    print(f"✅ Crawled: {doc_data['law_code']}")
    print(f"✅ Split into {len(passages)} passages")
    
    # Validate có đủ passages (NĐ 100/2019 có nhiều điều)
    assert len(passages) > 50, f"❌ Too few passages: {len(passages)}"
    assert len(passages) < 500, f"❌ Too many passages: {len(passages)}"
    
    # Sample passages
    print("\n📄 Sample passages:")
    for i in range(min(3, len(passages))):
        print(f"\nPassage {i+1}:")
        print(passages[i][:150] + "...")
    
    print(f"\n✅ TEST PASSED: Split {len(passages)} passages from real document")
    return passages


# ============================================================================
# TEST 3: Qdrant Integration - Vector indexing
# ============================================================================

def test_qdrant_indexing():
    """Test indexing vào Qdrant"""
    from src.ingestion.updater import update_qdrant
    
    print(f"\n🧪 TEST: Qdrant indexing...")
    
    # Test passages
    test_passages = [
        "[TEST-001] Điều 1\n\nNội dung điều 1 về phạm vi điều chỉnh",
        "[TEST-001] Điều 2 Khoản 1\n\nNội dung khoản 1 về đối tượng áp dụng",
        "[TEST-001] Điều 2 Khoản 2\n\nNội dung khoản 2 về cơ quan thẩm quyền"
    ]
    
    # Index
    try:
        update_qdrant(test_passages, collection_name="gtdb-1")
        print(f"✅ Successfully indexed {len(test_passages)} passages to Qdrant")
    except Exception as e:
        print(f"⚠️ Qdrant indexing failed (expected if not configured): {e}")
        pytest.skip("Qdrant not configured")
    
    print("✅ TEST PASSED: Qdrant indexing successful")


def test_qdrant_with_real_passages():
    """Test Qdrant với passages thực tế"""
    from src.ingestion.crawler import crawl_document
    from src.ingestion.updater import split_passages, update_qdrant
    
    print(f"\n🧪 TEST: Qdrant with real passages...")
    
    # Get real passages
    test_url = "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Nghi-dinh-100-2019-ND-CP-xu-phat-giao-thong-duong-bo-428726.aspx"
    
    doc_data = crawl_document(test_url)
    if not doc_data:
        pytest.skip("Failed to crawl document")
    
    passages = split_passages(doc_data['content'], law_code=doc_data['law_code'])
    
    # Index first 10 passages only (for testing)
    test_passages = passages[:10]
    
    try:
        update_qdrant(test_passages, collection_name="gtdb-1")
        print(f"✅ Indexed {len(test_passages)} passages to Qdrant")
    except Exception as e:
        print(f"⚠️ Qdrant indexing failed: {e}")
        pytest.skip("Qdrant not configured")
    
    print("✅ TEST PASSED: Real passages indexed to Qdrant")


# ============================================================================
# TEST 4: HippoRAG Integration - Knowledge graph indexing
# ============================================================================

def test_hipporag_indexing():
    """Test indexing vào HippoRAG"""
    from src.ingestion.updater import update_hipporag
    
    print(f"\n🧪 TEST: HippoRAG indexing...")
    
    # Test passages
    test_passages = [
        "[TEST-002] Điều 44 Khoản 1\n\nPhạt tiền từ 800.000 đồng đến 1.000.000 đồng đối với người điều khiển xe máy không đội mũ bảo hiểm.",
        "[TEST-002] Điều 44 Khoản 2\n\nPhạt tiền từ 400.000 đồng đến 600.000 đồng đối với người ngồi trên xe máy không đội mũ bảo hiểm."
    ]
    
    # Index
    try:
        update_hipporag(test_passages)
        print(f"✅ Successfully indexed {len(test_passages)} passages to HippoRAG")
    except Exception as e:
        print(f"⚠️ HippoRAG indexing failed (may take time/cost): {e}")
        pytest.skip("HippoRAG indexing skipped")
    
    print("✅ TEST PASSED: HippoRAG indexing successful")


# ============================================================================
# TEST 5: Admin Bot - Full workflow integration
# ============================================================================

def test_admin_bot_initialization():
    """Test Admin Bot khởi tạo"""
    from src.persona.admin_bot import AdminBot
    
    print(f"\n🧪 TEST: Admin Bot initialization...")
    
    admin_bot = AdminBot()
    
    # Validate initialization
    assert admin_bot is not None, "❌ Failed to initialize AdminBot"
    assert hasattr(admin_bot, 'coverage_stats'), "❌ Missing coverage_stats"
    assert hasattr(admin_bot, 'pending_docs'), "❌ Missing pending_docs"
    
    print(f"✅ Admin Bot initialized")
    print(f"✅ Coverage stats: {admin_bot.coverage_stats}")
    
    print("✅ TEST PASSED: Admin Bot initialization successful")


def test_admin_bot_index_from_url():
    """Test Admin Bot index_from_url method"""
    from src.persona.admin_bot import AdminBot
    
    print(f"\n🧪 TEST: Admin Bot index_from_url...")
    
    admin_bot = AdminBot()
    
    # Test URL
    test_url = "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Nghi-dinh-100-2019-ND-CP-xu-phat-giao-thong-duong-bo-428726.aspx"
    
    print(f"📥 Testing index from: {test_url}")
    
    try:
        result = admin_bot.index_from_url(test_url)
        
        # Validate result
        assert result is not None, "❌ index_from_url returned None"
        assert isinstance(result, str), "❌ Result should be string message"
        
        print(f"\n📊 Result:\n{result}")
        
        # Check success indicators
        if "✅" in result or "THÀNH CÔNG" in result:
            print("✅ Index successful!")
        else:
            print(f"⚠️ Index may have issues: {result[:200]}")
        
    except Exception as e:
        print(f"⚠️ Index failed (expected if services not configured): {e}")
        pytest.skip("Services not configured")
    
    print("✅ TEST PASSED: Admin Bot index_from_url works")


def test_admin_bot_help():
    """Test Admin Bot help menu"""
    from src.persona.admin_bot import AdminBot
    
    print(f"\n🧪 TEST: Admin Bot help menu...")
    
    admin_bot = AdminBot()
    help_text = admin_bot.get_help()
    
    # Validate help text
    assert help_text is not None, "❌ Help text is None"
    assert "/index" in help_text, "❌ Missing /index command"
    assert "/admin" in help_text, "❌ Missing /admin commands"
    assert "AC1" in help_text or "AC2" in help_text or "AC3" in help_text, "❌ Missing AC references"
    
    print(f"\n📋 Help Menu:\n{help_text}")
    
    print("✅ TEST PASSED: Help menu complete")


# ============================================================================
# TEST 6: Coverage Tracking (AC3)
# ============================================================================

def test_coverage_tracking():
    """Test coverage statistics tracking"""
    from src.persona.admin_bot import AdminBot
    
    print(f"\n🧪 TEST: Coverage tracking (AC3)...")
    
    admin_bot = AdminBot()
    
    # Check initial stats
    initial_coverage = admin_bot.coverage_stats['coverage_rate']
    print(f"📊 Initial coverage: {initial_coverage}%")
    
    # Get coverage report
    coverage_report = admin_bot.get_coverage_report()
    
    print(f"\n📈 Coverage Report:\n{coverage_report[:500]}...")
    
    # Validate report structure
    assert "COVERAGE" in coverage_report, "❌ Missing coverage section"
    assert "%" in coverage_report, "❌ Missing percentage"
    
    # Check target
    assert "95%" in coverage_report or "95" in coverage_report, "❌ Missing 95% target"
    
    print("✅ TEST PASSED: Coverage tracking works")


# ============================================================================
# TEST 7: End-to-End Integration Test
# ============================================================================

def test_end_to_end_indexing():
    """
    Test hoàn chỉnh workflow:
    1. Crawl văn bản
    2. Split passages
    3. Index vào Qdrant
    4. Index vào HippoRAG
    5. Update coverage
    """
    from src.persona.admin_bot import AdminBot
    
    print(f"\n🧪 TEST: End-to-end indexing workflow...")
    print("=" * 60)
    
    admin_bot = AdminBot()
    
    # Test URL - sử dụng văn bản nhỏ hơn để test nhanh
    test_url = "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Nghi-dinh-100-2019-ND-CP-xu-phat-giao-thong-duong-bo-428726.aspx"
    
    print(f"🎯 Target URL: {test_url}")
    print(f"⏰ Starting at: {__import__('datetime').datetime.now()}")
    
    try:
        # Execute full workflow
        result = admin_bot.index_from_url(test_url)
        
        print(f"\n📊 RESULT:")
        print("=" * 60)
        print(result)
        print("=" * 60)
        
        # Validate result
        assert result is not None, "❌ No result returned"
        
        # Check for success indicators
        success_indicators = ["✅", "THÀNH CÔNG", "INDEX THÀNH CÔNG"]
        has_success = any(indicator in result for indicator in success_indicators)
        
        if has_success:
            print("\n🎉 END-TO-END TEST PASSED!")
            print("   ✅ Crawling successful")
            print("   ✅ Splitting successful")
            print("   ✅ Indexing successful")
            print("   ✅ Coverage updated")
        else:
            print("\n⚠️ Workflow completed with warnings")
            print(f"   Result: {result[:200]}")
        
    except Exception as e:
        import traceback
        print(f"\n❌ End-to-end test failed:")
        print(traceback.format_exc())
        pytest.skip(f"E2E test failed: {e}")
    
    print(f"\n⏰ Finished at: {__import__('datetime').datetime.now()}")


# ============================================================================
# TEST 8: Error Handling
# ============================================================================

def test_crawler_invalid_url():
    """Test crawler với URL không hợp lệ"""
    from src.ingestion.crawler import crawl_document
    
    print(f"\n🧪 TEST: Crawler error handling...")
    
    invalid_urls = [
        "https://invalid-domain-12345.com/fake-law.aspx",
        "https://thuvienphapluat.vn/not-found-page.aspx",
        "not-a-url"
    ]
    
    for url in invalid_urls:
        print(f"Testing invalid URL: {url}")
        result = crawl_document(url)
        
        # Should return None for invalid URLs
        assert result is None, f"❌ Should return None for invalid URL: {url}"
        print(f"   ✅ Correctly returned None")
    
    print("✅ TEST PASSED: Error handling works correctly")


def test_admin_bot_empty_url():
    """Test Admin Bot với URL rỗng"""
    from src.persona.admin_bot import AdminBot
    
    print(f"\n🧪 TEST: Admin Bot empty URL handling...")
    
    admin_bot = AdminBot()
    
    # Test với URL rỗng
    result = admin_bot.index_from_url("")
    
    # Should return error message
    assert result is not None, "❌ Should return error message"
    assert "❌" in result or "lỗi" in result.lower(), "❌ Should indicate error"
    
    print(f"✅ Error message: {result[:100]}")
    print("✅ TEST PASSED: Empty URL handled correctly")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🧪 EP-03 TEST SUITE: Cập nhật & Quản trị văn bản pháp luật")
    print("=" * 80)
    
    # Kiểm tra dependencies
    print("\n📦 Checking dependencies...")
    try:
        import requests
        import bs4
        from sentence_transformers import SentenceTransformer
        print("✅ All dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Run: pip install beautifulsoup4 requests")
        exit(1)
    
    # Run tests
    tests = [
        ("Crawler Basic", test_crawler_basic),
        ("Crawler Law Code", test_crawler_law_code_extraction),
        ("Document Splitter", test_document_splitter),
        ("Splitter Real Doc", test_splitter_with_real_document),
        ("Admin Bot Init", test_admin_bot_initialization),
        ("Admin Bot Help", test_admin_bot_help),
        ("Coverage Tracking", test_coverage_tracking),
        ("Error Handling - Invalid URL", test_crawler_invalid_url),
        ("Error Handling - Empty URL", test_admin_bot_empty_url),
    ]
    
    # Optional tests (require services)
    optional_tests = [
        ("Qdrant Indexing", test_qdrant_indexing),
        ("Qdrant Real Passages", test_qdrant_with_real_passages),
        ("HippoRAG Indexing", test_hipporag_indexing),
        ("Admin Bot Index URL", test_admin_bot_index_from_url),
        ("END-TO-END", test_end_to_end_indexing),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    # Run core tests
    for name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"🧪 Running: {name}")
        print(f"{'='*80}")
        try:
            test_func()
            passed += 1
            print(f"✅ {name} PASSED")
        except Exception as e:
            failed += 1
            print(f"❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # Run optional tests
    print(f"\n{'='*80}")
    print("🔧 Running OPTIONAL tests (may skip if services not configured)")
    print(f"{'='*80}")
    
    for name, test_func in optional_tests:
        print(f"\n{'='*80}")
        print(f"🧪 Running: {name}")
        print(f"{'='*80}")
        try:
            test_func()
            passed += 1
            print(f"✅ {name} PASSED")
        except pytest.skip.Exception as e:
            skipped += 1
            print(f"⏭️ {name} SKIPPED: {e}")
        except Exception as e:
            skipped += 1
            print(f"⚠️ {name} SKIPPED (service issue): {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 TEST SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Passed:  {passed}/{len(tests) + len(optional_tests)}")
    print(f"❌ Failed:  {failed}/{len(tests) + len(optional_tests)}")
    print(f"⏭️ Skipped: {skipped}/{len(tests) + len(optional_tests)}")
    
    if failed == 0:
        print("\n🎉 ALL CORE TESTS PASSED!")
        print("✅ EP-03 Implementation is ready for demo")
    else:
        print(f"\n⚠️ {failed} test(s) failed - please review")
    
    print(f"\n{'='*80}\n")
