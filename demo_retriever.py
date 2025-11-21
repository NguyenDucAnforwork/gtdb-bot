"""
Demo file để test 3 chế độ retriever:
1. vector_only: Chỉ dùng Qdrant vector store
2. hipporag_only: Chỉ dùng HippoRAG knowledge graph
3. combined: Kết hợp cả hai

Usage:
    python demo_retriever.py --mode vector_only
    python demo_retriever.py --mode hipporag_only
    python demo_retriever.py --mode combined
    python demo_retriever.py --mode all  # Test cả 3 modes
"""

import argparse
import sys
from src.chatbot_core import ChatbotCore
from langchain_huggingface import HuggingFaceEmbeddings

# Test queries
TEST_QUERIES = [
    "Tự ý thay đổi kết cấu xe máy (độ xe) bị phạt bao nhiêu tiền?",
    "Chở con 8 tuổi không đội mũ bảo hiểm khi đi xe máy bị phạt bao nhiêu?",
    "Trưởng Công an xã được phạt tiền tối đa bao nhiêu đối với vi phạm giao thông?"
]


def print_separator(char="=", length=80):
    """Print separator line"""
    print(char * length)


def print_header(text):
    """Print formatted header"""
    print_separator()
    print(f"  {text}")
    print_separator()


def test_retriever_mode(chatbot: ChatbotCore, mode: str, queries: list):
    """
    Test một retriever mode với list queries
    
    Args:
        chatbot: ChatbotCore instance
        mode: "vector_only", "hipporag_only", or "combined"
        queries: List of test queries
    """
    print_header(f"🧪 Testing {mode.upper()} Mode")
    
    # Switch to requested mode
    chatbot.switch_retriever_mode(mode)
    
    # Get retriever info
    info = chatbot.get_retriever_info()
    print(f"\n📊 Retriever Configuration:")
    print(f"   • Current Mode: {info['current_mode']}")
    print(f"   • Description: {info['mode_descriptions'][info['current_mode']]}")
    
    # Test each query
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}/{len(queries)}: {query}")
        print(f"{'-'*80}")
        
        try:
            # Process query
            response = chatbot.process_query(query, chat_history=[])
            
            # Display response
            print(f"\n💬 Response:")
            print(response)
            
            # Extract and highlight citations
            print(f"\n📚 Citations Found:")
            if "[📚 Nguồn:" in response:
                import re
                citations = re.findall(r'\[📚 Nguồn: ([^\]]+)\]', response)
                for j, citation in enumerate(citations, 1):
                    print(f"   {j}. {citation}")
            else:
                print("   ⚠️ No structured citations found")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
    
    print_separator()


def compare_all_modes(queries: list):
    """
    So sánh tất cả 3 modes cho cùng một query
    
    Args:
        queries: List of test queries
    """
    print_header("🔍 Comparing All Retriever Modes")
    
    # Initialize chatbot
    print("\n🚀 Initializing ChatbotCore...")
    chatbot = ChatbotCore()
    
    modes = ["vector_only", "hipporag_only", "combined"]
    
    for query in queries:
        print_separator("=")
        print(f"\n📝 Query: {query}")
        print_separator("=")
        
        results = {}
        
        for mode in modes:
            print(f"\n🔄 Testing mode: {mode}")
            print("-" * 40)
            
            try:
                # Switch mode
                chatbot.switch_retriever_mode(mode)
                
                # Get response
                response = chatbot.process_query(query, chat_history=[])
                
                # Extract citations
                import re
                citations = re.findall(r'\[📚 Nguồn: ([^\]]+)\]', response)
                
                results[mode] = {
                    "response": response,
                    "citations": citations,
                    "response_length": len(response)
                }
                
                print(f"✅ Response length: {len(response)} chars")
                print(f"✅ Citations found: {len(citations)}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                results[mode] = {"error": str(e)}
        
        # Comparison table
        print(f"\n📊 Comparison Results:")
        print("-" * 80)
        print(f"{'Mode':<20} {'Response Length':<20} {'Citations':<20}")
        print("-" * 80)
        
        for mode in modes:
            if "error" not in results[mode]:
                resp_len = results[mode]['response_length']
                citations_count = len(results[mode]['citations'])
                print(f"{mode:<20} {resp_len:<20} {citations_count:<20}")
            else:
                print(f"{mode:<20} {'ERROR':<20} {'-':<20}")
        
        print("-" * 80)
        
        # Show detailed citations
        print(f"\n📚 Detailed Citations:")
        for mode in modes:
            if "error" not in results[mode] and results[mode]['citations']:
                print(f"\n{mode}:")
                for j, citation in enumerate(results[mode]['citations'], 1):
                    print(f"   {j}. {citation}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Demo HippoRAG + Qdrant Retriever")
    parser.add_argument(
        "--mode",
        choices=["vector_only", "hipporag_only", "combined", "all"],
        default="all",
        help="Retriever mode to test"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Custom query to test (optional)"
    )
    
    args = parser.parse_args()
    
    # Use custom query if provided
    queries = [args.query] if args.query else TEST_QUERIES
    
    if args.mode == "all":
        # Compare all modes
        compare_all_modes(queries)
    else:
        # Test single mode
        print("\n🚀 Initializing ChatbotCore...")
        chatbot = ChatbotCore()
        test_retriever_mode(chatbot, args.mode, queries)
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main()
