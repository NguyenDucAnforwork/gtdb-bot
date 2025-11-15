# app.py
import os, requests, time
from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv
from mem0 import MemoryClient, Memory
from src.chatbot_core import ChatbotCore

load_dotenv()

VERIFY_TOKEN = os.getenv("FB_VERIFY_TOKEN", "verify-me")
PAGE_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ FIX 1: Configure Memory with Qdrant Cloud (same as debug_memory.py)
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

print("🚀 Initializing Memory with Qdrant Cloud config...")
memory = Memory.from_config(config)
chatbot = ChatbotCore()
app = FastAPI()

@app.get("/webhook")
def verify(request: Request):
    args = request.query_params
    mode = args.get("hub.mode")
    token = args.get("hub.verify_token")
    challenge = args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return int(challenge)
    raise HTTPException(status_code=403, detail="Verification failed")


def send_message(psid, text):
    url = "https://graph.facebook.com/v18.0/me/messages"
    params = {"access_token": PAGE_TOKEN}
    payload = {"recipient": {"id": psid}, "message": {"text": text}}
    r = requests.post(url, params=params, json=payload, timeout=10)
    print(r.text)
    r.raise_for_status()

# Message deduplication cache
processed_messages = set()
last_processed_time = {}  # Track last processing time per user

def cleanup_message_cache():
    """Clean up old message IDs to prevent memory buildup."""
    global processed_messages, last_processed_time
    if len(processed_messages) > 1000:  # Keep only recent 1000
        processed_messages = set(list(processed_messages)[-500:])  # Keep last 500
    # Clean up old timestamps (older than 1 hour)
    current_time = time.time()
    last_processed_time = {k: v for k, v in last_processed_time.items() if current_time - v < 3600}

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    print("📩 Received webhook data:", data)
    
    for entry in data.get("entry", []):
        for msg_event in entry.get("messaging", []):
            psid = msg_event["sender"]["id"]
            
            if "message" in msg_event and "text" in msg_event["message"]:
                # Get message ID and timestamp for deduplication
                message_id = msg_event["message"].get("mid")
                message_timestamp = msg_event.get("timestamp", 0) / 1000  # Convert to seconds
                query = msg_event["message"]["text"]
                current_time = time.time()
                
                # Skip if message is too old (more than 10 seconds)
                message_age = current_time - message_timestamp
                if message_age > 10:
                    print(f"⏰ Skipping old message ({message_age:.1f}s old): {message_id}")
                    continue
                
                # Skip if already processed
                if message_id in processed_messages:
                    print(f"⚠️ Duplicate message ignored: {message_id}")
                    continue
                
                # Skip if same user sent message within 2 seconds (rate limiting)
                if psid in last_processed_time:
                    time_since_last = current_time - last_processed_time[psid]
                    if time_since_last < 2:
                        print(f"⏱️ Rate limit: user {psid} sent message too quickly ({time_since_last:.1f}s)")
                        continue
                
                # Add to processed set
                processed_messages.add(message_id)
                last_processed_time[psid] = current_time
                print(f"💬 User {psid}: {query} (mid: {message_id}, age: {message_age:.1f}s)")
                
                # Clean up cache if too large
                cleanup_message_cache()
                
                # Smart memory strategy based on query type
                response = await process_user_query(psid, query)
                
                # Send response (truncate if too long)
                reply = response[:1000] if len(response) > 1000 else response
                send_message(psid, reply)
    
    return {"status": "ok"}

async def process_user_query(user_id: str, query: str) -> str:
    """Process user query with LLM-based intelligence."""
    try:
        # ✅ Step 1: Get ALL memories for this user
        res = memory.get_all(user_id=user_id)
        all_memories = res.get('results', [])  # List of dicts
        
        # Extract just the 'memory' text field
        memories = [mem['memory'] for mem in all_memories]
        
        # Debug logging
        print(f"🧠 MEMORY DEBUG for user {user_id}:")
        print(f"   Found memories: {len(memories)}")
        if memories:
            for i, mem in enumerate(memories[:3], 1):  # Show first 3
                print(f"   {i}. {mem[:80]}...")        
        
        # Step 2: Build contextual query if we have history
        contextual_query = query
        if memories:
            # Use last 5 memories as context
            contextual_query = f"""NGỮ CẢNH CUỘC TRÒ CHUYỆN:
{chr(10).join([f'- {mem}' for mem in memories[:-5]])}

CÂU HỎI: {query}

Hãy trả lời dựa trên ngữ cảnh trên."""
        
        # Step 3: Process with chatbot
        response = chatbot.process_query(contextual_query, memories, user_id=user_id)
        
        # Step 4: Decide whether to save using LLM
        if should_save_to_memory(query, response):
            messages_to_save = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response}
            ]
            memory.add(messages_to_save, user_id=user_id, infer=False)
            print(f"💾 Saved to Mem0 memory:")
            print(f"   User: {query[:80]}...")
            print(f"   Bot: {response[:80]}...")
        else:
            print("❌ Not saved to memory (greeting/low-value content)")
        
        return response
        
    except Exception as e:
        import traceback
        print(f"❌ Error processing query: {e}")
        traceback.print_exc()
        return "Xin lỗi, có lỗi xảy ra. Bạn có thể thử lại không?"


def should_save_to_memory(query: str, response: str) -> bool:
    """Use LLM to decide whether to save conversation to memory."""
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
- Chỉ là chào hỏi đơn giản ("xin chào", "cảm ơn")
- Trả lời mơ hồ, không cụ thể, không có số liệu
- Không có giá trị tham khảo
            
Chỉ trả lời "YES" hoặc "NO"."""
        )
        
        chain = memory_prompt | llm | StrOutputParser()
        result = chain.invoke({"query": query, "response": response}).strip().upper()
        return result == "YES"
    except Exception:
        # Fallback to simple heuristic
        legal_keywords = ['phạt', 'luật', 'quy định', 'nghị định', 'tôi tên', 'tôi là', 'tôi đang']
        return any(word in query.lower() for word in legal_keywords) and len(response) > 50

