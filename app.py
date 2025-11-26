# app.py
import os, requests, time
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from dotenv import load_dotenv
from src.chatbot_core import ChatbotCore
from src.memory import MemoryManager

load_dotenv()

VERIFY_TOKEN = os.getenv("FB_VERIFY_TOKEN", "verify-me")
PAGE_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN")

# --- USER PERSONA MANAGEMENT (In-Memory) ---
user_personas = {}  # Dictionary to store user personas {user_id: persona}

def get_user_persona(psid: str) -> str:
    """Lấy persona hiện tại của user (mặc định là 'default')"""
    return user_personas.get(psid, "default")

def set_user_persona(psid: str, persona: str):
    """Lưu persona của user vào memory"""
    user_personas[psid] = persona
    print(f"👤 User {psid} switched to persona: {persona}")

print("🚀 Initializing MemoryManager and ChatbotCore...")
memory_manager = MemoryManager()
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
async def webhook(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    print("📩 Received webhook data:", data)
    
    try:
        for entry in data.get("entry", []):
            for msg_event in entry.get("messaging", []):
                if "message" in msg_event and "text" in msg_event["message"]:
                    psid = msg_event["sender"]["id"]
                    query = msg_event["message"]["text"]
                    
                    # Add to background task to avoid timeout
                    background_tasks.add_task(process_message_task, psid, query)
    except Exception as e:
        print(f"Error parsing webhook: {e}")
        
    return {"status": "ok"}

def process_message_task(psid: str, query: str):
    """Process message with persona support and deduplication."""
    command = query.lower().strip()
    

    
    # Handle persona switching commands
    if command.startswith("/"):
        new_persona = None
        msg_reply = ""
        
        # Kiểm tra mode hiện tại
        if command in ["/mode", "/chedo", "/status", "/current", "/checkmode"]:
            current_persona = get_user_persona(psid)
            persona_names = {
                "default": "👤 NGƯỜI DÂN (Mặc định)",
                "csgt": "👮 CẢNH SÁT GIAO THÔNG", 
                "lawyer": "⚖️ LUẬT SƯ TƯ VẤN",
                "hipporag": "🦄 HIPPORAG (Knowledge Graph)"
            }
            current_name = persona_names.get(current_persona, "👤 NGƯỜI DÂN (Mặc định)")
            
            # Thêm hướng dẫn CSGT nếu đang ở chế độ CSGT
            csgt_help = ""
            if current_persona == "csgt":
                csgt_help = "\n\n👮 LỆNH CSGT:\n/lookup [từ khóa] - Tra cứu nhanh\n/checklist [lỗi] - Tạo checklist\n/quick [mã] - Tra offline\n/help - Xem tất cả lệnh"
            
            msg_reply = f"""🤖 TRẠNG THÁI HIỆN TẠI:
{current_name}

📋 CÁC LỆNH CHUYỂN ĐỔI:
• /mode default - Chế độ người dân (thân thiện)
• /mode csgt - Chế độ CSGT (ngắn gọn, tập trung mức phạt)
• /mode lawyer - Chế độ luật sư (phân tích sâu)
• /mode hipporag - Chế độ HippoRAG (knowledge graph)

💡 Gõ "/mode" để kiểm tra lại trạng thái.{csgt_help}"""
            send_message(psid, msg_reply)
            return
            
        # Chuyển đổi persona
        elif command in ["/changemode: csgt", "/mode csgt", "/chedo csgt", "/csgt"]:
            new_persona = "csgt"
            msg_reply = """👮 ĐÃ CHUYỂN SANG: CHẾ ĐỘ CẢNH SÁT GIAO THÔNG
📋 Phong cách: Ngắn gọn - Chính xác - Tập trung mức phạt

👮 LỆNH CSGT ĐẶC BIỆT:
• /lookup [từ khóa] - Tra cứu nhanh ≤10s
• /checklist [lỗi] - Tạo checklist lập biên bản  
• /quick [mã] - Tra offline (VDR,QTS,KMB,NCN...)
• /help - Xem tất cả lệnh

VÍ DỤ: /lookup vượt đèn đỏ xe máy"""
        elif command in ["/changemode: lawyer", "/mode lawyer", "/chedo luatsu", "/lawyer"]:
            new_persona = "lawyer"
            msg_reply = "⚖️ ĐÃ CHUYỂN SANG: CHẾ ĐỘ LUẬT SƯ TƯ VẤN\n📋 Phong cách: Phân tích sâu - Tư vấn pháp lý chuyên nghiệp"
        elif command in ["/mode hipporag", "/hipporag", "/hippo"]:
            new_persona = "hipporag"
            msg_reply = "🦄 ĐÃ CHUYỂN SANG: CHẾ ĐỘ HIPPORAG\n📋 Phong cách: Dùng Knowledge Graph thuần túy - Bypass memory và vector search\n💡 Giờ bạn có thể chat bình thường, tôi sẽ chỉ dùng HippoRAG"
        elif command in ["/changemode: default", "/mode default", "/chedo macdinh", "/default"]:
            new_persona = "default"
            msg_reply = "👤 ĐÃ TRỞ VỀ: CHẾ ĐỘ NGƯỜI DÂN (Mặc định)\n📋 Phong cách: Thân thiện - Dễ hiểu"
        
        if new_persona:
            set_user_persona(psid, new_persona)
            send_message(psid, msg_reply)
            return
        
        # ✅ XỬ LÝ LỆNH CSGT (lookup, checklist, quick, draft, help)
        current_persona = get_user_persona(psid)
        
        if query.startswith("/lookup ") or query.startswith("/checklist ") or \
           query.startswith("/quick ") or query.startswith("/draft ") or \
           command in ["/help", "/huongdan"]:
            # Gọi handler trong chatbot_core
            response = chatbot._handle_csgt_commands(query, user_id=psid)
            send_message(psid, f"👮 {response}")
            return
        
        # ✅ XỬ LÝ LỆNH INDEX (EP-03: Crawl & Index văn bản từ URL)
        if query.startswith("/index "):
            url = query[7:].strip()  # Remove "/index " prefix
            
            if not url:
                send_message(psid, "❌ Vui lòng cung cấp URL văn bản.\nVD: /index https://thuvienphapluat.vn/van-ban/.../Nghi-dinh-158-2024-ND-CP-...")
                return
            
            # Notify user that indexing started
            send_message(psid, f"🚀 Đang crawl & index văn bản từ:\n{url}\n\n⏳ Vui lòng đợi, quá trình này có thể mất vài phút...")
            
            try:
                # Call admin bot's index_from_url method
                from src.persona.admin_bot import AdminBot
                admin_bot = AdminBot(chatbot_core=chatbot)
                response = admin_bot.index_from_url(url)
                send_message(psid, response)
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                print(f"❌ Index error: {error_detail}")
                send_message(psid, f"❌ Lỗi khi index văn bản:\n{str(e)}")
            
            return
        
        # ✅ XỬ LÝ LỆNH ADMIN (EP-03: Quản trị văn bản pháp luật)
        if query.startswith("/admin"):
            # Kiểm tra quyền admin (TODO: implement proper auth)
            # For now, allow all users to test
            response = chatbot._handle_admin_commands(query, user_id=psid)
            send_message(psid, f"👨‍💼 {response}")
            return
    
    # Handle normal chat with deduplication
    try:
        # Get message metadata for deduplication (simplified for background task)
        current_time = time.time()
        
        # Skip if same user sent message within 2 seconds (rate limiting)
        if psid in last_processed_time:
            time_since_last = current_time - last_processed_time[psid]
            if time_since_last < 2:
                print(f"⏱️ Rate limit: user {psid} sent message too quickly ({time_since_last:.1f}s)")
                return
        
        # Update last processed time
        last_processed_time[psid] = current_time
        print(f"💬 User {psid}: {query}")
        
        # Clean up cache if too large
        cleanup_message_cache()
        
        # Get current persona from Redis
        current_persona = get_user_persona(psid)
        
        # Process with chatbot using persona
        response = chatbot.process_query(
            question=query,
            user_id=psid,
            persona_key=current_persona,
            force_hipporag=(current_persona == "hipporag")
        )
        
        # Save to memory if valuable
        if memory_manager.should_save_memory(query, response):
            memory_manager.add_conversation(query, response, psid)
            print(f"💾 Saved to Mem0 memory:")
            print(f"   User: {query[:80]}...")
            print(f"   Bot: {response[:80]}...")
        else:
            print("❌ Not saved to memory (greeting/low-value content)")
        
        # Add persona indicator to response
        persona_indicators = {
            "default": "👤",
            "csgt": "👮", 
            "lawyer": "⚖️",
            "hipporag": "🦄"
        }
        persona_icon = persona_indicators.get(current_persona, "👤")
        
        # Send response (truncate if too long)
        reply = response[:10000] if len(response) > 10000 else response
        final_reply = f"{persona_icon} {reply}"
        send_message(psid, final_reply)
        
    except Exception as e:
        import traceback
        print(f"❌ Error processing message: {e}")
        traceback.print_exc()
        send_message(psid, "Xin lỗi, hệ thống đang gặp sự cố xử lý.")

