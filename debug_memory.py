from mem0 import Memory
import os
from dotenv import load_dotenv
from openai import OpenAI 

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "test-api-key")

client = Memory()
USER_ID = "user123_fixed_infer_false" 

client.delete_all(user_id=USER_ID)

# --- (1) LƯU DỮ LIỆU ---
# SỬA LỖI: Thêm infer=False để ép mem0 lưu trữ text thô
messages_1 = [
    {"role": "user", "content": "Mức phạt không đội mũ bảo hiểm là bao nhiêu?"},
    {"role": "assistant", "content": "Mức phạt... là từ 400.000 đồng đến 600.000 đồng."}
]
client.add(messages_1, user_id=USER_ID, infer=False)

messages_2 = [
    {"role": "user", "content": "Còn nếu không có bằng lái thì sao?"},
    {"role": "assistant", "content": "Nếu... không có bằng lái... mức phạt... từ 4.000.000 VNĐ đến 6.000.000 VNĐ."}
]
client.add(messages_2, user_id=USER_ID, infer=False)
print("--- Đã lưu 2 facts (với infer=False) ---")

res = client.get_all(user_id=USER_ID)
context = [mem['memory'] for mem in res['results']]
print(f"All memories for user {USER_ID}: {context}")
