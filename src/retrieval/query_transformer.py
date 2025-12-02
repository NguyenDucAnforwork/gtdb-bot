# src/retrieval/query_transformer.py
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

def create_query_transformer(retriever, llm):
    # Prompt mới: Tập trung vào TỪ KHÓA, loại bỏ các giả định sai về văn bản cũ
    legal_query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""Bạn là Chuyên gia Phân tích Ý định Pháp lý.
Nhiệm vụ: Phân tích câu hỏi và sinh ra 3 truy vấn tìm kiếm để bao quát mọi khía cạnh (Mức phạt + Quy định gốc + Thẩm quyền).

CÂU HỎI: {question}

PHÂN TÍCH VÀ TẠO TRUY VẤN:

1. **Truy vấn Cốt lõi (Core Intent):**
   - Nếu hỏi "bị phạt bao nhiêu" -> Tìm "Mức phạt tiền hành vi...".
   - Nếu hỏi "được phạt bao nhiêu/thẩm quyền" -> Tìm "Thẩm quyền xử phạt của...".

2. **Truy vấn Ngữ cảnh/Ngoại lệ (Context/Exception) - QUAN TRỌNG:**
   - Nếu câu hỏi nhắc đến **ĐỐI TƯỢNG ĐẶC BIỆT** (trẻ em, người già, xe ưu tiên) hoặc **CON SỐ** (8 tuổi, dưới 16 tuổi) -> BẮT BUỘC tạo truy vấn tìm quy định riêng.
   - Ví dụ: "Trẻ 8 tuổi" -> Truy vấn: "Quy định độ tuổi bắt buộc đội mũ bảo hiểm Luật Giao thông".

3. **Truy vấn Mở rộng (Expansion):**
   - Dùng từ khóa đồng nghĩa hoặc văn bản liên quan (Nghị định 168, Luật Trật tự ATGT 2024).

NGUYÊN TẮC TẠO TRUY VẤN (BẮT BUỘC):
1. **Trung lập về văn bản:** KHÔNG được tự ý thêm tên Nghị định cũ (như 100/2019, 123/2021) vào truy vấn. Chỉ dùng từ khóa chung như "nghị định", "luật", "quy định".
2. **Tập trung từ khóa:** Loại bỏ các từ nối rườm rà. Tập trung vào thực thể chính (mũ bảo hiểm, độ tuổi, mức phạt).
3. **Đa dạng từ ngữ:** Dùng các từ đồng nghĩa (ví dụ: "xử phạt", "phạt tiền", "chế tài").
4. **Phân biệt phương tiện:** Nếu câu hỏi là "xe máy" (mô tô, xe gắn máy), hãy dùng từ khóa chính xác "xe mô tô", "xe gắn máy". TRÁNH nhầm với "xe đạp máy".
5. **Tìm mức phạt:** Tìm trong Nghị định xử phạt mới nhất.
6. **Tìm quy định gốc:** Tìm trong Luật Trật tự an toàn giao thông đường bộ.

VÍ DỤ MINH HỌA:
Q: "Chở trẻ 8 tuổi không mũ phạt bao nhiêu?"
-> 1. Mức phạt tiền chở người không đội mũ bảo hiểm xe máy
-> 2. Quy định độ tuổi trẻ em bắt buộc đội mũ bảo hiểm Luật TTATGT 2024
-> 3. Ngoại lệ xử phạt không đội mũ bảo hiểm với trẻ em

Q: "Trưởng Công an xã được phạt bao nhiêu?"
-> 1. Thẩm quyền xử phạt của Trưởng Công an cấp xã Nghị định 168
-> 2. Mức phạt tiền tối đa Công an xã được phép phạt
-> 3. Các hành vi Trưởng Công an xã được quyền xử lý

DANH SÁCH 3 TRUY VẤN (Mỗi dòng 1 câu):"""
    )

    return MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
        prompt=legal_query_prompt
    )