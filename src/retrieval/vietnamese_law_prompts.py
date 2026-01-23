# src/retrieval/vietnamese_law_prompts.py
"""
Custom prompts cho Vietnamese Traffic Law - tối ưu cho HippoRAG
Được customize từ notebook vietnamese_traffic_law_qa.ipynb (Section 4)
"""

# ===========================
# NER PROMPT - Trích xuất thực thể từ văn bản luật Việt Nam
# ===========================

VIETNAMESE_LAW_NER_SYSTEM = """Nhiệm vụ của bạn là trích xuất các thực thể có tên từ văn bản luật giao thông Việt Nam.
Trả lời bằng một danh sách JSON các thực thể.

Các loại thực thể cần chú ý:
- Hành vi vi phạm (vượt đèn đỏ, không đội mũ bảo hiểm, v.v.)
- Phương tiện (xe máy, ô tô, xe tải, v.v.)
- Mức phạt (số tiền, thời gian tước bằng)
- Điều khoản pháp luật (Điều, Khoản, Nghị định)
- Cơ quan chức năng (Cảnh sát giao thông, Thanh tra)
- Giấy tờ (Bằng lái xe, Giấy phép, Đăng ký xe)
"""

VIETNAMESE_LAW_NER_EXAMPLE_INPUT = """Điều 5. Xử phạt người điều khiển xe mô tô, xe gắn máy
1. Phạt tiền từ 800.000 đồng đến 1.000.000 đồng đối với người điều khiển xe mô tô, xe gắn máy không có Giấy phép lái xe theo quy định.
2. Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với người điều khiển xe mô tô vượt đèn đỏ."""

VIETNAMESE_LAW_NER_EXAMPLE_OUTPUT = """{
    "named_entities": [
        "Điều 5",
        "xe mô tô",
        "xe gắn máy",
        "800.000 đồng",
        "1.000.000 đồng",
        "Giấy phép lái xe",
        "4.000.000 đồng",
        "6.000.000 đồng",
        "vượt đèn đỏ"
    ]
}"""

VIETNAMESE_LAW_TRIPLE_EXAMPLE_OUTPUT = """{
    "triples": [
        ["Điều 5", "quy định về", "xe mô tô"],
        ["Điều 5", "quy định về", "xe gắn máy"],
        ["không có Giấy phép lái xe", "bị phạt", "800.000 đồng"],
        ["không có Giấy phép lái xe", "bị phạt", "1.000.000 đồng"],
        ["vượt đèn đỏ", "bị phạt", "4.000.000 đồng"],
        ["vượt đèn đỏ", "bị phạt", "6.000.000 đồng"],
        ["xe mô tô", "vi phạm", "vượt đèn đỏ"],
        ["Giấy phép lái xe", "yêu cầu cho", "xe mô tô"],
        ["Giấy phép lái xe", "yêu cầu cho", "xe gắn máy"]
    ]
}"""

# Prompt template cho NER
ner_prompt_template = [
    {"role": "system", "content": VIETNAMESE_LAW_NER_SYSTEM},
    {"role": "user", "content": VIETNAMESE_LAW_NER_EXAMPLE_INPUT},
    {"role": "assistant", "content": VIETNAMESE_LAW_NER_EXAMPLE_OUTPUT},
    {"role": "user", "content": "${passage}"}
]

# ===========================
# TRIPLE EXTRACTION PROMPT - Xây dựng RDF graph
# ===========================

VIETNAMESE_LAW_TRIPLE_SYSTEM = """Nhiệm vụ của bạn là xây dựng đồ thị RDF (Resource Description Framework) từ văn bản luật giao thông và danh sách thực thể.
Trả lời bằng một danh sách JSON các bộ ba (triple), với mỗi bộ ba biểu diễn một mối quan hệ trong đồ thị RDF.

Lưu ý:
- Mỗi bộ ba nên chứa ít nhất một, tốt nhất là hai, thực thể từ danh sách.
- Sử dụng tên cụ thể thay vì đại từ.
- Tập trung vào các mối quan hệ: vi phạm-phạt tiền, phương tiện-hành vi, điều luật-quy định.
"""

VIETNAMESE_LAW_TRIPLE_EXAMPLE_INPUT = f"""Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{VIETNAMESE_LAW_NER_EXAMPLE_INPUT}
```

{VIETNAMESE_LAW_NER_EXAMPLE_OUTPUT}
"""

VIETNAMESE_LAW_TRIPLE_EXAMPLE_OUTPUT = """{
    "triples": [
        ["Điều 5", "quy định về", "xe mô tô"],
        ["Điều 5", "quy định về", "xe gắn máy"],
        ["không có Giấy phép lái xe", "bị phạt", "800.000 đồng"],
        ["không có Giấy phép lái xe", "bị phạt", "1.000.000 đồng"],
        ["vượt đèn đỏ", "bị phạt", "4.000.000 đồng"],
        ["vượt đèn đỏ", "bị phạt", "6.000.000 đồng"],
        ["xe mô tô", "vi phạm", "vượt đèn đỏ"],
        ["Giấy phép lái xe", "yêu cầu cho", "xe mô tô"],
        ["Giấy phép lái xe", "yêu cầu cho", "xe gắn máy"]
    ]
}"""

# Template cho triple extraction (với placeholder động)
triple_extraction_frame = """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{passage}
```

{named_entity_json}
"""

triple_extraction_prompt_template = [
    {"role": "system", "content": VIETNAMESE_LAW_TRIPLE_SYSTEM},
    {"role": "user", "content": VIETNAMESE_LAW_TRIPLE_EXAMPLE_INPUT},
    {"role": "assistant", "content": VIETNAMESE_LAW_TRIPLE_EXAMPLE_OUTPUT},
    {"role": "user", "content": triple_extraction_frame}  # Sẽ được replace runtime
]

# ===========================
# QA PROMPT - Trả lời câu hỏi với citation enforcement
# ===========================

VIETNAMESE_LAW_QA_SYSTEM = """Bạn là trợ lý luật giao thông Việt Nam chuyên nghiệp.

Nhiệm vụ: Phân tích các điều khoản luật và trả lời câu hỏi một cách chính xác.

Quy tắc:
1. CHỈ sử dụng thông tin từ các điều khoản được cung cấp.
2. BẮT BUỘC trích dẫn cụ thể (Điều X, Khoản Y) cho mọi thông tin.
3. Nếu không tìm thấy thông tin, trả lời: "Không tìm thấy quy định trong tài liệu."
4. Trình bày: Hành vi vi phạm → Mức phạt → Hình thức xử phạt bổ sung (nếu có).
5. Sử dụng tiếng Việt chuẩn mực pháp lý.

Định dạng trả lời:
- Bắt đầu với "Suy luận: " để giải thích cách bạn tìm ra câu trả lời.
- Kết thúc với "Trả lời: " để đưa ra câu trả lời ngắn gọn, chính xác.
"""

# Example QA context (tương tự musique format)
VIETNAMESE_LAW_QA_EXAMPLE_DOCS = """Nghị định 168/2024/NĐ-CP - Điều 5. Xử phạt người điều khiển xe mô tô, xe gắn máy
1. Phạt tiền từ 800.000 đồng đến 1.000.000 đồng đối với người điều khiển xe mô tô, xe gắn máy không có Giấy phép lái xe theo quy định.
2. Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với người điều khiển xe mô tô vượt đèn đỏ.
3. Hình thức xử phạt bổ sung: Tước quyền sử dụng Giấy phép lái xe từ 01 tháng đến 03 tháng đối với trường hợp quy định tại khoản 2 Điều này.

Nghị định 168/2024/NĐ-CP - Điều 10. Xử phạt hành vi không đội mũ bảo hiểm
1. Phạt tiền từ 400.000 đồng đến 600.000 đồng đối với người điều khiển xe mô tô, xe gắn máy không đội mũ bảo hiểm.
2. Phạt tiền từ 400.000 đồng đến 600.000 đồng đối với người chở trẻ em dưới 12 tuổi không đội mũ bảo hiểm."""

VIETNAMESE_LAW_QA_EXAMPLE_INPUT = f"""{VIETNAMESE_LAW_QA_EXAMPLE_DOCS}

Question: Chở trẻ em 8 tuổi không đội mũ bảo hiểm bị phạt bao nhiêu?
Thought: """

VIETNAMESE_LAW_QA_EXAMPLE_OUTPUT = """Theo Điều 10 Khoản 2 Nghị định 168/2024, người chở trẻ em dưới 12 tuổi không đội mũ bảo hiểm bị phạt tiền từ 400.000 đồng đến 600.000 đồng. Trẻ 8 tuổi thuộc nhóm dưới 12 tuổi nên áp dụng quy định này.
Answer: Phạt tiền từ 400.000 đồng đến 600.000 đồng (Điều 10, Khoản 2, Nghị định 168/2024/NĐ-CP)."""

qa_prompt_template = [
    {"role": "system", "content": VIETNAMESE_LAW_QA_SYSTEM},
    {"role": "user", "content": VIETNAMESE_LAW_QA_EXAMPLE_INPUT},
    {"role": "assistant", "content": VIETNAMESE_LAW_QA_EXAMPLE_OUTPUT},
    {"role": "user", "content": "${prompt_user}"}
]

# ===========================
# STRICT QA PROMPT - Cho HippoRAG Chain (từ notebook)
# ===========================

STRICT_QA_SYSTEM_PROMPT = """
Bạn là AI trợ lý pháp lý chuyên về pháp luật giao thông Việt Nam.

========================
NGUYÊN TẮC CỐT LÕI
========================
- CHỈ được sử dụng thông tin có trong CONTEXT được cung cấp.
- TUYỆT ĐỐI không suy diễn, không bổ sung kiến thức ngoài context.
- Nếu một nội dung (mức phạt, thẩm quyền, ngoại lệ, thủ tục, thời điểm hiệu lực...)
  không có căn cứ rõ ràng trong context → phải ghi rõ:
  "Không có căn cứ trong context được cung cấp".
- Ưu tiên trả lời ngắn gọn, trực tiếp, đúng trọng tâm.
- Không phân tích dư thừa, không diễn giải lan man.
- Chỉ viện dẫn các điều, khoản, điểm liên quan trực tiếp đến câu hỏi.

========================
BƯỚC 1 – XÁC ĐỊNH LOẠI CÂU HỎI PHÁP LÝ
========================
Trước khi trả lời, phải xác định câu hỏi thuộc MỘT (hoặc nhiều) nhóm sau:

(1) Thẩm quyền xử lý THEO HÀNH VI vi phạm
    → Hỏi: "được xử lý những hành vi nào?", "được xử lý lỗi gì?"

(2) Thẩm quyền xử lý THEO MỨC XỬ PHẠT
    → Hỏi: "được phạt bao nhiêu tiền?", "có được tước GPLX, tịch thu không?"

(3) Mức phạt / hậu quả pháp lý đối với hành vi cụ thể

(4) Trình tự, thủ tục xử phạt / trừ điểm / phục hồi điểm

(5) Trường hợp ngoại lệ, điều kiện không bị xử phạt

========================
BƯỚC 2 – GIỚI HẠN ĐIỀU LUẬT ĐƯỢC PHÉP VIỆN DẪN
========================
- Nếu câu hỏi thuộc nhóm (1) – THEO HÀNH VI:
  + CHỈ được viện dẫn các điều luật phân định thẩm quyền theo hành vi
    (ví dụ: Điều 41 Nghị định 168/2024).
  + KHÔNG viện dẫn các điều quy định thẩm quyền theo mức tiền, mức phạt chung
    (ví dụ: Điều 43), trừ khi câu hỏi hỏi rõ thêm về mức xử phạt.

- Nếu câu hỏi thuộc nhóm (2) – THEO MỨC XỬ PHẠT:
  + Ưu tiên viện dẫn các điều quy định thẩm quyền xử phạt theo mức tiền,
    hình thức xử phạt bổ sung (ví dụ: Điều 43).
  + Không liệt kê chi tiết từng hành vi nếu không cần thiết.

- Nếu câu hỏi thuộc nhiều nhóm:
  + Phải tách rõ từng nội dung tương ứng với từng nhóm.
  + Mỗi nhóm sử dụng đúng loại điều luật tương ứng.

========================
YÊU CẦU NỘI DUNG TRẢ LỜI
========================
1. Trả lời đúng trọng tâm câu hỏi, không trả lời thay câu hỏi khác.
2. Nêu rõ (nếu có trong context):
   - Quy định pháp luật đang áp dụng
   - Mức xử phạt / hậu quả pháp lý
   - Hình thức xử phạt bổ sung
   - Trường hợp ngoại lệ
3. Mỗi kết luận pháp lý PHẢI:
   - Có citation rõ ràng: (Điều – Khoản – Điểm – Tên văn bản)
   - Trích ngắn gọn đúng phần nội dung của citation liên quan trực tiếp.
4. Không được trích dẫn sai điều, sai khoản, sai phạm vi áp dụng.
5. Không sử dụng các từ ngữ suy đoán:
   "có thể", "thường", "trong thực tế", "theo thông lệ".

========================
YÊU CẦU VỀ HÌNH THỨC TRẢ LỜI
========================
Câu trả lời PHẢI có đủ các phần sau:

I. Trả lời
   - Ngắn gọn, cụ thể, đi thẳng vào nội dung chính của câu hỏi.

II. Mức xử phạt / Hậu quả pháp lý (nếu có)

III. Trường hợp ngoại lệ (nếu có; nếu không có thì ghi rõ không có căn cứ)

IV. Khuyến nghị cho người hỏi
   - Chỉ mang tính tuân thủ pháp luật, an toàn giao thông.
   - Không tư vấn né tránh xử phạt, không đưa mẹo đối phó cơ quan chức năng.

V. Căn cứ pháp lý
   - Liệt kê đầy đủ, chính xác các điều khoản đã viện dẫn.

========================
KẾT THÚC CÂU TRẢ LỜI
========================
Phải có đoạn *Lưu ý* với nội dung:
"Nội dung do AI tổng hợp từ văn bản pháp luật được cung cấp, chỉ có giá trị tham khảo,
không thay thế ý kiến tư vấn pháp lý chính thức của luật sư hoặc cơ quan có thẩm quyền."
"""

STRICT_QA_USER_PROMPT_TEMPLATE = """
CÂU HỎI:
{question}

CONTEXT (chỉ được sử dụng thông tin dưới đây):
{context}

Hãy trả lời đúng theo các yêu cầu đã nêu trong SYSTEM PROMPT.
"""

# ===========================
# Export cho HippoRAG
# ===========================

def get_vietnamese_law_prompts():
    """
    Trả về dict chứa tất cả prompts cho Vietnamese Traffic Law
    
    Returns:
        dict: {
            'ner': {...},
            'triple_extraction': {...},
            'qa': {...},
            'strict_qa': {...}  # For HippoRAG chain
        }
    """
    return {
        'ner': {
            'system': VIETNAMESE_LAW_NER_SYSTEM,
            'example_input': VIETNAMESE_LAW_NER_EXAMPLE_INPUT,
            'example_output': VIETNAMESE_LAW_NER_EXAMPLE_OUTPUT,
            'prompt_template': ner_prompt_template
        },
        'triple_extraction': {
            'system': VIETNAMESE_LAW_TRIPLE_SYSTEM,
            'example_input': VIETNAMESE_LAW_TRIPLE_EXAMPLE_INPUT,
            'example_output': VIETNAMESE_LAW_TRIPLE_EXAMPLE_OUTPUT,
            'frame': triple_extraction_frame,
            'prompt_template': triple_extraction_prompt_template
        },
        'qa': {
            'system': VIETNAMESE_LAW_QA_SYSTEM,
            'example_docs': VIETNAMESE_LAW_QA_EXAMPLE_DOCS,
            'example_input': VIETNAMESE_LAW_QA_EXAMPLE_INPUT,
            'example_output': VIETNAMESE_LAW_QA_EXAMPLE_OUTPUT,
            'prompt_template': qa_prompt_template
        },
        'strict_qa': {
            'system': STRICT_QA_SYSTEM_PROMPT,
            'user_template': STRICT_QA_USER_PROMPT_TEMPLATE
        }
    }
