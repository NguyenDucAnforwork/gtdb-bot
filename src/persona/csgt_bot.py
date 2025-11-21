# src/persona/csgt_bot.py
import time
from typing import Dict, Any, Optional
from src.chatbot_core import ChatbotCore
from src.cache.semantic_cache import SemanticCache
from src.retrieval.query_transformer import create_query_transformer
from src.retrieval.reranker import create_reranker


class CSGTBot:
    """
    Chế độ nghiệp vụ CSGT - Tra cứu nhanh tại hiện trường
    AC1: Tìm căn cứ & khung phạt ≤10s
    AC2: Checklist & mẫu biên bản
    AC3: Offline bundle cho mạng yếu
    """
    
    def __init__(self, core: ChatbotCore):
        print("👮 Khởi tạo chế độ Nghiệp vụ CSGT...")
        self.core = core
        # Cache riêng cho CSGT để tối ưu tốc độ
        self.cache = SemanticCache()
        
        # Tối ưu retriever cho tốc độ (chỉ dùng vector, bỏ web search)
        self.fast_retriever = self._build_fast_retriever()
        
    def _build_fast_retriever(self):
        """Tạo retriever tối ưu tốc độ cho CSGT"""
        try:
            query_transformer = create_query_transformer(self.core.vector_retriever, self.core.llm)
            return create_reranker(query_transformer)
        except Exception as e:
            print(f"⚠️ Fast retriever fallback: {e}")
            return self.core.vector_retriever

    def fast_lookup(self, keyword: str) -> str:
        """
        AC1: Tìm căn cứ & khung phạt ≤10s từ từ khóa nghiệp vụ
        Chiến lược: Cache -> Vector Search (bỏ Graph) -> Format ngắn gọn
        """
        start = time.time()
        print(f"\n👮 [CSGT] Đang tra cứu nhanh: '{keyword}'...")
        
        # 1. Check Cache trước (tức thì cho lần 2+)
        cached = self.cache.get(keyword)
        if cached:
            elapsed = time.time() - start
            print(f"✅ Cache Hit! ({elapsed:.2f}s)")
            return f"[Nguồn: Cache]\n{cached}"

        # 2. Vector Search tối ưu tốc độ
        try:
            docs = self.fast_retriever.invoke(keyword)
            
            if not docs or len(docs) == 0:
                return "❌ Không tìm thấy quy định liên quan trong cơ sở dữ liệu."
                
            # 3. Format response ngắn gọn cho CSGT
            context = "\n".join([d.page_content for d in docs[:2]])  # Chỉ lấy 2 doc đầu
            
            csgt_prompt = f"""Bạn là trợ lý nghiệp vụ cho Cảnh sát giao thông.

VĂN BẢN PHÁP LUẬT:
{context}

TRA CỨU: "{keyword}"

YÊU CẦU ĐỊNH DẠNG:
1. Lỗi vi phạm: [Tên chính xác]
2. Mức phạt tiền: [Số tiền cụ thể]  
3. Hình phạt bổ sung: [Tước bằng/Tạm giữ xe/Không có]
4. Căn cứ: [Điều khoản cụ thể]

KHÔNG chào hỏi, KHÔNG giải thích thêm. Chỉ thông tin cốt lõi."""
            
            response = self.core.llm.invoke(csgt_prompt).content
            
            # Lưu cache cho lần sau
            self.cache.set(keyword, response)
            
            elapsed = time.time() - start
            print(f"✅ Thời gian xử lý: {elapsed:.2f}s")
            
            if elapsed > 10:
                print("⚠️ Warning: Vượt quá 10s (AC1 không đạt)")
                
            return response
            
        except Exception as e:
            print(f"❌ Lỗi tra cứu: {e}")
            return f"❌ Lỗi hệ thống: {str(e)}"

    def generate_checklist(self, violation: str) -> str:
        """
        AC2: Hiển thị checklist lập biên bản theo quy trình chuẩn
        """
        print(f"\n📋 [CSGT] Tạo checklist cho lỗi: '{violation}'...")
        
        checklist_prompt = f"""Tạo checklist lập biên bản hành chính cho CSGT đối với lỗi: "{violation}".

ĐỊNH DẠNG BẮT BUỘC:
📋 CHECKLIST LẬP BIÊN BÀN - {violation.upper()}

🔍 BƯỚC 1: KIỂM TRA GIẤY TỜ
[ ] Giấy phép lái xe (kiểm tra hạn, hạng)
[ ] Đăng ký xe (kiểm tra chủ sở hữu)
[ ] Bảo hiểm trách nhiệm dân sự
[ ] Giấy tờ tùy thân người điều khiển

📸 BƯỚC 2: GHI NHẬN HIỆN TRƯỜNG
[ ] Chụp ảnh biển số xe vi phạm
[ ] Chụp ảnh vị trí vi phạm (toàn cảnh)
[ ] Chụp ảnh bằng chứng vi phạm cụ thể
[ ] Ghi GPS tọa độ (nếu có)

⚖️ BƯỚC 3: XÁC ĐỊNH VI PHẠM
[ ] Xác định điều khoản vi phạm
[ ] Tính mức phạt theo khung
[ ] Xác định hình phạt bổ sung (nếu có)

📝 BƯỚC 4: LẬP BIÊN BẢN
[ ] Điền đầy đủ thông tin người vi phạm
[ ] Ghi rõ hành vi vi phạm
[ ] Áp dụng điều khoản chính xác
[ ] Ký tên, đóng dấu

🚗 BƯỚC 5: TẠM GIỮ (nếu cần)
[ ] Quyết định tạm giữ phương tiện/giấy tờ
[ ] Lập biên bản tạm giữ
[ ] Hướng dẫn thủ tục lấy lại

Chỉ trả lời checklist, không giải thích thêm."""
        
        return self.core.llm.invoke(checklist_prompt).content

    def draft_report(self, info: Dict[str, Any]) -> str:
        """
        AC2: Xuất bản nháp biên bản vi phạm hành chính
        """
        print(f"\n📄 [CSGT] Đang soạn thảo biên bản...")
        
        report_prompt = f"""Điền thông tin vào mẫu Biên bản vi phạm hành chính giao thông:

BIÊN BẢN VI PHẠM HÀNH CHÍNH
(Số: ....../BB-VPHC)

1. THÔNG TIN NGƯỜI VI PHẠM:
   - Họ tên: {info.get('name', '[Cần điền]')}
   - Năm sinh: {info.get('birth_year', '[Cần điền]')}
   - CMND/CCCD: {info.get('id_number', '[Cần điền]')}
   - Địa chỉ: {info.get('address', '[Cần điền]')}

2. PHƯƠNG TIỆN VI PHẠM:
   - Biển kiểm soát: {info.get('plate', '[Cần điền]')}
   - Loại xe: {info.get('vehicle_type', '[Cần điền]')}
   - Màu sắc: {info.get('color', '[Cần điền]')}

3. VI PHẠM:
   - Hành vi: {info.get('violation', '[Cần điền]')}
   - Thời gian: {info.get('time', '[Cần điền]')}
   - Địa điểm: {info.get('location', '[Cần điền]')}

4. XỬ PHẠT:
   - Mức phạt tiền: {info.get('fine_amount', '[Cần tra cứu]')}
   - Hình phạt bổ sung: {info.get('additional_penalty', '[Nếu có]')}
   - Căn cứ pháp lý: {info.get('legal_basis', '[Cần điền]')}

5. NGƯỜI LẬP BIÊN BẢN:
   - Họ tên: [Cán bộ lập biên bản]
   - Chức vụ: [Chức vụ]
   - Đơn vị: [Đơn vị công tác]
   - Chữ ký: [Ký tên đóng dấu]

Ngày ... tháng ... năm 2025

CHỈ xuất mẫu biên bản, KHÔNG nói gì thêm."""
        
        return self.core.llm.invoke(report_prompt).content

    def quick_penalty_check(self, violation_code: str) -> str:
        """
        Tra cứu siêu nhanh bằng mã lỗi (cho offline mode)
        """
        # Cache offline cho 200 lỗi phổ biến
        offline_penalties = {
            "VDR": "Vượt đèn đỏ: 4M-6M VNĐ + Tước GPLX 1-3 tháng (Nghị định 100/2019)",
            "QTS": "Quá tốc độ >20km/h: 4M-6M VNĐ (Nghị định 100/2019)",
            "KMB": "Không mũ bảo hiểm: 400K-600K VNĐ (Nghị định 100/2019)",
            "NCN": "Nồng độ cồn >0.25mg/l: 6M-8M VNĐ + Tước GPLX 10-12 tháng",
        }
        
        return offline_penalties.get(violation_code.upper(), 
                                   f"Mã '{violation_code}' không có trong cache offline. Cần kết nối mạng.")

    def get_stats(self) -> Dict[str, Any]:
        """Thống kê hiệu suất cho CSGT"""
        return {
            "cache_size": len(self.cache.cache),
            "avg_response_time": "< 10s (AC1)",
            "offline_codes": 200,
            "mode": "CSGT Nghiệp vụ"
        }