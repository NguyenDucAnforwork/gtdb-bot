# TEST CASES CHO CHATBOT LUẬT AN TOÀN GIAO THÔNG

## 🎯 Mục tiêu test:
- Kiểm tra khả năng hiểu và phân loại câu hỏi
- Test long-term memory và follow-up capabilities
- Đánh giá độ chính xác thông tin pháp lý
- Kiểm tra tính tương tác và user experience

---

## 📋 NHÓM TEST 1: GREETING & BASIC INTERACTION

### Test 1.1: Chào hỏi đầu tiên
**Tin nhắn:** `Xin chào!`

**Kỳ vọng:**
- Phân loại: GREETING
- Response thân thiện với giới thiệu vai trò
- Có 2-3 câu hỏi follow-up về luật giao thông
- Không cần retrieval từ database

### Test 1.2: Cảm ơn
**Tin nhắn:** `Cảm ơn bạn nhé`

**Kỳ vọng:** 
- Response lịch sự
- Gợi ý tiếp tục hỏi thêm

---

## 📋 NHÓM TEST 2: SIMPLE LEGAL QUERIES

### Test 2.1: Câu hỏi cơ bản về mức phạt
**Tin nhắn:** `Mức phạt không đội mũ bảo hiểm là bao nhiêu?`

**Kỳ vọng:**
- Phân loại: SIMPLE_LEGAL
- Sử dụng vector retriever
- Trả lời chính xác mức phạt (số tiền cụ thể)
- Có trích dẫn Nghị định, Điều, Khoản
- Có câu hỏi follow-up liên quan

### Test 2.2: Quy định tốc độ
**Tin nhắn:** `Tốc độ tối đa trong khu dân cư là bao nhiêu?`

**Kỳ vọng:**
- Response nhanh với thông tin chính xác
- Trích dẫn quy định pháp lý
- Gợi ý câu hỏi về các khu vực khác

### Test 2.3: Phạt vượt đèn đỏ
**Tin nhắn:** `Phạt vượt đèn đỏ có bị tạm giữ bằng lái không?`

**Kỳ vọng:**
- Thông tin về mức phạt và biện pháp xử lý
- Phân biệt các trường hợp khác nhau nếu có

---

## 📋 NHÓM TEST 3: COMPLEX LEGAL QUERIES

### Test 3.1: So sánh quy định theo thời gian
**Tin nhắn:** `Từ năm 2022, mức phạt không đội mũ bảo hiểm có thay đổi gì so với trước không?`

**Kỳ vọng:**
- Phân loại: COMPLEX_LEGAL  
- Sử dụng cả HippoRAG + vector retriever
- So sánh cụ thể giữa các mốc thời gian
- Phân tích sự thay đổi
- Trích dẫn từ nhiều nguồn

### Test 3.2: Multiple conditions
**Tin nhắn:** `Nếu không đội mũ bảo hiểm và còn vi phạm tốc độ thì sẽ bị phạt như thế nào?`

**Kỳ vọng:**
- Phân tích đa yếu tố
- Giải thích cách tính phạt kết hợp
- Thông tin chi tiết và toàn diện

---

## 📋 NHÓM TEST 4: LONG-TERM MEMORY (QUAN TRỌNG NHẤT!)

### Test 4.1: Memory sequence - Basic conversation
**Lần 1:** `Xin chào, tôi muốn hỏi về luật giao thông`
**Lần 2:** `Mức phạt không đội mũ bảo hiểm là bao nhiêu?`
**Lần 3:** `Còn nếu không có bằng lái thì sao?`
**Lần 4:** `Vậy tổng cộng sẽ bị phạt bao nhiêu?`
**Lần 5:** `Thời hạn đóng phạt là gì?`

**Kỳ vọng:**
- Bot nhớ context về "không đội mũ bảo hiểm + không có bằng lái"
- Lần 4: Hiểu "tổng cộng" = phạt cả 2 loại vi phạm
- Lần 5: Hiểu "thời hạn đóng phạt" liên quan đến các vi phạm trên
- Context được maintain qua 5 turns

### Test 4.2: Memory với specific context
**Lần 1:** `Tôi đang quan tâm về các quy định mới nhất`
**Lần 2:** `Mức phạt rượu bia khi lái xe là bao nhiêu?`
**Lần 3:** `Vậy còn ma túy thì sao?`
**Lần 4:** `Có thể kháng cáo được không?`
**Lần 5:** `Quy trình kháng cáo như thế nào?`
**Lần 6:** `Nếu thua kiện thì đóng phạt gấp đôi không?`

**Kỳ vọng:**
- Lần 3: Bot hiểu "ma túy" trong context "lái xe"
- Lần 4: Bot hiểu "kháng cáo" liên quan đến "phạt ma túy lái xe"
- Lần 5-6: Bot duy trì chuỗi ngữ cảnh về "kháng cáo phạt ma túy lái xe"
- Memory context phức tạp được maintain chính xác qua 6 turns

### Test 4.3: Context switching
**Lần 1:** `Phạt không đội mũ bảo hiểm là bao nhiêu?`
**Lần 2:** `Cảm ơn bạn`
**Lần 3:** `Tôi muốn hỏi về đăng ký xe`
**Lần 4:** `Cần giấy tờ gì?`
**Lần 5:** `Phí bao nhiêu?`
**Lần 6:** `Quay lại chủ đề mũ bảo hiểm lúc nãy, nếu không có giấy tờ thì sao?`

**Kỳ vọng:**
- Context switch từ "mũ bảo hiểm" sang "đăng ký xe" (lần 3)
- Lần 4-5: Hiểu context "giấy tờ + phí cho đăng ký xe"
- Lần 6: Context switch ngược lại + kết hợp 2 chủ đề
- Bot nhớ cả 2 context: "mũ bảo hiểm" và "giấy tờ đăng ký xe"

### Test 4.4: Long-term persistent memory
**Session 1 - Ngày 1:**
- `Tôi tên An, tôi đang học luật tại Đại học Quốc gia`
- `Tôi đang nghiên cứu về luật giao thông mới`
- `Mức phạt không đội mũ bảo hiểm là bao nhiêu?`

**Session 2 - Ngày 1 (sau 2h):**
- `Tôi là An đây, hỏi tiếp về luật giao thông`
- `Lúc nãy tôi muốn hỏi về vi phạm tốc độ`

**Session 3 - Ngày 2:**
- `Hôm qua tôi có hỏi về mũ bảo hiểm và tốc độ`
- `Bây giờ tôi muốn so sánh 2 mức phạt đó`

**Kỳ vọng:**
- Bot nhớ tên "An" và thông tin "học luật tại ĐHQG"
- Bot nhớ 2 chủ đề đã hỏi: "mũ bảo hiểm" và "tốc độ"
- Cross-session memory hoạt động qua 3 sessions
- Bot hiểu "so sánh 2 mức phạt đó" = so sánh mũ bảo hiểm vs tốc độ

---

## 📋 NHÓM TEST 5: EDGE CASES

### Test 5.1: Ambiguous questions
**Tin nhắn:** `Nó như thế nào?`

**Kỳ vọng:**
- Bot yêu cầu clarification
- Gợi ý cách hỏi rõ ràng hơn

### Test 5.2: Out-of-domain
**Tin nhắn:** `Công thức nấu phở là gì?`

**Kỳ vọng:**
- Từ chối lịch sự
- Redirect về domain luật giao thông

### Test 5.3: Current events
**Tin nhắn:** `Luật giao thông mới nhất hiện tại thay đổi gì?`

**Kỳ vọng:**
- Phân loại: WEB_SEARCH
- Tìm kiếm thông tin mới nhất
- Trích dẫn nguồn tin

---

## 📋 NHÓM TEST 6: FOLLOW-UP CAPABILITIES

### Test 6.1: Natural follow-ups
**Lần 1:** `Phạt không đội mũ bảo hiểm`
*(Bot trả lời và gợi ý: "Có thể kháng cáo quyết định phạt không?")*
**Lần 2:** `Có thể kháng cáo quyết định phạt không?`

**Kỳ vọng:**
- Bot cung cấp thông tin về quy trình kháng cáo
- Liên kết với context trước đó

### Test 6.2: Progressive questioning
**Lần 1:** `Mức phạt vượt tốc độ`
**Lần 2:** `Còn bị tước bằng lái không?` 
**Lần 3:** `Tước bao lâu?`
**Lần 4:** `Làm thế nào để lấy lại?`

**Kỳ vọng:**
- Conversation flow tự nhiên
- Context maintained qua multiple turns
- Thông tin ngày càng chi tiết

---

## 🎯 CÁCH ĐÁNH GIÁ:

### ✅ PASS Criteria:
- **Memory**: Bot nhớ context ít nhất 5-7 turns, cross-session >24h
- **Accuracy**: Thông tin pháp lý chính xác với trích dẫn
- **Classification**: Phân loại query type đúng >90%
- **Follow-up**: Gợi ý tự nhiên, không có "câu hỏi gợi ý" trong response
- **UX**: Response time <5s, không crash
- **Context Switching**: Chuyển đổi và kết hợp nhiều context

### ❌ FAIL Criteria:
- Thông tin pháp lý sai lệch
- Memory context bị lost sau 2-3 turns
- Phân loại query sai (ví dụ: complex legal -> simple legal)
- Không trích dẫn nguồn pháp lý
- Response generic không relate đến context

---

## 📁 KẾT QUẢ MONG ĐỢI:

1. **Memory Test**: 95% context retention qua 7 turns, 85% cross-session retention
2. **Legal Accuracy**: 95% thông tin chính xác với citation hoặc fallback retrieval
3. **Query Classification**: 92% phân loại đúng type với LLM classifier
4. **Follow-up Quality**: Gợi ý tự nhiên, không duplicate, phù hợp với context
5. **Performance**: Average response time < 4s (bao gồm fallback retrievers)
6. **Retrieval Robustness**: Vector → Web Search → HippoRAG fallback hoạt động

---

*Lưu ý: Test theo thứ tự từ cơ bản đến phức tạp. Đặc biệt chú ý TEST NHÓM 4 về long-term memory và hệ thống fallback retrieval mới (Vector → Web Search → HippoRAG).*