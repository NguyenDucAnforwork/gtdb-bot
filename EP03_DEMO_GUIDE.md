# EP-03: Demo Guide - Cập nhật & Quản trị văn bản pháp luật

## 🎯 Use Case 3 Overview

**EPIC ID**: EP-03  
**Owner**: Legal Ops Lead  
**Business Value**: Đảm bảo độ chính xác trích dẫn; nền tảng cho mọi surface; đáp ứng RAW-003/002

### Acceptance Criteria
- **AC1**: Crawler + chuẩn hoá JSONL → cập nhật KG/VecStore ≤7 ngày
- **AC2**: Gắn trạng thái hiệu lực; hiển thị diff cũ–mới  
- **AC3**: Coverage ≥95% nguồn GTĐB trọng yếu (Luật/NĐ/TT)

---

## 📋 Demo Flow

### 1️⃣ Chuyển sang chế độ Admin

**Command**: `/mode admin`

**Expected Response**:
```
👨‍💼 ĐÃ CHUYỂN SANG: CHẾ ĐỘ ADMIN
📋 Phong cách: Quản trị văn bản pháp luật - Chỉ dành cho maintainer
💡 Sử dụng lệnh /admin help để xem menu đầy đủ
```

---

### 2️⃣ Index văn bản từ URL (AC1 Demo)

**Command**: `/index <URL>`

**Example**:
```
/index https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Nghi-dinh-158-2024-ND-CP-quy-dinh-hoat-dong-van-tai-duong-bo-636875.aspx
```

**Flow**:
1. Bot gửi thông báo bắt đầu crawl
2. Crawl nội dung từ thuvienphapluat.vn
3. Phân tích và tách văn bản theo Điều/Khoản
4. Index vào Qdrant (Vector Search)
5. Index vào HippoRAG (Knowledge Graph)
6. Cập nhật coverage statistics

**Expected Response**:
```
✅ INDEX THÀNH CÔNG!

📄 VĂN BẢN:
   - Mã: 158/2024/NĐ-CP
   - Tiêu đề: Nghị định 158/2024/NĐ-CP quy định hoạt động vận tải đường bộ
   - Nguồn: https://thuvienphapluat.vn/van-ban/.../

📊 KẾT QUẢ:
   - Số đoạn văn: 156 passages
   - Đã index vào: Qdrant ✅ + HippoRAG ✅
   
📈 COVERAGE:
   - Tổng đã index: 8/12
   - Tỷ lệ: 66.7%
   - Mục tiêu: ≥95% ⚠️

⏰ SLA: Hoàn thành trong ≤7 ngày ✅
```

---

### 3️⃣ Kiểm tra Coverage (AC3 Demo)

**Command**: `/admin coverage`

**Expected Response**:
```
📊 COVERAGE REPORT - NGUỒN TRỌNG YẾU

✅ ĐÃ CÓ (8/12 - 66.7%):
━━━━━━━━━━━━━━━━━━━━
📚 Luật:
   ✅ Luật GTĐB 2008
   ✅ Luật sửa đổi 2024

📋 Nghị định:
   ✅ NĐ 100/2019
   ✅ NĐ 123/2021
   ✅ NĐ 168/2024
   ✅ NĐ 158/2024 (Mới thêm)

📑 Thông tư:
   ✅ TT 24/2023
   ✅ TT 65/2020

❌ THIẾU (4/12 - 33.3%):
━━━━━━━━━━━━━━━━━━━━
📋 Nghị định:
   ⚠️ NĐ 15/2023
   ⚠️ NĐ 171/2013

📑 Thông tư:
   ⚠️ TT 11/2020
   ⚠️ TT 58/2015

🎯 MỤC TIÊU: ≥95% (12/12 văn bản)
📊 HIỆN TẠI: 66.7% ⚠️
💡 CẦN THÊM: 4 văn bản để đạt mục tiêu
```

---

### 4️⃣ Xem văn bản chờ duyệt (US1 Demo)

**Command**: `/admin pending`

**Expected Response**:
```
📋 DANH SÁCH VĂN BẢN CHỜ DUYỆT:

1. 📄 Nghi-dinh-15-2023-ND-CP.pdf
   - Ngày upload: 2024-11-25 14:30
   - Kích thước: 2.3 MB
   - Lệnh: /admin approve Nghi-dinh-15-2023-ND-CP.pdf

2. 📄 Thong-tu-11-2020-TT-BCA.pdf
   - Ngày upload: 2024-11-25 15:45
   - Kích thước: 1.8 MB
   - Lệnh: /admin approve Thong-tu-11-2020-TT-BCA.pdf

━━━━━━━━━━━━━━━━━━━━
📊 Tổng: 2 văn bản chờ duyệt
⏰ SLA: Xử lý trong vòng 7 ngày
```

---

### 5️⃣ Duyệt văn bản (US1 Demo)

**Command**: `/admin approve <filename>`

**Example**:
```
/admin approve Nghi-dinh-15-2023-ND-CP.pdf
```

**Expected Response**:
```
✅ ĐÃ DUYỆT VĂN BẢN

📄 Văn bản: Nghi-dinh-15-2023-ND-CP.pdf
⏰ Thời gian duyệt: 2024-11-26 10:15
👤 Người duyệt: Admin

📋 Hành động tiếp theo:
   1. Văn bản đã được chuyển vào thư mục approved/
   2. Sử dụng /admin index Nghi-dinh-15-2023-ND-CP.pdf để đưa vào hệ thống
   3. Hoặc /admin index-all để index tất cả văn bản đã duyệt
```

---

### 6️⃣ So sánh phiên bản (US2/AC2 Demo)

**Command**: `/admin diff <tên> <v1> <v2>`

**Example**:
```
/admin diff "NĐ 168/2024" "100/2019" "168/2024"
```

**Expected Response**:
```
🔄 SO SÁNH PHIÊN BẢN VĂN BẢN

📋 Văn bản: Nghị định về xử phạt GTĐB
   - Phiên bản cũ: 100/2019/NĐ-CP
   - Phiên bản mới: 168/2024/NĐ-CP

📊 THAY ĐỔI CHỦ YẾU:

➕ Điều mới (5 điều):
   • Điều 7: Xử phạt không đội mũ bảo hiểm (mức phạt tăng)
   • Điều 15: Xử phạt vi phạm tốc độ (bổ sung camera AI)
   • ...

📝 Điều sửa đổi (12 điều):
   • Điều 44: Mức phạt từ 800k-1tr → 1tr-2tr
   • Điều 58: Thêm trường hợp tước GPLX 2 tháng
   • ...

❌ Điều bãi bỏ (3 điều):
   • Điều 32: Đã được thay thế bởi Điều 7 mới
   • ...

🏷️ TRẠNG THÁI HIỆU LỰC:
   • 100/2019: Hết hiệu lực (superseded)
   • 168/2024: Đang có hiệu lực (active)
```

---

### 7️⃣ Cập nhật trạng thái hiệu lực (AC2 Demo)

**Command**: `/admin status <tên> <trạng thái>`

**Example**:
```
/admin status "NĐ 100/2019" superseded
```

**Trạng thái hợp lệ**: `active`, `expired`, `superseded`, `draft`

**Expected Response**:
```
✅ ĐÃ CẬP NHẬT TRẠNG THÁI

📄 Văn bản: NĐ 100/2019/NĐ-CP
🏷️ Trạng thái mới: superseded
⏰ Thời gian cập nhật: 2024-11-26 10:20

📋 Lưu ý:
   - Văn bản này đã được thay thế bởi NĐ 168/2024
   - Hệ thống sẽ ưu tiên trích dẫn từ văn bản mới
   - Văn bản cũ vẫn được giữ để tra cứu lịch sử
```

---

### 8️⃣ Xem thống kê tổng quan

**Command**: `/admin stats`

**Expected Response**:
```
📊 THỐNG KÊ HỆ THỐNG ADMIN

📋 VĂN BẢN:
   - Chờ duyệt: 2 văn bản
   - Đã duyệt: 8 văn bản
   - Từ chối: 0 văn bản
   - Tổng: 10 văn bản

📈 COVERAGE (AC3):
   - Yêu cầu: 12 văn bản trọng yếu
   - Đã có: 8 văn bản
   - Tỷ lệ: 66.7% ⚠️
   - Mục tiêu: ≥95%

⏰ HIỆU SUẤT (AC1):
   - Thời gian index: ≤7 ngày
   - SLA: Đang đạt ✅

🔧 LỆNH NHANH:
   /admin pending - Xem văn bản chờ duyệt
   /admin coverage - Chi tiết coverage
   /admin index-all - Index tất cả
```

---

## 🎬 Complete Demo Script

### Scenario: Index Nghị định 158/2024 mới

```
User: /mode admin
Bot: 👨‍💼 ĐÃ CHUYỂN SANG: CHẾ ĐỘ ADMIN...

User: /index https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Nghi-dinh-158-2024-ND-CP-quy-dinh-hoat-dong-van-tai-duong-bo-636875.aspx
Bot: 🚀 Đang crawl & index văn bản...
Bot: ✅ INDEX THÀNH CÔNG! (chi tiết như trên)

User: /admin coverage
Bot: 📊 COVERAGE REPORT... (66.7%)

User: /admin stats
Bot: 📊 THỐNG KÊ HỆ THỐNG... (8/12 văn bản)

User: Mức phạt không đội mũ bảo hiểm là bao nhiêu?
Bot: [Sử dụng văn bản vừa index để trả lời chính xác]
```

---

## 🔧 Technical Implementation

### Architecture Components

1. **Web Crawler** (`src/ingestion/crawler.py`)
   - Crawl từ thuvienphapluat.vn
   - Extract law code, title, content
   - Clean và normalize text

2. **Document Processor** (`src/ingestion/updater.py`)
   - Split theo Điều/Khoản/Điểm
   - Generate embeddings
   - Update Qdrant + HippoRAG

3. **Admin Bot** (`src/persona/admin_bot.py`)
   - Document approval workflow
   - Version tracking & diff
   - Coverage monitoring
   - Status management

4. **API Handler** (`app.py`)
   - `/index <url>` endpoint
   - `/admin <command>` routing
   - Error handling & notifications

### Data Flow

```
URL Input
    ↓
Crawler → Extract Content
    ↓
Splitter → Passages (Điều/Khoản)
    ↓
Embeddings Generator
    ↓
    ├→ Qdrant (Vector Search)
    └→ HippoRAG (Knowledge Graph)
    ↓
Coverage Update
    ↓
Success Notification
```

---

## ✅ Success Criteria Validation

### AC1: Crawler + Index ≤7 ngày
- ✅ Automated crawling từ URL
- ✅ Real-time indexing vào Qdrant + HippoRAG
- ✅ Processing time < 5 minutes per document
- ✅ SLA tracking và notification

### AC2: Trạng thái hiệu lực + Diff
- ✅ Status management (active/expired/superseded/draft)
- ✅ Version comparison (`/admin diff`)
- ✅ Historical tracking
- ✅ Automatic priority for latest version

### AC3: Coverage ≥95%
- ✅ Critical sources tracking
- ✅ Real-time coverage monitoring
- ✅ Missing documents alert
- ✅ Progress visualization

---

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install beautifulsoup4
   ```

2. **Start Server**:
   ```bash
   uvicorn app:app --port 8000 --reload
   ```

3. **Test on Messenger**:
   - Send: `/mode admin`
   - Send: `/index https://thuvienphapluat.vn/...`
   - Send: `/admin coverage`
   - Send: `/admin stats`

---

## 📝 Notes

- Admin mode chỉ dành cho maintainer
- Mọi thao tác được log lại
- Index process có thể mất 2-5 phút tùy văn bản
- Coverage target là ≥95% nguồn trọng yếu
- SLA là ≤7 ngày từ khi văn bản mới xuất hiện
