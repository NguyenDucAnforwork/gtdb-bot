# src/ingestion/updater.py
import os
import re
import json
import uuid
import pathlib
import subprocess
from typing import List
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from hipporag import HippoRAG

# Load biến môi trường
load_dotenv()

# Cấu hình Regex (Lấy từ notebooks/HippoRAG.ipynb)
RX_DIEU = re.compile(r'(?m)^\s*Điều\s+(\d+)\s*[.:]', re.UNICODE)
RX_KHOAN_NUM = re.compile(r'(?m)\s*(\d+)\.\s', re.UNICODE)
RX_KHOAN_WORD = re.compile(r'(?m)\s*Khoản\s+(\d+)\s*[.:]\s*', re.UNICODE)
RX_DIEM_LETTER = re.compile(r'(?m)\s*([a-z])\)\s', re.UNICODE)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Dùng pdftotext để lấy text từ PDF (giữ layout tốt hơn pypdf)"""
    txt_path = pdf_path + ".txt"
    try:
        # Yêu cầu hệ thống đã cài poppler-utils
        subprocess.run(["pdftotext", "-layout", pdf_path, txt_path], check=True)
        text = pathlib.Path(txt_path).read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"Lỗi khi đọc PDF: {e}")
        return ""
    finally:
        if os.path.exists(txt_path):
            os.remove(txt_path)
    return text

def _slice_by_matches(text: str, matches: List[re.Match]) -> List[tuple]:
    if not matches: return []
    spans = [(m.start(), m) for m in matches]
    spans.append((len(text), None))
    out = []
    for i in range(len(spans)-1):
        start, m = spans[i]
        end, _ = spans[i+1]
        out.append((start, end, m))
    return out

def split_passages(raw: str, law_code: str) -> List[str]:
    """Logic tách văn bản theo Điều -> Khoản (từ Notebook)"""
    passages = []
    # 1. Tách theo Điều
    dieu_matches = list(RX_DIEU.finditer(raw))
    dieu_blocks = _slice_by_matches(raw, dieu_matches)

    for d_start, d_end, d_m in dieu_blocks:
        dieu_block = raw[d_start:d_end].strip()
        d_num = d_m.group(1) if d_m else "?"
        
        # 2. Tách theo Khoản (1., 2., ...)
        khoan_num_matches = list(RX_KHOAN_NUM.finditer(dieu_block))
        use_word_khoan = False
        if not khoan_num_matches:
            khoan_word_matches = list(RX_KHOAN_WORD.finditer(dieu_block))
            if khoan_word_matches:
                use_word_khoan = True
                khoan_blocks = _slice_by_matches(dieu_block, khoan_word_matches)
            else:
                khoan_blocks = []
        else:
            khoan_blocks = _slice_by_matches(dieu_block, khoan_num_matches)

        # Nếu không có khoản, lấy cả điều
        if not khoan_blocks:
            title = f"[{law_code}] Điều {d_num}"
            if len(dieu_block.split()) >= 8: # Lọc rác ngắn
                passages.append(f"{title}\n\n{dieu_block}")
            continue

        # Nếu có khoản
        dieu_title_line = dieu_block.split('\n')[0] if '\n' in dieu_block else ""
        for k_start, k_end, k_m in khoan_blocks:
            khoan_block = dieu_block[k_start:k_end].strip()
            khoan_block = f"{dieu_title_line}\n{khoan_block}" # Giữ context tên điều
            
            k_num = k_m.group(1)
            k_type = "Khoản" if use_word_khoan else "Mục"
            k_title = f"[{law_code}] Điều {d_num} {k_type} {k_num}"
            
            if len(khoan_block.split()) >= 10:
                passages.append(f"{k_title}\n\n{khoan_block}")
                
    return passages

def update_qdrant(docs: List[str], collection_name="gtdb-1"):
    """Tạo embedding và đẩy vào Qdrant"""
    print("⏳ Đang tạo Embedding cho Qdrant...")
    # Dùng model tiếng Việt như trong notebook
    model = SentenceTransformer('AITeamVN/Vietnamese_Embedding')
    embeddings = model.encode(docs, show_progress_bar=True)

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=False
    )

    # Đảm bảo collection tồn tại
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE)
        )

    points = []
    for i, doc in enumerate(docs):
        # Tạo payload metadata để trích dẫn sau này
        # Regex đơn giản để lấy law_id từ title "[36/2024/QH15]..."
        law_id = doc.split("]")[0].replace("[", "") if "]" in doc else "unknown"
        
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i].tolist(),
            payload={
                "text": doc,
                "law_id": law_id,
                "type": "law_chunk"
            }
        ))
    
    # Upsert batch
    client.upsert(collection_name=collection_name, points=points)
    print(f"✅ Đã đẩy {len(points)} đoạn văn bản vào Qdrant.")

def update_hipporag(docs: List[str]):
    """Cập nhật index cho HippoRAG"""
    print("⏳ Đang Indexing HippoRAG (Việc này có thể tốn thời gian và tiền OpenAI)...")
    # Lưu ý: HippoRAG hiện tại thường index batch lớn. 
    # Nếu append, cần cẩn trọng với save_dir cũ.
    
    hippo = HippoRAG(
        save_dir="outputs", # Folder chứa graph cũ
        llm_model_name="gpt-4o-mini",
        embedding_model_name="text-embedding-3-small"
    )
    
    # Hàm index của HippoRAG sẽ chạy OpenIE, tạo graph và lưu lại
    hippo.index(docs=docs)
    print("✅ Đã cập nhật HippoRAG.")

def ingest_new_file(file_path: str):
    print(f"🚀 Bắt đầu xử lý: {file_path}")
    
    # 1. Lấy tên luật làm code (VD: 36-2024-QH15)
    filename = os.path.basename(file_path).replace(".pdf", "")
    
    # 2. Đọc và Split
    raw_text = extract_text_from_pdf(file_path)
    if not raw_text: return
    
    chunks = split_passages(raw_text, law_code=filename)
    print(f"-> Tách được {len(chunks)} đoạn (passages).")
    
    # 3. Đẩy Qdrant (Vector Search)
    update_qdrant(chunks)
    
    # 4. Đẩy HippoRAG (Graph Search)
    update_hipporag(chunks)

if __name__ == "__main__":
    # Test với một file cụ thể
    # Đảm bảo bạn đã cài poppler-utils: sudo apt install poppler-utils
    ingest_new_file("data/new_laws/ThongTu_24_2023.pdf")