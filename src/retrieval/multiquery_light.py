# src/retrieval/multiquery_light.py
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Callable, Iterable, List, Optional

# ---- Helpers ---------------------------------------------------------------

def _doc_text(d: Any) -> str:
    # Hỗ trợ cả LangChain Document và dict
    if hasattr(d, "page_content"):
        return getattr(d, "page_content")
    if isinstance(d, dict):
        # thử vài tên phổ biến
        for k in ("text", "content", "page_content", "snippet", "body"):
            if k in d and isinstance(d[k], str):
                return d[k]
    return repr(d)

def _doc_id(d: Any) -> str:
    # ID để dedupe: ưu tiên metadata/id, fallback theo text
    if hasattr(d, "metadata") and isinstance(d.metadata, dict):
        if "id" in d.metadata:
            return f"id:{d.metadata['id']}"
        if "source" in d.metadata and "chunk_id" in d.metadata:
            return f"src:{d.metadata['source']}#chunk:{d.metadata['chunk_id']}"
    if isinstance(d, dict):
        for k in ("id", "doc_id", "uuid"):
            if k in d:
                return f"id:{d[k]}"
        if "source" in d and "chunk_id" in d:
            return f"src:{d['source']}#chunk:{d['chunk_id']}"
    # cuối cùng dedupe theo text (cắt ngắn để key gọn)
    return f"txt:{_doc_text(d)[:256]}"

def _call_llm_to_queries(llm: Any, user_query: str, k: int, prompt_template: Optional[str]) -> List[str]:
    """
    Gọi LLM sinh k truy vấn khác góc nhìn. Chấp nhận:
    - llm.invoke(prompt) -> object có .content hoặc là string
    - llm(prompt)        -> string (nhiều lib hỗ trợ __call__)
    Kỳ vọng LLM trả về các dòng (mỗi dòng 1 truy vấn) hoặc dạng bullet/numbered list.
    """
    prompt = (prompt_template or
              "Bạn là công cụ tạo truy vấn tìm kiếm. Hãy viết {k} biến thể truy vấn khác góc nhìn, "
              "ngắn gọn, mỗi truy vấn trên một dòng, không giải thích.\n\nTruy vấn gốc: {q}\n")
    p = prompt.format(k=k, q=user_query)

    # Gọi theo các “hình” API phổ biến
    out = None
    if hasattr(llm, "invoke"):
        out = llm.invoke(p)
        out = getattr(out, "content", out)
    elif callable(llm):
        out = llm(p)
    else:
        raise TypeError("LLM không hỗ trợ invoke() hoặc __call__().")

    if out is None:
        return [user_query]

    if not isinstance(out, str):
        out = str(out)

    # tách dòng, bỏ bullet số thứ tự
    lines = []
    for raw in out.splitlines():
        s = raw.strip().lstrip("-•*").lstrip()
        # bỏ prefix số thứ tự kiểu "1. " "2) " "[a]"
        for prefix in (". ", ") ", "] "):
            if len(s) > 2 and s[0].isalnum() and s[1] in ".])":
                s = s[2:].lstrip()
                break
        if s:
            lines.append(s)

    # fallback nếu LLM trả ít dòng
    if len(lines) < k:
        lines = list(dict.fromkeys(lines))  # unique, giữ thứ tự
        base = user_query.strip()
        aug = [
            f"{base} tổng quan",
            f"{base} định nghĩa và bối cảnh",
            f"{base} ví dụ và trường hợp biên",
            f"{base} thuật ngữ liên quan",
            f"{base} từ khóa tương đương",
        ]
        lines.extend(aug[: max(0, k - len(lines))])

    # unique & cắt đúng k
    uniq = []
    seen = set()
    for s in lines:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq[:k]

# ---- Main Retriever --------------------------------------------------------

class MultiQueryLight:
    """
    Sinh nhiều truy vấn bằng LLM rồi truy hồi song song qua một retriever gốc.
    - retriever: có get_relevant_documents(query) hoặc invoke(query) -> List[Document-like]
    - llm: có invoke(prompt) hoặc __call__(prompt) -> (str | object with .content)
    """
    def __init__(
        self,
        retriever: Any,
        llm: Any,
        k_queries: int = 5,
        max_workers: Optional[int] = None,
        prompt_template: Optional[str] = None,
        top_k: Optional[int] = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.k_queries = k_queries
        self.max_workers = max_workers
        self.prompt_template = prompt_template
        self.top_k = top_k

    def _retrieve_once(self, q: str) -> List[Any]:
        r = self.retriever
        if hasattr(r, "get_relevant_documents"):
            return r.get_relevant_documents(q)
        if hasattr(r, "invoke"):
            return r.invoke(q)
        if callable(r):
            return r(q)
        raise TypeError("Retriever không có phương thức phù hợp.")

    def _merge(self, batches: Iterable[List[Any]]) -> List[Any]:
        merged = []
        seen = set()
        for batch in batches:
            for d in batch or []:
                key = _doc_id(d)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(d)
        if self.top_k is not None:
            merged = merged[: self.top_k]
        return merged

    @lru_cache(maxsize=256)
    def _gen_queries_cached(self, user_query: str, k: int, prompt_key: str) -> tuple:
        qs = _call_llm_to_queries(self.llm, user_query, k, self.prompt_template)
        # trả tuple để lru_cache hashable
        return tuple(qs)

    def _gen_queries(self, user_query: str) -> List[str]:
        prompt_key = self.prompt_template or ""
        return list(self._gen_queries_cached(user_query, self.k_queries, prompt_key))

    def get_relevant_documents(self, query: str) -> List[Any]:
        qs = self._gen_queries(query)
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(self._retrieve_once, q) for q in qs]
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception:
                    results.append([])
        return self._merge(results)

    # tương thích LCEL Runnable
    def invoke(self, query: str) -> List[Any]:
        return self.get_relevant_documents(query)
