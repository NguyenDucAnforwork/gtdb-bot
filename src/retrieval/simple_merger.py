# src/retrieval/simple_merger.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Any, Optional, Callable

class SimpleMergerRetriever:
    """
    Gộp kết quả từ nhiều retriever chạy song song.
    Mỗi retriever cần có hàm:
      - get_relevant_documents(query) -> List[Document-like]
    Hoặc hỗ trợ .invoke(query) trả về List[Document-like].
    """

    def __init__(
        self,
        retrievers: List[Any],
        max_workers: Optional[int] = None,
        dedupe_key: Optional[Callable[[Any], str]] = None,
        top_k: Optional[int] = None,
    ):
        self.retrievers = retrievers
        self.max_workers = max_workers
        # dedupe theo text nếu có, tuỳ bạn — mặc định dùng getattr(doc, "page_content", repr(doc))
        self.dedupe_key = dedupe_key or (lambda d: getattr(d, "page_content", repr(d)))
        self.top_k = top_k  # nếu muốn cắt bớt kết quả cuối

    def _call_one(self, r, query):
        # Ưu tiên API của LangChain retriever
        if hasattr(r, "get_relevant_documents"):
            return r.get_relevant_documents(query)
        # Hỗ trợ Runnable .invoke
        if hasattr(r, "invoke"):
            return r.invoke(query)
        # Hỗ trợ callable tuỳ biến
        if callable(r):
            return r(query)
        raise TypeError(f"Retriever {r} không có phương thức thích hợp")

    def _merge(self, batches: List[List[Any]]) -> List[Any]:
        merged = []
        seen = set()
        for batch in batches:
            for item in batch or []:
                key = self.dedupe_key(item)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)
        if self.top_k is not None:
            merged = merged[: self.top_k]
        return merged

    def get_relevant_documents(self, query: str) -> List[Any]:
        # chạy song song
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(self._call_one, r, query) for r in self.retrievers]
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    # tuỳ ý: log/ignore
                    # print(f"[SimpleMergerRetriever] retriever error: {e}")
                    results.append([])
        return self._merge(results)

    # để tương thích với LCEL/Runnable
    def invoke(self, query: str) -> List[Any]:
        return self.get_relevant_documents(query)
