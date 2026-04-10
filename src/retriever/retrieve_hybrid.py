"""
Hybrid retriever combining BGE dense+sparse RRF and BM25 via union dedup.

BGE results are ranked by RRF internally; BM25 serves as a safety net for
exact keyword matches that BGE sparse may miss. Final candidate pool preserves
BGE rank order with BM25-only results appended — not relevance-sorted.

Args:
    docs          (List[Document]) : all chunks from MongoDB "manual_text"
    force_rebuild (bool)           : passed through to BGERetriever

Usage:
    retriever = HybridRetriever(docs, force_rebuild=False)
    results   = retriever.retrieve(query, topk=10)  # List[Document]
"""

from typing import List

from langchain_core.documents import Document

from src.retriever.retrieve_bm25 import BM25Retriever
from src.retriever.retrieve_bge import BGERetriever


class HybridRetriever:

    def __init__(self, docs: List[Document], force_rebuild: bool = False):
        self.bm25 = BM25Retriever(docs)
        self.bge  = BGERetriever(docs, force_rebuild=force_rebuild)

    def retrieve(self, query: str, topk: int = 10) -> List[Document]:
        """
        1. BGE dense+sparse → RRF (internal) → top-k
        2. BM25 → top-k
        3. Dedup + union (BGE first, BM25 appended)
        """
        bge_results  = self.bge.retrieve_topk(query, topk)
        bm25_results = self.bm25.retrieve_topk(query, topk)

        # dedup
        seen = set()
        candidates = []
        for doc in bge_results + bm25_results:
            uid = doc.metadata["unique_id"]
            if uid not in seen:
                seen.add(uid)
                candidates.append(doc)

        return candidates
