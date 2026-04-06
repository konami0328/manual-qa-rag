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


if __name__ == "__main__":
    from langchain_core.documents import Document
    from src.client.mongodb_config import MongoConfig

    col  = MongoConfig.get_collection("manual_text")
    docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in col.find()]

    retriever = HybridRetriever(docs)
    results   = retriever.retrieve("How to Adjust the Shoulder Anchor Height", topk=10)

    print(f"Total candidates: {len(results)}\n")
    for r in results:
        print(f"Page: {r.metadata.get('page')}")
        print(r.page_content[:100], "...")
        print("=" * 60)