from typing import List

from FlagEmbedding import FlagReranker
from langchain_core.documents import Document

from config import RERANKER_MODEL_PATH, RERANKER_THRESHOLD


class Reranker:

    def __init__(self):
        self._model = FlagReranker(RERANKER_MODEL_PATH, use_fp16=True)

    def rerank(self, query: str, docs: List[Document], batch_size: int = 32) -> List[Document]:
        """Score each (query, doc) pair, filter by threshold, sort by score descending."""
        pairs  = [(query, doc.page_content) for doc in docs]
        scores = self._model.compute_score(pairs, normalize=True, batch_size=batch_size)

        results = []
        for doc, score in zip(docs, scores):
            if score >= RERANKER_THRESHOLD:
                doc.metadata["rerank_score"] = round(score, 4)
                results.append(doc)

        return sorted(results, key=lambda d: d.metadata["rerank_score"], reverse=True)


if __name__ == "__main__":
    from langchain_core.documents import Document
    from src.client.mongodb_config import MongoConfig
    from src.retriever.retrieve_hybrid import HybridRetriever

    col  = MongoConfig.get_collection("manual_text")
    docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in col.find()]

    query      = "How to Adjust the Shoulder Anchor Height"
    retriever  = HybridRetriever(docs)
    reranker   = Reranker()

    candidates = retriever.retrieve(query, topk=10)
    results    = reranker.rerank(query, candidates)

    print(f"Candidates: {len(candidates)} → After rerank: {len(results)}\n")
    for r in results:
        print(f"Page: {r.metadata.get('page')} | Score: {r.metadata.get('rerank_score')}")
        print(r.page_content[:200], "...")
        print("=" * 60)