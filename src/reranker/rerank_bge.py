from typing import List

from FlagEmbedding import FlagReranker
from langchain_core.documents import Document

from config import RERANKER_MODEL_PATH, RERANKER_THRESHOLD

import warnings


class Reranker:

    def __init__(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
