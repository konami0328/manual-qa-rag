"""
Off-the-shelf cross-encoder reranker using bge-reranker-v2-m3.

Scores each (query, doc) pair, filters by RERANKER_THRESHOLD, and returns
docs sorted by score descending. Scores are normalized to [0, 1] and stored
in doc.metadata["rerank_score"] for downstream threshold calibration.

Args:
    None — model path and threshold loaded from config.py

Usage:
    reranker = Reranker()
    results  = reranker.rerank(query, docs, batch_size=32)  # List[Document]
"""

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
