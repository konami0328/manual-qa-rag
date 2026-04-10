"""
BM25 keyword retriever backed by rank_bm25.BM25Okapi, persisted to disk.

Builds BM25 index from tokenized chunk texts on first run and pickles it to
BM25_PKL_FILE. Subsequent instantiations load from pickle, skipping rebuild.
Tokenization: lowercase, strip punctuation, remove English stopwords.

Args:
    docs (List[Document]) : all chunks from MongoDB "manual_text";
                            ignored if BM25_PKL_FILE already exists

Usage:
    retriever = BM25Retriever(docs)
    results   = retriever.retrieve_topk(query, topk=5)  # List[Document]
"""

import pickle
import string
from typing import List

import nltk
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from config import BM25_PKL_FILE
from pathlib import Path

nltk.download("stopwords", quiet=True)
_stopwords = set(stopwords.words("english"))

BM25_PKL_FILE = Path(BM25_PKL_FILE)

class BM25Retriever:

    def __init__(self, docs: List[Document]):
        if BM25_PKL_FILE.exists():
            self._load()
        else:
            self._build(docs)
            self._save()

    # --- public ---

    def retrieve_topk(self, query: str, topk: int = 5) -> List[Document]:
        """Return top-k Documents by BM25 score."""
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
        return [self.docs[i] for i in top_indices]

    # --- private ---

    def _build(self, docs: List[Document]) -> None:
        self.docs = docs
        tokenized = [self._tokenize(doc.page_content) for doc in docs]
        self.bm25  = BM25Okapi(tokenized)

    def _save(self) -> None:
        BM25_PKL_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BM25_PKL_FILE, "wb") as f:
            pickle.dump((self.bm25, self.docs), f)

    def _load(self) -> None:
        with open(BM25_PKL_FILE, "rb") as f:
            self.bm25, self.docs = pickle.load(f)

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase, remove punctuation and stopwords."""
        tokens = text.lower().translate(str.maketrans("", "", string.punctuation)).split()
        return [t for t in tokens if t not in _stopwords]
