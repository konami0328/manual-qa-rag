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


if __name__ == "__main__":
    from src.client.mongodb_config import MongoConfig
    col     = MongoConfig.get_collection("manual_text")
    docs    = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in col.find()]
    bm25    = BM25Retriever(docs)
    results = bm25.retrieve_topk("How to Adjust the Shoulder Anchor Height", topk=3)
    for r in results:
        print(r.page_content[:200])
        print("=" * 60)