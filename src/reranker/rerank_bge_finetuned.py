from typing import List

import torch
from langchain_core.documents import Document
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import RERANKER_MODEL_PATH, RERANKER_BEST_CKPT

MAX_LENGTH = 768  # covers 99% chunk length (686 tokens) + query + special tokens


class FinetunedReranker:

    def __init__(self):
        base = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL_PATH,
            num_labels=1,
            torch_dtype=torch.bfloat16,  # bfloat16 has same exponent range as float32, avoids fp16 overflow
        )
        model = PeftModel.from_pretrained(base, RERANKER_BEST_CKPT)
        self._model     = model.merge_and_unload().eval().cuda()
        self._tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_PATH)

    def rerank(self, query: str, docs: List[Document], batch_size: int = 32) -> List[Document]:
        """Score each (query, doc) pair, sort by score descending."""
        results = []
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i : i + batch_size]
            encoded = self._tokenizer(
                [query] * len(batch_docs),
                [d.page_content for d in batch_docs],
                max_length=MAX_LENGTH,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            with torch.no_grad():
                logits = self._model(**encoded).logits.squeeze(-1).float().cpu().tolist()
            for doc, score in zip(batch_docs, logits):
                doc.metadata["rerank_score"] = round(score, 4)
                results.append(doc)
        return sorted(results, key=lambda d: d.metadata["rerank_score"], reverse=True)


if __name__ == "__main__":
    from langchain_core.documents import Document
    from src.client.mongodb_config import MongoConfig
    from src.retriever.retrieve_hybrid import HybridRetriever

    col  = MongoConfig.get_collection("manual_text")
    docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in col.find()]

    query     = "How to Adjust the Shoulder Anchor Height"
    retriever = HybridRetriever(docs)
    reranker  = FinetunedReranker()

    candidates = retriever.retrieve(query, topk=10)
    results    = reranker.rerank(query, candidates)

    print(f"Candidates: {len(candidates)} → After rerank: {len(results)}\n")
    for r in results:
        print(f"Page: {r.metadata.get('page')} | Score: {r.metadata.get('rerank_score')}")
        print(r.page_content[:200], "...")
        print("=" * 60)