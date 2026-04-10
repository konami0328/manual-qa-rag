"""
Fine-tuned cross-encoder reranker using bge-reranker-v2-m3 with LoRA weights merged.

Loads base model + PEFT checkpoint, merges LoRA weights at init, runs in eval
mode on CUDA. Outputs raw logits (not normalized) stored in
doc.metadata["rerank_score"]. Threshold calibration required before production use.

Args:
    None — model path and checkpoint loaded from config.py

Usage:
    reranker = FinetunedReranker()
    results  = reranker.rerank(query, docs, batch_size=32)  # List[Document]
"""

from typing import List

import torch
import warnings
from langchain_core.documents import Document
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

from config import RERANKER_MODEL_PATH, RERANKER_BEST_CKPT

MAX_LENGTH = 768  # covers P99 chunk length (686 tokens) + query + special tokens


class FinetunedReranker:

    def __init__(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
