"""
Run the full inference pipeline on the test set and save raw results for evaluation.

Pipeline:
    For each sample in test set:
        positives → HybridRetriever → FinetunedReranker → LLM → save with retrieval_hit flag
        negatives → HybridRetriever → FinetunedReranker → LLM → save (no ground truth)

Output:
    eval/generate/results/generation_raw_<timestamp>.jsonl
    one record per sample:
    {
        "sample_type":     "positive" | "negative",
        "question":        str,
        "answer":          str,           # LLM output
        "ground_truth":    str | null,    # positive only
        "source_chunk_id": str | null,    # positive only
        "retrieval_hit":   bool | null,   # positive only
        "contexts":        list[str] | null  # null if no chunks passed threshold
    }
"""

import os
import json
import random
from datetime import datetime

from langchain_core.documents import Document

from config import TEST_PATH, TOPK, GENERATION_TOPK, GENERATION_THRESHOLD
from src.client.mongodb_config import MongoConfig
from src.retriever.retrieve_hybrid import HybridRetriever
from src.reranker.rerank_bge_finetuned import FinetunedReranker
from src.client.llm_generate import request_chat

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEBUG      = True
DEBUG_SIZE = 50
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eval", "generate", "results")
NO_ANSWER  = "This information is not covered in the provided context."

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_test(path: str) -> tuple[list[dict], list[dict]]:
    positives, negatives = [], []
    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get("source_chunk_id") is not None:
                positives.append(item)
            else:
                negatives.append(item)
    return positives, negatives


def load_docs() -> list[Document]:
    col = MongoConfig.get_collection("manual_text")
    return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in col.find()]

# ---------------------------------------------------------------------------
# Infer
# ---------------------------------------------------------------------------

def infer(query: str, retriever: HybridRetriever, reranker: FinetunedReranker) -> tuple[str, list[Document] | None]:
    candidates = retriever.retrieve(query, topk=TOPK)
    ranked     = reranker.rerank(query, candidates)
    chunks     = [c for c in ranked if c.metadata["rerank_score"] >= GENERATION_THRESHOLD][:GENERATION_TOPK]

    if not chunks:
        return NO_ANSWER, None

    context = "\n".join(
        f"[{i+1}] (Page {doc.metadata.get('page', '?')}) {doc.page_content}"
        for i, doc in enumerate(chunks)
    )
    return request_chat(query, context), chunks

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"generation_raw_{timestamp}.jsonl")

    print("Loading test set...")
    positives, negatives = load_test(TEST_PATH)
    print(f"Positives: {len(positives)}  Negatives: {len(negatives)}")

    if DEBUG:
        random.seed(42)
        positives = random.sample(positives, min(DEBUG_SIZE, len(positives)))
        negatives = random.sample(negatives, min(DEBUG_SIZE, len(negatives)))
        print(f"DEBUG: {len(positives)} pos / {len(negatives)} neg")

    print("Loading docs...")
    docs      = load_docs()
    retriever = HybridRetriever(docs)
    reranker  = FinetunedReranker()

    records = []

    print("\nInferring positives...")
    for i, sample in enumerate(positives):
        answer, chunks = infer(sample["question"], retriever, reranker)
        chunk_ids      = [c.metadata["unique_id"] for c in chunks] if chunks else []
        retrieval_hit  = sample["source_chunk_id"] in chunk_ids

        records.append({
            "sample_type":     "positive",
            "question":        sample["question"],
            "answer":          answer,
            "ground_truth":    sample["answer"],
            "source_chunk_id": sample["source_chunk_id"],
            "retrieval_hit":   retrieval_hit,
            "contexts":        [c.page_content for c in chunks] if chunks else None,
        })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(positives)}")

    print("\nInferring negatives...")
    for i, sample in enumerate(negatives):
        answer, chunks = infer(sample["question"], retriever, reranker)

        records.append({
            "sample_type":     "negative",
            "question":        sample["question"],
            "answer":          answer,
            "ground_truth":    None,
            "source_chunk_id": None,
            "retrieval_hit":   None,
            "contexts":        [c.page_content for c in chunks] if chunks else None,
        })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(negatives)}")

    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pos_hit  = sum(1 for r in records if r["sample_type"] == "positive" and r["retrieval_hit"])
    pos_miss = sum(1 for r in records if r["sample_type"] == "positive" and not r["retrieval_hit"])
    print(f"\nDone.")
    print(f"  Positive hit : {pos_hit}")
    print(f"  Positive miss: {pos_miss}")
    print(f"  Negatives    : {len(negatives)}")
    print(f"  Saved → {output_path}")


if __name__ == "__main__":
    main()