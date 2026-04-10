"""
Run the full inference pipeline on the test set and save raw results for evaluation.
Uses vLLM server via OpenAI-compatible API with ThreadPoolExecutor for concurrency.

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
        "answer":          str,
        "ground_truth":    str | null,
        "source_chunk_id": str | null,
        "retrieval_hit":   bool | null,
        "contexts":        list[str] | null
    }
"""

import os
import json
import random
import threading
import concurrent.futures
from datetime import datetime

from langchain_core.documents import Document

from config import TEST_PATH, TOPK, GENERATION_TOPK, GENERATION_THRESHOLD, VLLM_MAX_WORKERS
from src.client.mongodb_config import MongoConfig
from src.retriever.retrieve_hybrid import HybridRetriever
from src.reranker.rerank_bge_finetuned import FinetunedReranker
from src.client.llm_generate_vllm import request_chat

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEBUG      = False
DEBUG_SIZE = 50
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")
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

def _build_context(chunks: list[Document]) -> str:
    return "\n".join(
        f"[{i+1}] (Page {doc.metadata.get('page', '?')}) {doc.page_content}"
        for i, doc in enumerate(chunks)
    )

_reranker_lock = threading.Lock()

def _retrieve_and_rank(
    query: str,
    retriever: HybridRetriever,
    reranker: FinetunedReranker,
) -> list[Document] | None:
    candidates = retriever.retrieve(query, topk=TOPK)
    with _reranker_lock:
        ranked = reranker.rerank(query, candidates)
    chunks = [c for c in ranked if c.metadata["rerank_score"] >= GENERATION_THRESHOLD][:GENERATION_TOPK]
    return chunks if chunks else None

# ---------------------------------------------------------------------------
# Process samples
# ---------------------------------------------------------------------------

def _process(
    sample: dict,
    retriever: HybridRetriever,
    reranker: FinetunedReranker,
) -> dict:
    query  = sample["question"]
    chunks = _retrieve_and_rank(query, retriever, reranker)

    if chunks is None:
        answer = NO_ANSWER
    else:
        answer = request_chat(query, _build_context(chunks))

    if sample.get("source_chunk_id") is not None:
        chunk_ids     = [c.metadata["unique_id"] for c in chunks] if chunks else []
        retrieval_hit = sample["source_chunk_id"] in chunk_ids
        return {
            "sample_type":     "positive",
            "question":        query,
            "answer":          answer,
            "ground_truth":    sample["answer"],
            "source_chunk_id": sample["source_chunk_id"],
            "retrieval_hit":   retrieval_hit,
            "contexts":        [c.page_content for c in chunks] if chunks else None,
        }
    else:
        return {
            "sample_type":     "negative",
            "question":        query,
            "answer":          answer,
            "ground_truth":    None,
            "source_chunk_id": None,
            "retrieval_hit":   None,
            "contexts":        [c.page_content for c in chunks] if chunks else None,
        }

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

    all_samples = positives + negatives
    records     = [None] * len(all_samples)
    file_lock   = threading.Lock()
    done_count  = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=VLLM_MAX_WORKERS) as executor:
        futures = {
            executor.submit(_process, sample, retriever, reranker): i
            for i, sample in enumerate(all_samples)
        }
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            try:
                records[i] = future.result()
            except Exception as e:
                print(f"ERROR on sample {i}: {e}")
                records[i] = None
            with file_lock:
                done_count += 1
                if done_count % 10 == 0 or done_count == len(all_samples):
                    print(f"  {done_count}/{len(all_samples)}")

    valid_records = [r for r in records if r is not None]

    with open(output_path, "w") as f:
        for r in valid_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pos_hit  = sum(1 for r in valid_records if r["sample_type"] == "positive" and r["retrieval_hit"])
    pos_miss = sum(1 for r in valid_records if r["sample_type"] == "positive" and not r["retrieval_hit"])
    print(f"\nDone.")
    print(f"  Positive hit : {pos_hit}")
    print(f"  Positive miss: {pos_miss}")
    print(f"  Negatives    : {len(negatives)}")
    print(f"  Saved → {output_path}")


if __name__ == "__main__":
    main()