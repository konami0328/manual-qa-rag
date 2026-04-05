"""
Input:  test.jsonl        — {question, answer, unique_id} one per line
        MongoDB manual_text — all chunks to build retrievers
Output: data/eval/retrieval_results.csv

Metrics: Hit@k, MRR
"""

import os
import json
import csv
from typing import List

from langchain_core.documents import Document

from config import EVAL_K_VALUES, EVAL_RETRIEVAL_PATH, TEST_PATH
from src.client.mongodb_config import MongoConfig
from src.retriever.retrieve_bm25 import BM25Retriever
from src.retriever.retrieve_bge import BGERetriever
from src.retriever.retrieve_hybrid import HybridRetriever
from src.reranker.rerank_bge import Reranker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEBUG            = True    # True → only run DEBUG_SIZE samples; False → full test set
DEBUG_SIZE       = 500
RERANKER_BATCH   = 32      # batch_size for FlagReranker.compute_score


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_test(path: str) -> List[dict]:
    """Load test.jsonl → list of {question, answer, unique_id}."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_docs() -> List[Document]:
    """Load all chunks from MongoDB."""
    col = MongoConfig.get_collection("manual_text")
    return [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in col.find()
    ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def hit_at_k(results: List[Document], ground_truth_id: str, k: int) -> int:
    """1 if ground truth unique_id appears in top-k results, else 0."""
    top_k_ids = [doc.metadata["unique_id"] for doc in results[:k]]
    return 1 if ground_truth_id in top_k_ids else 0


def reciprocal_rank(results: List[Document], ground_truth_id: str) -> float:
    """1/rank if ground truth found in results, else 0."""
    for rank, doc in enumerate(results, start=1):
        if doc.metadata["unique_id"] == ground_truth_id:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Evaluate one retriever
# ---------------------------------------------------------------------------

def evaluate_retriever(
    name: str,
    samples: List[dict],
    retrieve_fn,          # callable: query (str) -> List[Document] (already top-max(k))
    k_values: List[int],
) -> dict:
    """
    Run retrieval for all samples, compute Hit@k and MRR.

    Returns:
        {
            "name": str,
            "hit": {k: float, ...},   # averaged over all samples
            "mrr": float,
        }
    """
    max_k   = max(k_values)
    hit_sum = {k: 0 for k in k_values}
    rr_sum  = 0.0

    print(f"\n[{name}] evaluating {len(samples)} samples...")

    for i, sample in enumerate(samples):
        query          = sample["question"]
        ground_truth   = sample["source_chunk_id"]

        results = retrieve_fn(query)           # List[Document], up to max_k

        # Hit@k for each k
        for k in k_values:
            hit_sum[k] += hit_at_k(results, ground_truth, k)

        # MRR — over full max_k results
        rr_sum += reciprocal_rank(results[:max_k], ground_truth)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(samples)} done")

    n = len(samples)
    return {
        "name": name,
        "hit":  {k: round(hit_sum[k] / n, 4) for k in k_values},
        "mrr":  round(rr_sum / n, 4),
    }


# ---------------------------------------------------------------------------
# Checkpoint: save + load
# ---------------------------------------------------------------------------

def load_checkpoint(path: str) -> set:
    """Return set of retriever names already saved in CSV."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row["retriever"])
    return done


def append_csv(result: dict, k_values: List[int], path: str) -> None:
    """Append one retriever's results to CSV (write header only if file is new)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["retriever", "k", "hit@k", "mrr"])
        for k in k_values:
            writer.writerow([result["name"], k, result["hit"][k], result["mrr"]])
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------

def print_table(results: List[dict], k_values: List[int]) -> None:
    """Print results as a formatted table."""
    col_w   = 22
    k_w     = 8
    header  = f"{'Retriever':<{col_w}}" + "".join(f"Hit@{k:<{k_w-4}}" for k in k_values) + f"{'MRR':<{k_w}}"
    divider = "-" * len(header)

    print(f"\n{divider}")
    print(header)
    print(divider)
    for r in results:
        row = f"{r['name']:<{col_w}}"
        row += "".join(f"{r['hit'][k]:<{k_w}.4f}" for k in k_values)
        row += f"{r['mrr']:<{k_w}.4f}"
        print(row)
    print(divider)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    k_values = EVAL_K_VALUES
    max_k    = max(k_values)

    # --- load ---
    print("Loading test set...")
    samples = load_test(TEST_PATH)
    print(f"Test samples loaded: {len(samples)}")

    if DEBUG:
        import random
        random.seed(42)
        samples = random.sample(samples, min(DEBUG_SIZE, len(samples)))
        print(f"DEBUG mode: using {len(samples)} samples")

    print("Loading docs from MongoDB...")
    docs = load_docs()
    print(f"Docs loaded: {len(docs)}")

    # --- init retrievers ---
    print("\nInitializing retrievers...")
    bm25     = BM25Retriever(docs)
    bge      = BGERetriever(docs)
    hybrid   = HybridRetriever(docs)
    reranker = Reranker()

    # --- define retrieve_fn for each retriever ---
    # retrieve top-max(k) once; slice per k inside evaluate_retriever

    def bm25_fn(query: str)   -> List[Document]: return bm25.retrieve_topk(query, topk=max_k)
    def bge_fn(query: str)    -> List[Document]: return bge.retrieve_topk(query, topk=max_k)
    def hybrid_fn(query: str) -> List[Document]: return hybrid.retrieve(query, topk=max_k)

    def hybrid_reranker_fn(query: str) -> List[Document]:
        candidates = hybrid.retrieve(query, topk=max_k)
        reranked   = reranker.rerank(query, candidates, batch_size=RERANKER_BATCH)
        return reranked   # sorted by score; slice to k inside evaluate_retriever

    retrievers = [
        ("BM25",            bm25_fn),
        ("BGE",             bge_fn),
        ("Hybrid",          hybrid_fn),
        ("Hybrid+Reranker", hybrid_reranker_fn),
    ]

    # --- checkpoint: skip already-done retrievers ---
    done = load_checkpoint(EVAL_RETRIEVAL_PATH)
    if done:
        print(f"\nCheckpoint found — skipping: {', '.join(done)}")

    # --- evaluate ---
    all_results = []
    for name, fn in retrievers:
        if name in done:
            print(f"[{name}] already done, skipping.")
            continue
        result = evaluate_retriever(name, samples, fn, k_values)
        all_results.append(result)
        append_csv(result, k_values, EVAL_RETRIEVAL_PATH)   # save immediately after each retriever

    # --- final table: reload all rows from CSV for complete view ---
    final_results = []
    saved_names   = [name for name, _ in retrievers]   # preserve order
    rows_by_name  = {}
    with open(EVAL_RETRIEVAL_PATH, newline="") as f:
        for row in csv.DictReader(f):
            n = row["retriever"]
            if n not in rows_by_name:
                rows_by_name[n] = {"name": n, "hit": {}, "mrr": float(row["mrr"])}
            rows_by_name[n]["hit"][int(row["k"])] = float(row["hit@k"])
    for name in saved_names:
        if name in rows_by_name:
            final_results.append(rows_by_name[name])

    print_table(final_results, k_values)


if __name__ == "__main__":
    main()