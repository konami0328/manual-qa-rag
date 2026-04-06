"""
Input:  test.jsonl        — {question, answer, source_chunk_id, unique_id, page}
        MongoDB manual_text — all chunks to build retrievers
Output: data/eval/retrieval_results.csv
        data/eval/bad_cases_<retriever>_<timestamp>.txt  (DEBUG mode only)

Metrics: Hit@k, MRR
Negative samples (source_chunk_id=None) are excluded from retrieval eval.
False positive rate on negatives is only meaningful after reranker threshold
filtering and should be evaluated at the Hybrid+Reranker stage separately.
"""

import os
import json
import csv
import random
from datetime import datetime
from typing import List, Tuple

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

DEBUG          = False
DEBUG_SIZE     = 500
RERANKER_BATCH = 32
BAD_CASE_DIR   = os.path.join(os.path.dirname(EVAL_RETRIEVAL_PATH))

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_test(path: str) -> Tuple[List[dict], List[dict]]:
    """
    Load test.jsonl and split into positive and negative samples.
    Positive: source_chunk_id is not None (have ground truth chunk)
    Negative: source_chunk_id is None (MS MARCO negatives, no answer in corpus)
    """
    positives, negatives = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                sample = json.loads(line)
                if sample.get("source_chunk_id") is not None:
                    positives.append(sample)
                else:
                    negatives.append(sample)
    return positives, negatives


def load_docs() -> List[Document]:
    col = MongoConfig.get_collection("manual_text")
    return [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in col.find()
    ]


def build_chunk_lookup(docs: List[Document]) -> dict:
    return {d.metadata["unique_id"]: d for d in docs}

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def hit_at_k(results: List[Document], ground_truth_id: str, k: int) -> int:
    top_k_ids = [doc.metadata["unique_id"] for doc in results[:k]]
    return 1 if ground_truth_id in top_k_ids else 0


def reciprocal_rank(results: List[Document], ground_truth_id: str) -> float:
    for rank, doc in enumerate(results, start=1):
        if doc.metadata["unique_id"] == ground_truth_id:
            return 1.0 / rank
    return 0.0

# ---------------------------------------------------------------------------
# Bad case writer
# ---------------------------------------------------------------------------

def wc(text: str) -> int:
    return len(text.split())


def open_bad_case_file(name: str, timestamp: str):
    """Open (or create) the bad case txt file for a retriever, return file handle."""
    os.makedirs(BAD_CASE_DIR, exist_ok=True)
    path = os.path.join(BAD_CASE_DIR, f"bad_cases_{name}_{timestamp}.txt")
    f = open(path, "a", encoding="utf-8")
    f.write(f"BAD CASES — {name}\n")
    f.write(f"Generated: {timestamp}\n")
    f.write("═" * 80 + "\n\n")
    return f, path


def write_bad_case(f, idx: int, sample: dict, gt_doc: Document, results: List[Document], max_k: int):
    """Write one bad case entry and flush immediately."""
    gt_text = gt_doc.page_content if gt_doc else "[chunk not found in MongoDB]"
    gt_page = gt_doc.metadata.get("page", "?") if gt_doc else "?"
    gt_wc   = wc(gt_text) if gt_doc else 0

    f.write(f"[{idx}]  page={sample['page']}  source_chunk_id={sample['source_chunk_id']}\n")
    f.write("─" * 80 + "\n")
    f.write(f"QUESTION:\n  {sample['question']}\n\n")
    f.write(f"ANSWER:\n  {sample['answer']}\n\n")
    f.write(f"GROUND TRUTH CHUNK  (page={gt_page}  words={gt_wc}):\n")
    f.write(f"  {gt_text}\n\n")
    f.write(f"TOP-{max_k} RETRIEVED:\n")

    for rank, doc in enumerate(results, 1):
        uid      = doc.metadata.get("unique_id", "?")
        page     = doc.metadata.get("page", "?")
        content  = doc.page_content
        words    = wc(content)
        f.write(f"\n  [{rank}] uid={uid}  page={page}  words={words}\n")
        f.write(f"  {content}\n")

    f.write("\n" + "═" * 80 + "\n\n")
    f.flush()

# ---------------------------------------------------------------------------
# Evaluate one retriever
# ---------------------------------------------------------------------------

def evaluate_retriever(
    name: str,
    samples: List[dict],
    retrieve_fn,
    k_values: List[int],
    chunk_lookup: dict,
    timestamp: str,
) -> dict:
    max_k   = max(k_values)
    hit_sum = {k: 0 for k in k_values}
    rr_sum  = 0.0
    miss_count = 0

    bad_case_path = None
    if DEBUG:
        bad_f, bad_case_path = open_bad_case_file(name, timestamp)

    print(f"\n[{name}] evaluating {len(samples)} samples...")

    for i, sample in enumerate(samples):
        query        = sample["question"]
        ground_truth = sample["source_chunk_id"]

        results = retrieve_fn(query)

        for k in k_values:
            hit_sum[k] += hit_at_k(results, ground_truth, k)

        rr_sum += reciprocal_rank(results[:max_k], ground_truth)

        # bad case: miss at max_k
        if DEBUG and hit_at_k(results, ground_truth, max_k) == 0:
            miss_count += 1
            gt_doc = chunk_lookup.get(ground_truth)
            write_bad_case(bad_f, miss_count, sample, gt_doc, results, max_k)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(samples)} done")

    if DEBUG:
        bad_f.close()
        print(f"  Bad cases ({miss_count}) → {bad_case_path}")

    n = len(samples)
    return {
        "name": name,
        "hit":  {k: round(hit_sum[k] / n, 4) for k in k_values},
        "mrr":  round(rr_sum / n, 4),
    }

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row["retriever"])
    return done


def append_csv(result: dict, k_values: List[int], path: str) -> None:
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
    col_w   = 22
    k_w     = 8
    header  = f"{'Retriever':<{col_w}}" + "".join(f"Hit@{k:<{k_w-4}}" for k in k_values) + f"{'MRR':<{k_w}}"
    divider = "-" * len(header)
    print(f"\n{divider}")
    print(header)
    print(divider)
    for r in results:
        row  = f"{r['name']:<{col_w}}"
        row += "".join(f"{r['hit'][k]:<{k_w}.4f}" for k in k_values)
        row += f"{r['mrr']:<{k_w}.4f}"
        print(row)
    print(divider)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    k_values  = EVAL_K_VALUES
    max_k     = max(k_values)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Loading test set...")
    positives, negatives = load_test(TEST_PATH)
    print(f"Positive samples: {len(positives)}  Negative samples: {len(negatives)}")

    samples = positives
    if DEBUG:
        random.seed(42)
        samples = random.sample(samples, min(DEBUG_SIZE, len(samples)))
        print(f"DEBUG mode: using {len(samples)} positive samples")

    print("Loading docs from MongoDB...")
    docs         = load_docs()
    chunk_lookup = build_chunk_lookup(docs)
    print(f"Docs loaded: {len(docs)}")

    print("\nInitializing retrievers...")
    bm25     = BM25Retriever(docs)
    bge      = BGERetriever(docs)
    hybrid   = HybridRetriever(docs)
    reranker = Reranker()

    def bm25_fn(q): return bm25.retrieve_topk(q, topk=max_k)
    def bge_fn(q):  return bge.retrieve_topk(q, topk=max_k)
    def hybrid_reranker_fn(q):
        candidates = hybrid.retrieve(q, topk=max_k)
        return reranker.rerank(q, candidates, batch_size=RERANKER_BATCH)

    retrievers = [
        ("BM25",            bm25_fn),
        ("BGE",             bge_fn),
        ("Hybrid+Reranker", hybrid_reranker_fn),
    ]

    done = load_checkpoint(EVAL_RETRIEVAL_PATH)
    if done:
        print(f"\nCheckpoint found — skipping: {', '.join(done)}")

    all_results = []
    for name, fn in retrievers:
        if name in done:
            print(f"[{name}] already done, skipping.")
            continue
        result = evaluate_retriever(name, samples, fn, k_values, chunk_lookup, timestamp)
        append_csv(result, k_values, EVAL_RETRIEVAL_PATH)
        all_results.append(result)

    print_table(all_results, k_values)


if __name__ == "__main__":
    main()