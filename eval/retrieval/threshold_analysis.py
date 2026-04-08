"""
Collect reranker scores for positive and negative val samples, save to CSV.
All analysis (percentiles, plots, threshold sweep) is done separately in threshold_analysis.py.

Output:
    eval/retrieval/results/threshold_gt_scores_<timestamp>.csv
    columns: type, question, source_chunk_id, score, retrieval_miss
      - positive rows: gt_score of ground truth chunk (0.0 if retrieval miss)
      - negative rows: max score among all candidates (strongest false positive signal)
"""

import os
import json
import csv
import random
from datetime import datetime

from langchain_core.documents import Document

from config import VAL_PATH, EVAL_RETRIEVAL_PATH
from src.client.mongodb_config import MongoConfig
from src.retriever.retrieve_hybrid import HybridRetriever
from src.reranker.rerank_bge_finetuned import FinetunedReranker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEBUG          = False
DEBUG_SIZE     = 100
RETRIEVE_TOPK  = 10
RERANKER_BATCH = 32
OUTPUT_DIR     = os.path.join(os.path.dirname(EVAL_RETRIEVAL_PATH))

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_val_samples(path: str) -> tuple[list[dict], list[dict]]:
    """Load val set, split into positives and negatives."""
    positives, negatives = [], []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            if item.get("source_chunk_id") is not None:
                positives.append(item)
            else:
                negatives.append(item)
    return positives, negatives


def load_docs() -> list[Document]:
    col = MongoConfig.get_collection("manual_text")
    return [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in col.find()
    ]

# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------

def collect_positive_scores(
    samples: list[dict],
    retriever: HybridRetriever,
    reranker: FinetunedReranker,
) -> list[dict]:
    """
    For each positive query, record gt_score and all_scores.
    gt_score = 0.0 if GT chunk not in top-RETRIEVE_TOPK (retrieval miss).
    all_scores = all candidate scores after reranking (needed for avg-candidates analysis).
    """
    records = []
    for i, sample in enumerate(samples):
        query = sample["question"]
        gt_id = sample["source_chunk_id"]

        candidates    = retriever.retrieve(query, topk=RETRIEVE_TOPK)
        ranked        = reranker.rerank(query, candidates, batch_size=RERANKER_BATCH)
        all_scores    = [doc.metadata["rerank_score"] for doc in ranked]
        candidate_ids = [doc.metadata["unique_id"]    for doc in ranked]

        retrieval_miss = gt_id not in candidate_ids
        if retrieval_miss:
            gt_score = 0.0
        else:
            idx      = candidate_ids.index(gt_id)
            gt_score = ranked[idx].metadata["rerank_score"]

        records.append({
            "question":        query,
            "source_chunk_id": gt_id,
            "gt_score":        round(gt_score, 4),
            "all_scores":      all_scores,
            "retrieval_miss":  retrieval_miss,
        })

        if (i + 1) % 50 == 0:
            print(f"  [pos] {i+1}/{len(samples)} done")

    return records


def collect_negative_scores(
    samples: list[dict],
    retriever: HybridRetriever,
    reranker: FinetunedReranker,
) -> list[float]:
    """
    For each negative query, record the max reranker score among all candidates.
    Represents the strongest false positive signal for that query.
    """
    max_scores = []
    for i, sample in enumerate(samples):
        query      = sample["question"]
        candidates = retriever.retrieve(query, topk=RETRIEVE_TOPK)
        ranked     = reranker.rerank(query, candidates, batch_size=RERANKER_BATCH)

        max_scores.append(ranked[0].metadata["rerank_score"] if ranked else 0.0)

        if (i + 1) % 50 == 0:
            print(f"  [neg] {i+1}/{len(samples)} done")

    return max_scores

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_csv(
    pos_records: list[dict],
    neg_max_scores: list[float],
    output_dir: str,
    timestamp: str,
):
    path = os.path.join(output_dir, f"threshold_gt_scores_{timestamp}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "question", "source_chunk_id", "score", "retrieval_miss", "all_scores"])
        for r in pos_records:
            writer.writerow(["positive", r["question"], r["source_chunk_id"], r["gt_score"], r["retrieval_miss"], r["all_scores"]])
        for s in neg_max_scores:
            writer.writerow(["negative", "", "", s, "na", ""])
    print(f"Saved → {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading val set...")
    positives, negatives = load_val_samples(VAL_PATH)
    print(f"Positive samples: {len(positives)}  Negative samples: {len(negatives)}")

    if DEBUG:
        random.seed(42)
        positives = random.sample(positives, min(DEBUG_SIZE, len(positives)))
        negatives = random.sample(negatives, min(DEBUG_SIZE, len(negatives)))
        print(f"DEBUG mode: {len(positives)} pos / {len(negatives)} neg")

    print("Loading docs from MongoDB...")
    docs = load_docs()
    print(f"Docs: {len(docs)}")

    print("Initializing retriever and reranker...")
    retriever = HybridRetriever(docs)
    reranker  = FinetunedReranker()

    print("\nCollecting positive GT scores...")
    pos_records = collect_positive_scores(positives, retriever, reranker)

    print("\nCollecting negative max scores...")
    neg_max_scores = collect_negative_scores(negatives, retriever, reranker)

    save_csv(pos_records, neg_max_scores, OUTPUT_DIR, timestamp)
    print("Done.")


if __name__ == "__main__":
    main()