"""
Goal: Find a safe threshold value for filtering reranker candidates before generation.

Approach:
    Positive samples (source_chunk_id is not None):
        For each query, retrieve top-RETRIEVE_TOPK candidates, rerank, record the
        score of the ground truth chunk. If GT not in top-RETRIEVE_TOPK → score = 0.0.

    Negative samples (source_chunk_id is None, e.g. MS MARCO queries):
        For each query, retrieve + rerank, record the MAX score among all candidates.
        There is no GT chunk; max score represents the strongest false positive signal.

    Plot both distributions on the same histogram to visualize separation.
    A good threshold sits in the gap between the two distributions.

    Additionally sweep threshold values to compute:
        - GT survival rate (positive samples): % of queries where gt_score >= threshold
        - False positive rate (negative samples): % of queries where max_score >= threshold
        - Avg surviving candidates per query (positive samples): proxy for LLM context size

Output:
    - Printed percentile table + threshold sweep table
    - threshold_analysis_<timestamp>.png  (two subplots)
    - threshold_gt_scores_<timestamp>.csv (per-query scores)
"""

import os
import json
import csv
import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from langchain_core.documents import Document

from config import VAL_PATH, EVAL_RETRIEVAL_PATH
from src.client.mongodb_config import MongoConfig
from src.retriever.retrieve_hybrid import HybridRetriever
from src.reranker.rerank_bge_finetuned import FinetunedReranker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEBUG           = False
DEBUG_SIZE      = 100
RETRIEVE_TOPK   = 10
THRESHOLD_SWEEP = [round(x, 2) for x in np.arange(0.0, 0.8, 0.05)]
OUTPUT_DIR      = os.path.join(os.path.dirname(EVAL_RETRIEVAL_PATH))
RERANKER_BATCH  = 32

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
# Core analysis
# ---------------------------------------------------------------------------

def collect_positive_scores(
    samples: list[dict],
    retriever: HybridRetriever,
    reranker: FinetunedReranker,
) -> list[dict]:
    """
    For each positive query, record:
      - gt_score: reranker score of GT chunk (0.0 if not retrieved)
      - all_scores: all candidate scores (for avg-candidates computation)
      - retrieval_miss: True if GT was not in top-RETRIEVE_TOPK
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
    For each negative query, record the MAX reranker score among all candidates.
    This represents the strongest false positive signal for that query.
    """
    max_scores = []
    for i, sample in enumerate(samples):
        query      = sample["question"]
        candidates = retriever.retrieve(query, topk=RETRIEVE_TOPK)
        ranked     = reranker.rerank(query, candidates, batch_size=RERANKER_BATCH)

        if ranked:
            max_scores.append(ranked[0].metadata["rerank_score"])  # already sorted desc
        else:
            max_scores.append(0.0)

        if (i + 1) % 50 == 0:
            print(f"  [neg] {i+1}/{len(samples)} done")

    return max_scores

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_percentiles(scores: list[float]) -> dict:
    percentiles = [1, 5, 10, 15, 20, 25, 50]
    return {p: round(float(np.percentile(scores, p)), 4) for p in percentiles}


def compute_threshold_stats(
    pos_records: list[dict],
    neg_max_scores: list[float],
    thresholds: list[float],
) -> list[dict]:
    """
    For each threshold compute:
      - gt_survival_rate:    % of positive queries where gt_score >= threshold
      - false_positive_rate: % of negative queries where max_score >= threshold
      - avg_candidates:      avg number of chunks passing threshold (positive queries)
    """
    n_pos = len(pos_records)
    n_neg = len(neg_max_scores)
    stats = []

    for t in thresholds:
        survived      = sum(1 for r in pos_records if r["gt_score"] >= t)
        false_pos     = sum(1 for s in neg_max_scores if s >= t)
        avg_remaining = np.mean([
            sum(1 for s in r["all_scores"] if s >= t)
            for r in pos_records
        ])
        stats.append({
            "threshold":           round(t, 2),
            "gt_survival_rate":    round(survived / n_pos, 4),
            "false_positive_rate": round(false_pos / n_neg, 4) if n_neg > 0 else 0.0,
            "avg_candidates":      round(float(avg_remaining), 2),
        })

    return stats

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_report(
    pos_percentiles: dict,
    neg_percentiles: dict,
    threshold_stats: list[dict],
    n_miss: int,
    n_pos: int,
    n_neg: int,
):
    print(f"\n{'='*62}")
    print(f"  Positive GT Score Percentiles  (retrieval miss → 0.0)")
    print(f"{'='*62}")
    for p, v in pos_percentiles.items():
        print(f"  P{p:<4} = {v:.4f}")

    print(f"\n  Negative Max Score Percentiles")
    print(f"  {'-'*42}")
    for p, v in neg_percentiles.items():
        print(f"  P{p:<4} = {v:.4f}")

    print(f"\n  Retrieval miss (GT not in top-{RETRIEVE_TOPK}): {n_miss}/{n_pos} ({n_miss/n_pos*100:.1f}%)")
    print(f"  Negative samples: {n_neg}")

    print(f"\n{'='*62}")
    print(f"  {'Threshold':<12} {'GT Survival':<16} {'False Pos Rate':<18} {'Avg Candidates'}")
    print(f"  {'-'*12} {'-'*16} {'-'*18} {'-'*14}")
    for s in threshold_stats:
        print(f"  {s['threshold']:<12.2f} {s['gt_survival_rate']:<16.4f} {s['false_positive_rate']:<18.4f} {s['avg_candidates']:.2f}")
    print(f"{'='*62}\n")


def save_csv(
    pos_records: list[dict],
    neg_max_scores: list[float],
    output_dir: str,
    timestamp: str,
):
    path = os.path.join(output_dir, f"threshold_gt_scores_{timestamp}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "question", "source_chunk_id", "score", "retrieval_miss"])
        for r in pos_records:
            writer.writerow(["positive", r["question"], r["source_chunk_id"], r["gt_score"], r["retrieval_miss"]])
        for s in neg_max_scores:
            writer.writerow(["negative", "", "", s, False])
    print(f"Scores saved → {path}")


def plot(
    pos_gt_scores: list[float],
    neg_max_scores: list[float],
    threshold_stats: list[dict],
    output_dir: str,
    timestamp: str,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Reranker Threshold Analysis", fontsize=14, fontweight="bold")

    # --- Left: overlapping score distributions ---
    bins = np.linspace(0, 1, 41)  # 40 bins, fixed range for fair comparison
    ax1.hist(neg_max_scores, bins=bins, alpha=0.6, color="#e74c3c", edgecolor="white",
             linewidth=0.5, label=f"Negative max score (n={len(neg_max_scores)})")
    ax1.hist(pos_gt_scores,  bins=bins, alpha=0.6, color="#4C9BE8", edgecolor="white",
             linewidth=0.5, label=f"Positive GT score  (n={len(pos_gt_scores)})")

    for p, color, ls in [(5, "#2ecc71", "--"), (10, "#f39c12", ":")]:
        val = float(np.percentile(pos_gt_scores, p))
        ax1.axvline(val, color=color, linestyle=ls, linewidth=1.5, label=f"Pos P{p}={val:.2f}")

    ax1.set_title("Score Distributions: Positive GT vs Negative Max")
    ax1.set_xlabel("Reranker Score")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=9)

    # --- Right: survival rate, false positive rate, avg candidates vs threshold ---
    thresholds      = [s["threshold"]           for s in threshold_stats]
    survival_rates  = [s["gt_survival_rate"]    for s in threshold_stats]
    false_pos_rates = [s["false_positive_rate"] for s in threshold_stats]
    avg_candidates  = [s["avg_candidates"]      for s in threshold_stats]

    ax2.plot(thresholds, survival_rates,  color="#4C9BE8", marker="o", markersize=4, label="GT Survival Rate")
    ax2.plot(thresholds, false_pos_rates, color="#e74c3c", marker="s", markersize=4, label="False Positive Rate")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Rate")
    ax2.set_ylim(0, 1.05)

    ax3 = ax2.twinx()
    ax3.plot(thresholds, avg_candidates, color="#95a5a6", marker="^", markersize=4,
             linestyle="--", label="Avg Candidates")
    ax3.set_ylabel("Avg Surviving Candidates", color="#95a5a6")
    ax3.tick_params(axis="y", labelcolor="#95a5a6")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")
    ax2.set_title("Survival Rate / False Positive Rate vs Threshold")

    plt.tight_layout()
    path = os.path.join(output_dir, f"threshold_analysis_{timestamp}.png")
    plt.savefig(path, dpi=150)
    print(f"Plot saved → {path}")
    plt.show()

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

    pos_gt_scores   = [r["gt_score"] for r in pos_records]
    n_miss          = sum(1 for r in pos_records if r["retrieval_miss"])
    pos_percentiles = compute_percentiles(pos_gt_scores)
    neg_percentiles = compute_percentiles(neg_max_scores)
    threshold_stats = compute_threshold_stats(pos_records, neg_max_scores, THRESHOLD_SWEEP)

    print_report(pos_percentiles, neg_percentiles, threshold_stats, n_miss, len(positives), len(negatives))
    save_csv(pos_records, neg_max_scores, OUTPUT_DIR, timestamp)
    plot(pos_gt_scores, neg_max_scores, threshold_stats, OUTPUT_DIR, timestamp)


if __name__ == "__main__":
    main()