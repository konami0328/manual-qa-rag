"""
Input:  TRAIN_PATH / VAL_PATH  — {unique_id, question, source_chunk_id, ...}
        MongoDB manual_text    — all chunks
Output: RERANKER_TRAIN_PATH / RERANKER_VAL_PATH  — triplets JSONL

Pipeline per query:
    HybridRetriever (topk=MINE_TOPK) → Reranker (threshold=0.0) → ranked list
    adjacency filter: exclude abs(candidate.chunk_index - gt.chunk_index) <= 1
    weak_pos pool = filtered rank 2-5  → sample 1
    neg pool      = filtered rank 6-10 → sample 1
    emit 3 samples: (query, gt, 1.0), (query, weak_pos, 0.5), (query, neg, 0.0)

Output format per line:
{
    "query":            str,
    "pos_chunk_id":     str,
    "weak_pos_chunk_id": str,
    "neg_chunk_id":     str,
}
"""

import os
import json
import random
import threading
import concurrent.futures
from tqdm import tqdm

from langchain_core.documents import Document
from FlagEmbedding import FlagReranker

from config import (
    TRAIN_PATH, VAL_PATH,
    RERANKER_TRAIN_PATH, RERANKER_VAL_PATH,
    RERANKER_MODEL_PATH, MINE_TOPK,
)
from src.client.mongodb_config import MongoConfig
from src.retriever.retrieve_hybrid import HybridRetriever

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEBUG        = True
DEBUG_SIZE   = 10
MINE_WORKERS = 1   # reranker is GPU-bound; parallelism here adds little

WEAK_POS_RANKS = (2, 5)   # inclusive, 1-indexed after reranker sort
NEG_RANKS      = (6, 10)

# ---------------------------------------------------------------------------
# Reranker wrapper (threshold=0.0, no filtering)
# ---------------------------------------------------------------------------

class _MineReranker:
    def __init__(self):
        self._model = FlagReranker(RERANKER_MODEL_PATH, use_fp16=True)

    def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        """Score and sort; no threshold filtering."""
        if not docs:
            return []
        pairs  = [(query, doc.page_content) for doc in docs]
        scores = self._model.compute_score(pairs, normalize=True)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        for doc, score in ranked:
            doc.metadata["rerank_score"] = round(score, 4)
        return [doc for doc, _ in ranked]

# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _load_positives(path: str) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            if item.get("source_chunk_id") is not None:
                samples.append(item)
    return samples


def _load_checkpoint(path: str) -> set:
    seen = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                seen.add(item["query_unique_id"])
    return seen


def _build_chunk_lookup(docs: list[Document]) -> dict:
    return {d.metadata["unique_id"]: d for d in docs}

# ---------------------------------------------------------------------------
# Mine one split
# ---------------------------------------------------------------------------

def mine_split(
    samples: list[dict],
    retriever: HybridRetriever,
    reranker: _MineReranker,
    chunk_lookup: dict,
    output_path: str,
) -> None:
    seen      = _load_checkpoint(output_path)
    file_lock = threading.Lock()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    to_process = [s for s in samples if s["unique_id"] not in seen]
    print(f"  Queries to mine: {len(to_process)}  (skipping {len(seen)} already done)")

    skipped_no_gt   = 0
    skipped_no_pool = 0
    emitted         = 0

    for sample in tqdm(to_process):
        query        = sample["question"]
        gt_chunk_id  = sample["source_chunk_id"]
        query_uid    = sample["unique_id"]

        # ground truth chunk must exist
        gt_doc = chunk_lookup.get(gt_chunk_id)
        if gt_doc is None:
            skipped_no_gt += 1
            continue
        gt_chunk_index = gt_doc.metadata.get("chunk_index")

        # retrieve → rerank
        candidates = retriever.retrieve(query, topk=MINE_TOPK)
        ranked     = reranker.rerank(query, candidates)

        # build ranked list excluding ground truth and adjacent chunks
        filtered = []
        for doc in ranked:
            uid = doc.metadata["unique_id"]
            if uid == gt_chunk_id:
                continue
            ci = doc.metadata.get("chunk_index")
            if ci is not None and gt_chunk_index is not None:
                if abs(ci - gt_chunk_index) <= 1:
                    continue
            filtered.append(doc)

        # rank windows are 1-indexed over filtered list
        weak_pos_pool = filtered[WEAK_POS_RANKS[0]-1 : WEAK_POS_RANKS[1]]   # index 1-4
        neg_pool      = filtered[NEG_RANKS[0]-1      : NEG_RANKS[1]]         # index 5-9

        if not weak_pos_pool or not neg_pool:
            skipped_no_pool += 1
            continue

        weak_pos_doc = random.choice(weak_pos_pool)
        neg_doc      = random.choice(neg_pool)

        triplet = {
            "query_unique_id":    query_uid,
            "query":              query,
            "pos_chunk_id":       gt_chunk_id,
            "weak_pos_chunk_id":  weak_pos_doc.metadata["unique_id"],
            "neg_chunk_id":       neg_doc.metadata["unique_id"],
        }

        with file_lock:
            with open(output_path, "a") as f:
                f.write(json.dumps(triplet) + "\n")
        emitted += 1

    print(f"  Emitted: {emitted}  |  Skipped (no GT): {skipped_no_gt}  |  Skipped (empty pool): {skipped_no_pool}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(42)

    print("Loading docs from MongoDB...")
    col          = MongoConfig.get_collection("manual_text")
    docs         = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in col.find()]
    chunk_lookup = _build_chunk_lookup(docs)
    print(f"  Docs loaded: {len(docs)}")

    print("Initializing retriever and reranker...")
    retriever = HybridRetriever(docs)
    reranker  = _MineReranker()

    for split_name, jsonl_path, output_path in [
        ("train", TRAIN_PATH,  RERANKER_TRAIN_PATH),
        ("val",   VAL_PATH,    RERANKER_VAL_PATH),
    ]:
        print(f"\n[{split_name}] Loading positives from {jsonl_path}...")
        samples = _load_positives(jsonl_path)
        print(f"  Positive samples: {len(samples)}")

        if DEBUG:
            samples = random.sample(samples, min(DEBUG_SIZE, len(samples)))
            print(f"  DEBUG mode: {len(samples)} samples")

        mine_split(samples, retriever, reranker, chunk_lookup, output_path)
        print(f"  Saved → {output_path}")


if __name__ == "__main__":
    main()