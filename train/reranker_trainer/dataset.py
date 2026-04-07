"""
Input:  RERANKER_TRAIN_PATH / RERANKER_VAL_PATH  — triplets JSONL from mine.py
        MongoDB manual_text                       — chunk text lookup
Output: RerankerDataset — flat list of (input_ids, attention_mask, label)

Each triplet emits 3 samples:
    (query, pos_chunk)      → label 1.0
    (query, weak_pos_chunk) → label 0.5
    (query, neg_chunk)      → label 0.0
"""

import json

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config import RERANKER_MODEL_PATH
from src.client.mongodb_config import MongoConfig

MAX_LENGTH = 768  # covers 99% chunk length (686 tokens) + query + special tokens


def _load_chunk_lookup() -> dict[str, str]:
    """Load all chunks from MongoDB, return {unique_id: page_content}."""
    col = MongoConfig.get_collection("manual_text")
    return {d["unique_id"]: d["page_content"] for d in col.find()}


def _load_triplets(path: str) -> list[dict]:
    triplets = []
    with open(path) as f:
        for line in f:
            triplets.append(json.loads(line))
    return triplets


class RerankerDataset(Dataset):

    def __init__(self, path: str):
        self._tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_PATH)
        chunk_lookup    = _load_chunk_lookup()
        triplets        = _load_triplets(path)

        self._samples: list[tuple[str, str, float]] = []  # (query, chunk_text, label)

        for t in triplets:
            query = t["query"]
            self._samples.append((query, chunk_lookup[t["pos_chunk_id"]],      1.0))
            self._samples.append((query, chunk_lookup[t["weak_pos_chunk_id"]], 0.5))
            self._samples.append((query, chunk_lookup[t["neg_chunk_id"]],      0.0))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        query, chunk_text, label = self._samples[idx]

        encoded = self._tokenizer(
            query,
            chunk_text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids":      encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.float),
        }


if __name__ == "__main__":
    from config import RERANKER_TRAIN_PATH

    ds = RerankerDataset(RERANKER_TRAIN_PATH)
    print(f"Total samples: {len(ds)}")

    sample = ds[0]
    print(f"input_ids shape:      {sample['input_ids'].shape}")
    print(f"attention_mask shape: {sample['attention_mask'].shape}")
    print(f"label:                {sample['label']}")