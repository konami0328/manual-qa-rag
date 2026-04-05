import os
import json
import random
import hashlib

from datasets import load_dataset

from config import (
    EXPAND_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH,
    NEGATIVE_COUNT, TRAIN_RATIO, VAL_RATIO
)

random.seed(42)


def _load_expanded() -> list[dict]:
    """Load qa_expand.jsonl, flatten original + paraphrases into individual QA pairs."""
    qa_pairs = []
    with open(EXPAND_PATH) as f:
        for line in f:
            item = json.loads(line)
            all_questions = [item["question"]] + item["paraphrases"]
            for q in all_questions:
                qa_pairs.append({
                    "unique_id":       hashlib.md5(q.encode()).hexdigest(),
                    "question":        q,
                    "answer":          item["answer"],
                    "source_chunk_id": item["source_chunk_id"],
                    "page":            item["page"],
                })
    return qa_pairs


def _load_negatives(n: int) -> list[dict]:
    """Sample n queries from MS MARCO as negative examples."""
    print(f"Loading MS MARCO negatives (n={n})...")
    ds        = load_dataset("ms_marco", "v2.1", split="train")
    negatives = [row["query"] for row in ds.shuffle(seed=42).select(range(n))]
    return [
        {
            "unique_id":       hashlib.md5(q.encode()).hexdigest(),
            "question":        q,
            "answer":          "No Answer",
            "source_chunk_id": None,
            "page":            None,
        }
        for q in negatives
    ]


def _split(items: list[dict]) -> tuple[list, list, list]:
    """Split items into train/val/test by TRAIN_RATIO:VAL_RATIO:rest."""
    random.shuffle(items)
    n         = len(items)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))
    return items[:train_end], items[train_end:val_end], items[val_end:]


def _save(items: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def build_dataset(qa_pairs: list[dict], negatives: list[dict]) -> None:
    """Split positives and negatives, merge, shuffle, save to JSONL."""
    pos_train, pos_val, pos_test = _split(qa_pairs)
    neg_train, neg_val, neg_test = _split(negatives)

    train = pos_train + neg_train
    val   = pos_val   + neg_val
    test  = pos_test  + neg_test

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    _save(train, TRAIN_PATH)
    _save(val,   VAL_PATH)
    _save(test,  TEST_PATH)

    print(f"\n{'='*40}")
    print(f"Positives  : {len(qa_pairs)} → train {len(pos_train)} / val {len(pos_val)} / test {len(pos_test)}")
    print(f"Negatives  : {len(negatives)} → train {len(neg_train)} / val {len(neg_val)} / test {len(neg_test)}")
    print(f"Train total: {len(train)}")
    print(f"Val total  : {len(val)}")
    print(f"Test total : {len(test)}")
    print(f"{'='*40}")
    print(f"Saved to {TRAIN_PATH}, {VAL_PATH}, {TEST_PATH}")


def main():
    qa_pairs  = _load_expanded()
    print(f"Total QA pairs (after flatten): {len(qa_pairs)}")

    negatives = _load_negatives(NEGATIVE_COUNT)
    build_dataset(qa_pairs, negatives)


if __name__ == "__main__":
    main()