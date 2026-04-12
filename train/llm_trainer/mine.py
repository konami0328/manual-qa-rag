"""
Mine training and validation samples for LLM fine-tuning.

Pipeline per query:
    1. HybridRetriever (topk=TOPK) + FinetunedReranker → retrieved context
    2. Positive samples: call DeepSeek with (question, context) → answer
       Negative samples: answer = NO_ANSWER (fixed string, no API call)
    3. Write to JSONL checkpoint

Splits processed:
    train.jsonl → LLM_TRAIN_PATH  (LLM_MINE_POS positives + LLM_MINE_NEG negatives)
    val.jsonl   → LLM_VAL_PATH    (LLM_MINE_POS // 5 positives + LLM_MINE_NEG // 5 negatives)

Positive sample selection:
    - source_chunk_id must not be None
    - retrieval hit required (gt chunk must appear in top-k results)
    - miss samples are skipped

Negative sample selection:
    - source_chunk_id is None (MS MARCO queries)
    - retrieval always runs to build a realistic noisy context
    - answer is fixed NO_ANSWER string

Output format per line:
{
    "question":    str,
    "context":     str,   # formatted retrieved chunks (Chunk i, p.N)
    "answer":      str,   # DeepSeek-generated for positives, NO_ANSWER for negatives
    "sample_type": "positive" | "negative"
}
"""

import os
import json
import random
import threading
import concurrent.futures
from tqdm import tqdm

from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.documents import Document

from config import (
    TRAIN_PATH, VAL_PATH,
    LLM_TRAIN_PATH, LLM_VAL_PATH,
    TOPK, GENERATION_TOPK, GENERATION_THRESHOLD,
    MAX_WORKERS,
)
from src.client.mongodb_config import MongoConfig
from src.retriever.retrieve_hybrid import HybridRetriever
from src.reranker.rerank_bge_finetuned import FinetunedReranker
from src.client.llm_generate_vllm import LLM_CHAT_PROMPT

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NO_ANSWER   = "This information is not covered in the provided context."
RANDOM_SEED = 42

# 3000 pos, 200 neg, val split is 1/5 of train targets
LLM_MINE_POS          = 3000
LLM_MINE_NEG          = 200
VAL_MINE_POS = LLM_MINE_POS // 5
VAL_MINE_NEG = LLM_MINE_NEG // 5

# ---------------------------------------------------------------------------
# DeepSeek client
# ---------------------------------------------------------------------------

_client = OpenAI(
    api_key  = os.environ["OPENAI_API_KEY"],
    base_url = os.environ["OPENAI_BASE_URL"],
)
_model = os.environ["OPENAI_MODEL_NAME"]


def _call_deepseek(question: str, context: str, max_retries: int = 3) -> str | None:
    """Call DeepSeek with the same prompt used at inference time."""
    prompt = LLM_CHAT_PROMPT.format(context=context, query=question)
    for attempt in range(max_retries):
        try:
            response = _client.chat.completions.create(
                model       = _model,
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0.3,
                max_tokens  = 1500,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [DeepSeek error] {e}")
                return None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_context(chunks: list[Document]) -> str:
    """Format retrieved chunks into context string, same as inference."""
    return "\n".join(
        f"[Chunk {i+1}, p.{doc.metadata.get('page', '?')}] {doc.page_content}"
        for i, doc in enumerate(chunks)
    )


def _load_split(path: str) -> tuple[list[dict], list[dict]]:
    """Load JSONL split, return (positives, negatives)."""
    positives, negatives = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("source_chunk_id") is not None:
                positives.append(item)
            else:
                negatives.append(item)
    return positives, negatives


def _load_checkpoint(path: str) -> set:
    """Return set of already-processed questions."""
    seen = set()
    if not os.path.exists(path):
        return seen
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                seen.add(json.loads(line)["question"])
    return seen


def _sample(items: list[dict], n: int, seen: set) -> list[dict]:
    """Sample n unseen items randomly."""
    unseen = [x for x in items if x["question"] not in seen]
    n = min(n, len(unseen))
    return random.sample(unseen, n)

# ---------------------------------------------------------------------------
# Mine one split
# ---------------------------------------------------------------------------

def mine_split(
    positives: list[dict],
    negatives: list[dict],
    target_pos: int,
    target_neg: int,
    retriever: HybridRetriever,
    reranker: FinetunedReranker,
    output_path: str,
) -> None:
    """Mine one split (train or val) and write to output_path."""
    seen      = _load_checkpoint(output_path)
    file_lock = threading.Lock()

    # count already done per type
    cur_pos = cur_neg = 0
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["sample_type"] == "positive":
                    cur_pos += 1
                else:
                    cur_neg += 1

    need_pos = max(0, target_pos - cur_pos)
    need_neg = max(0, target_neg - cur_neg)

    print(f"  Already done: pos={cur_pos}  neg={cur_neg}")
    print(f"  Need:         pos={need_pos}  neg={need_neg}")

    sample_pos = _sample(positives, need_pos, seen)
    sample_neg = _sample(negatives, need_neg, seen)

    if not sample_pos and not sample_neg:
        print("  Already at target, nothing to do.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ---------------------------------------------------------------------------
    # Step 1: retrieval + rerank (serial, GPU-bound)
    # ---------------------------------------------------------------------------

    print(f"  Running retrieval+rerank for {len(sample_pos)} pos + {len(sample_neg)} neg...")

    retrieved_pos = []  # list of (item, chunks, hit)
    retrieved_neg = []  # list of (item, chunks)

    for item in tqdm(sample_pos, desc="  Retrieving positives"):
        candidates = retriever.retrieve(item["question"], topk=TOPK)
        ranked     = reranker.rerank(item["question"], candidates)
        chunks     = [c for c in ranked if c.metadata["rerank_score"] >= GENERATION_THRESHOLD][:GENERATION_TOPK]

        # check retrieval hit
        chunk_ids = [c.metadata["unique_id"] for c in chunks]
        hit       = item["source_chunk_id"] in chunk_ids

        if not hit:
            continue  # skip retrieval misses

        retrieved_pos.append((item, chunks))

    for item in tqdm(sample_neg, desc="  Retrieving negatives"):
        candidates = retriever.retrieve(item["question"], topk=TOPK)
        ranked     = reranker.rerank(item["question"], candidates)
        chunks     = [c for c in ranked if c.metadata["rerank_score"] >= GENERATION_THRESHOLD][:GENERATION_TOPK]
        retrieved_neg.append((item, chunks))

    print(f"  After retrieval filter: pos={len(retrieved_pos)}  neg={len(retrieved_neg)}")

    # ---------------------------------------------------------------------------
    # Step 2: DeepSeek generation (concurrent, for positives only)
    # ---------------------------------------------------------------------------

    def _process_pos(item_chunks: tuple) -> dict | None:
        item, chunks = item_chunks
        context = _build_context(chunks)
        answer  = _call_deepseek(item["question"], context)
        if answer is None:
            return None
        return {
            "question":    item["question"],
            "context":     context,
            "answer":      answer,
            "sample_type": "positive",
        }

    def _process_neg(item_chunks: tuple) -> dict:
        item, chunks = item_chunks
        context = _build_context(chunks) if chunks else ""
        return {
            "question":    item["question"],
            "context":     context,
            "answer":      NO_ANSWER,
            "sample_type": "negative",
        }

    print(f"  Generating answers via DeepSeek ({len(retrieved_pos)} pos, concurrent)...")

    with open(output_path, "a") as out_f:

        # positives — concurrent DeepSeek calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_process_pos, x): x for x in retrieved_pos}
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="  DeepSeek positives",
            ):
                record = future.result()
                if record is None:
                    continue
                with file_lock:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()

        # negatives — no API call needed, write directly
        for item_chunks in tqdm(retrieved_neg, desc="  Writing negatives"):
            record = _process_neg(item_chunks)
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    # final count
    final = _load_checkpoint(output_path)
    print(f"  Done. Total in checkpoint: {len(final)}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(RANDOM_SEED)

    print("Loading docs from MongoDB...")
    col  = MongoConfig.get_collection("manual_text")
    docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in col.find()]
    print(f"  Docs loaded: {len(docs)}")

    print("Initializing retriever and reranker...")
    retriever = HybridRetriever(docs)
    reranker  = FinetunedReranker()

    for split_name, jsonl_path, output_path, target_pos, target_neg in [
        ("train", TRAIN_PATH, LLM_TRAIN_PATH, LLM_MINE_POS,     LLM_MINE_NEG),
        ("val",   VAL_PATH,   LLM_VAL_PATH,   VAL_MINE_POS,     VAL_MINE_NEG),
    ]:
        print(f"\n[{split_name}] Loading from {jsonl_path}...")
        positives, negatives = _load_split(jsonl_path)
        print(f"  Positives: {len(positives)}  Negatives: {len(negatives)}")
        print(f"  Targets:   pos={target_pos}  neg={target_neg}")

        mine_split(
            positives   = positives,
            negatives   = negatives,
            target_pos  = target_pos,
            target_neg  = target_neg,
            retriever   = retriever,
            reranker    = reranker,
            output_path = output_path,
        )
        print(f"  Saved → {output_path}")


if __name__ == "__main__":
    main()