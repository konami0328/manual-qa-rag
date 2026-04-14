"""
Per-sample evaluation of generation_raw_<timestamp>.jsonl.

Configure TARGET_POS and TARGET_NEG at the top to control how many samples
to evaluate. Samples are randomly drawn from the raw file, skipping any
already present in the output checkpoint file.

Metrics
-------
Positives (retrieval hit)  : ROUGE-L, Faithfulness, Answer Relevancy (RAGAS)
Positives (retrieval miss) : all scores = null, logged as miss
Negatives                  : refused (bool), other scores = null

Checkpoint
----------
Keyed by question string. Safe to interrupt and resume.

Output
------
eval/generate/results/generation_eval_<timestamp>.jsonl

Usage
-----
python -m eval.generate.eval <path/to/generation_raw_*.jsonl>
"""

import os
import sys
import json
import random
import traceback
from datetime import datetime

import numpy as np
from FlagEmbedding import BGEM3FlagModel
from rouge_score import rouge_scorer as rouge_scorer_lib
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import BaseRagasEmbeddings
from langchain_openai import ChatOpenAI
from datasets import Dataset
from dotenv import load_dotenv

from config import EMBEDDING_MODEL_PATH

load_dotenv()

# ===========================================================================
# Config
# ===========================================================================
TARGET_POS   = 400   # number of positive samples to evaluate
TARGET_NEG   = 150   # number of negative samples to evaluate
RANDOM_SEED  = 42
# ===========================================================================

NO_ANSWER   = "This information is not covered in the provided context."
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "results")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "generation_eval_processed.jsonl")

# ---------------------------------------------------------------------------
# BGE-M3 embedding wrapper for RAGAS
# ---------------------------------------------------------------------------

class BGEM3RagasEmbeddings(BaseRagasEmbeddings):
    def __init__(self, model_path: str):
        self._model = BGEM3FlagModel(model_path, use_fp16=True)

    def embed_query(self, text: str) -> list[float]:
        out = self._model.encode([text], return_dense=True)
        return out["dense_vecs"][0].tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out = self._model.encode(texts, return_dense=True)
        return out["dense_vecs"].tolist()

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_raw(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_checkpoint(path: str) -> tuple[set, int, int]:
    """Return (seen questions, current pos count, current neg count)."""
    seen  = set()
    n_pos = 0
    n_neg = 0
    if not os.path.exists(path):
        return seen, n_pos, n_neg
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            seen.add(item["question"])
            if item["sample_type"] == "positive":
                n_pos += 1
            else:
                n_neg += 1
    return seen, n_pos, n_neg


def sample_records(
    all_raw: list[dict],
    seen: set,
    cur_pos: int,
    cur_neg: int,
) -> list[dict]:
    """Randomly sample needed pos/neg from unseen records."""
    random.seed(RANDOM_SEED)

    unseen_pos = [r for r in all_raw if r["sample_type"] == "positive" and r["question"] not in seen]
    unseen_neg = [r for r in all_raw if r["sample_type"] == "negative" and r["question"] not in seen]

    need_pos = max(0, TARGET_POS - cur_pos)
    need_neg = max(0, TARGET_NEG - cur_neg)

    if need_pos == 0 and need_neg == 0:
        print("Already at target counts, nothing to do.")
        return []

    if len(unseen_pos) < need_pos:
        print(f"Warning: only {len(unseen_pos)} unseen positives available, need {need_pos}")
        need_pos = len(unseen_pos)
    if len(unseen_neg) < need_neg:
        print(f"Warning: only {len(unseen_neg)} unseen negatives available, need {need_neg}")
        need_neg = len(unseen_neg)

    sampled = random.sample(unseen_pos, need_pos) + random.sample(unseen_neg, need_neg)
    random.shuffle(sampled)

    print(f"Sampled: pos={need_pos}  neg={need_neg}  total={len(sampled)}")
    return sampled

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_rouge = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=True)


def score_rouge_l(prediction: str, reference: str) -> float:
    return round(_rouge.score(reference, prediction)["rougeL"].fmeasure, 4)


def score_ragas_single(question, answer, contexts, ground_truth, llm, embed) -> dict:
    try:
        dataset = Dataset.from_list([{
            "question":     question,
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": ground_truth,
        }])
        result = evaluate(
            dataset,
            metrics    = [faithfulness, answer_relevancy],
            llm        = llm,
            embeddings = embed,
        )
        return {
            "faithfulness":     round(float(result["faithfulness"][0]),     4),
            "answer_relevancy": round(float(result["answer_relevancy"][0]), 4),
        }
    except Exception as e:
        traceback.print_exc()
        print(f"  [RAGAS error] {e}")
        return {"faithfulness": None, "answer_relevancy": None}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python eval.py <path/to/generation_raw_*.jsonl>")
        sys.exit(1)

    raw_path = sys.argv[1]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading raw: {raw_path}")
    all_raw = load_raw(raw_path)
    print(f"Raw records: {len(all_raw)}")

    print(f"Loading checkpoint: {OUTPUT_PATH}")
    seen, cur_pos, cur_neg = load_checkpoint(OUTPUT_PATH)
    print(f"Already scored: pos={cur_pos}  neg={cur_neg}  total={len(seen)}")
    print(f"Targets: pos={TARGET_POS}  neg={TARGET_NEG}")

    to_process = sample_records(all_raw, seen, cur_pos, cur_neg)
    if not to_process:
        return

    print("Initializing LLM and embeddings for RAGAS...")
    llm = LangchainLLMWrapper(ChatOpenAI(
        model           = os.environ["OPENAI_MODEL_NAME"],
        openai_api_key  = os.environ["OPENAI_API_KEY"],
        openai_api_base = os.environ["OPENAI_BASE_URL"],
        temperature     = 0,
    ))
    embed = BGEM3RagasEmbeddings(EMBEDDING_MODEL_PATH)

    # ---------------------------------------------------------------------------
    # Per-sample loop
    # ---------------------------------------------------------------------------

    with open(OUTPUT_PATH, "a") as out_f:
        for i, r in enumerate(to_process):
            question      = r["question"]
            sample_type   = r["sample_type"]
            answer        = r["answer"]
            ground_truth  = r.get("ground_truth")
            retrieval_hit = r.get("retrieval_hit")
            contexts      = r.get("contexts")

            record = {
                "sample_type":      sample_type,
                "question":         question,
                "answer":           answer,
                "ground_truth":     ground_truth,
                "retrieval_hit":    retrieval_hit,
                "contexts":         contexts,
                "rouge_l":          None,
                "faithfulness":     None,
                "answer_relevancy": None,
                "refused":          None,
            }

            if sample_type == "negative":
                record["refused"] = NO_ANSWER in answer

            elif retrieval_hit is False:
                pass

            elif retrieval_hit is True:
                record["rouge_l"] = score_rouge_l(answer, ground_truth)
                ragas = score_ragas_single(
                    question     = question,
                    answer       = answer,
                    contexts     = contexts or [],
                    ground_truth = ground_truth,
                    llm          = llm,
                    embed        = embed,
                )
                record["faithfulness"]     = ragas["faithfulness"]
                record["answer_relevancy"] = ragas["answer_relevancy"]

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            seen.add(question)

            scores_str = ""
            if sample_type == "negative":
                scores_str = f"refused={record['refused']}"
            elif retrieval_hit is True:
                scores_str = (
                    f"rouge_l={record['rouge_l']}  "
                    f"faith={record['faithfulness']}  "
                    f"rel={record['answer_relevancy']}"
                )
            else:
                scores_str = "retrieval_miss"

            print(f"[{i+1}/{len(to_process)}] {sample_type:<8}  {scores_str}")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------

    all_scored = load_raw(OUTPUT_PATH)
    positives  = [r for r in all_scored if r["sample_type"] == "positive"]
    negatives  = [r for r in all_scored if r["sample_type"] == "negative"]
    hits       = [r for r in positives  if r["retrieval_hit"] is True]
    misses     = [r for r in positives  if r["retrieval_hit"] is False]
    refused    = [r for r in negatives  if r.get("refused") is True]

    rouge_vals = [r["rouge_l"]          for r in hits if r.get("rouge_l")          is not None]
    faith_vals = [r["faithfulness"]     for r in hits if r.get("faithfulness")     is not None]
    rel_vals   = [r["answer_relevancy"] for r in hits if r.get("answer_relevancy") is not None]

    print(f"\n{'='*50}")
    print(f"Positives       : {len(positives)}  (hit={len(hits)}, miss={len(misses)})")
    print(f"Negatives       : {len(negatives)}  refusal_rate={len(refused)/max(len(negatives),1):.4f}")
    if rouge_vals:
        print(f"ROUGE-L mean    : {round(float(np.mean(rouge_vals)), 4)}")
    if faith_vals:
        print(f"Faithfulness    : {round(float(np.mean(faith_vals)), 4)}")
    if rel_vals:
        print(f"Answer Relevancy: {round(float(np.mean(rel_vals)), 4)}")
    print(f"{'='*50}")
    print(f"Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()