"""
Reads generation_raw_<timestamp>.jsonl produced by generate.py and computes:

    Negatives  : refusal rate  (string match, zero cost)
    Positives
      retrieval miss : logged, skipped
      retrieval hit  : Faithfulness + Answer Relevancy (RAGAS) + ROUGE-L

LLM  : DeepSeek-chat via existing OPENAI_BASE_URL / OPENAI_API_KEY
Embed: bge-m3 (BGEM3FlagModel) for Answer Relevancy cosine similarity

Output:
    eval/generate/results/generation_summary_<timestamp>.json
"""

import os
import json
from datetime import datetime

import numpy as np
from FlagEmbedding import BGEM3FlagModel
from rouge_score import rouge_scorer
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import BaseRagasEmbeddings
from langchain_openai import ChatOpenAI
from datasets import Dataset
from dotenv import load_dotenv

from config import EMBEDDING_MODEL_PATH

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NO_ANSWER  = "This information is not covered in the provided context."
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eval", "generate", "results")

# ---------------------------------------------------------------------------
# BGE-M3 Embedding wrapper for RAGAS
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

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_raw(path: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records into negatives, positive hits, positive misses."""
    negatives, hits, misses = [], [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            if r["sample_type"] == "negative":
                negatives.append(r)
            elif r["retrieval_hit"]:
                hits.append(r)
            else:
                misses.append(r)
    return negatives, hits, misses

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def refusal_rate(negatives: list[dict]) -> dict:
    refused = sum(1 for r in negatives if NO_ANSWER in r["answer"])
    return {
        "total":        len(negatives),
        "refused":      refused,
        "refusal_rate": round(refused / len(negatives), 4),
    }


def compute_rouge_l(hits: list[dict]) -> list[float]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [
        scorer.score(r["ground_truth"], r["answer"])["rougeL"].fmeasure
        for r in hits
    ]


def compute_ragas(hits: list[dict]) -> dict:
    llm   = LangchainLLMWrapper(ChatOpenAI(
        model       = os.environ["OPENAI_MODEL_NAME"],
        openai_api_key  = os.environ["OPENAI_API_KEY"],
        openai_api_base = os.environ["OPENAI_BASE_URL"],
        temperature = 0,
    ))
    embed = BGEM3RagasEmbeddings(EMBEDDING_MODEL_PATH)

    dataset = Dataset.from_list([
        {
            "question":     r["question"],
            "answer":       r["answer"],
            "contexts":     r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in hits
    ])

    result = evaluate(
        dataset,
        metrics   = [faithfulness, answer_relevancy],
        llm       = llm,
        embeddings = embed,
    )
    return {
        "faithfulness":     round(result["faithfulness"],     4),
        "answer_relevancy": round(result["answer_relevancy"], 4),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python eval_generation.py <path/to/generation_raw_*.jsonl>")
        sys.exit(1)

    raw_path  = sys.argv[1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading {raw_path}...")
    negatives, hits, misses = load_raw(raw_path)
    print(f"Negatives: {len(negatives)}  Hits: {len(hits)}  Misses: {len(misses)}")

    print("\nComputing refusal rate...")
    neg_results = refusal_rate(negatives)
    print(f"  Refusal rate: {neg_results['refusal_rate']}  ({neg_results['refused']}/{neg_results['total']})")

    print("\nComputing ROUGE-L...")
    rouge_scores = compute_rouge_l(hits)
    rouge_mean   = round(float(np.mean(rouge_scores)), 4)
    print(f"  ROUGE-L mean: {rouge_mean}")

    print("\nComputing RAGAS (Faithfulness + Answer Relevancy)...")
    ragas_results = compute_ragas(hits)
    print(f"  Faithfulness    : {ragas_results['faithfulness']}")
    print(f"  Answer Relevancy: {ragas_results['answer_relevancy']}")

    summary = {
        "timestamp":       timestamp,
        "source":          raw_path,
        "negatives":       neg_results,
        "positives": {
            "total":            len(hits) + len(misses),
            "retrieval_hit":    len(hits),
            "retrieval_miss":   len(misses),
            "rouge_l_mean":     rouge_mean,
            "faithfulness":     ragas_results["faithfulness"],
            "answer_relevancy": ragas_results["answer_relevancy"],
        },
    }

    summary_path = os.path.join(OUTPUT_DIR, f"generation_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {summary_path}")


if __name__ == "__main__":
    main()