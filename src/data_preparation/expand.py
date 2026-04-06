import os
import json
import random
import threading
import concurrent.futures
from tqdm import tqdm

from openai import OpenAI
from dotenv import load_dotenv

from config import MIN_SCORE, FILTER_PATH, EXPAND_PATH, MAX_WORKERS

load_dotenv()

# --- Config ---
DEBUG      = False
DEBUG_SIZE = 10

# --- Prompt ---
# --- Prompt ---
GENERALIZE_PROMPT = """
Given the question below, generate 3 paraphrases that express the same meaning in different ways.

Requirements:
- Same intent, different phrasing
- Conversational tone, explore different ways people would naturally ask this

Output Format (JSON only, no markdown, no preamble):
["...", "...", "..."]

Question:
<question>
{question}
</question>
"""

# --- Client ---
_client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)
_model = os.environ["OPENAI_MODEL_NAME"]


def _call_llm(prompt: str, max_retries: int = 3) -> str | None:
    for attempt in range(max_retries):
        try:
            response = _client.chat.completions.create(
                model=_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"LLM call failed after {max_retries} attempts: {e}")
                return None


def _load_checkpoint() -> set:
    """Load already-expanded questions from checkpoint."""
    seen = set()
    if os.path.exists(EXPAND_PATH):
        with open(EXPAND_PATH) as f:
            for line in f:
                item = json.loads(line)
                seen.add(item["question"])
    return seen


def _load_qa_pairs() -> list[dict]:
    """Load qa_filtered.jsonl, keep only score >= MIN_SCORE."""
    qa_pairs = []
    with open(FILTER_PATH) as f:
        for line in f:
            item = json.loads(line)
            if item["score"] >= MIN_SCORE:
                qa_pairs.append(item)

    if DEBUG:
        random.seed(42)
        qa_pairs = random.sample(qa_pairs, min(DEBUG_SIZE, len(qa_pairs)))
        print(f"DEBUG mode: processing {len(qa_pairs)} random QA pairs")

    return qa_pairs


def expand_questions(qa_pairs: list[dict]) -> None:
    """Expand each question with 3 paraphrases, save to EXPAND_PATH (JSONL)."""
    seen      = _load_checkpoint()
    file_lock = threading.Lock()

    os.makedirs(os.path.dirname(EXPAND_PATH), exist_ok=True)

    to_process = [qa for qa in qa_pairs if qa["question"] not in seen]
    print(f"QA pairs to expand: {len(to_process)} (skipping {len(seen)} already done)")

    def _process(qa: dict):
        prompt   = GENERALIZE_PROMPT.format(question=qa["question"])
        raw_resp = _call_llm(prompt)
        if raw_resp is None:
            return

        try:
            raw        = raw_resp.strip()
            raw        = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            paraphrases = json.loads(raw)
        except json.JSONDecodeError:
            print(f"Failed to parse paraphrases: {raw_resp}")
            return

        if not isinstance(paraphrases, list) or len(paraphrases) == 0:
            return

        item = {
            "source_chunk_id": qa["source_chunk_id"],
            "page":            qa["page"],
            "question":        qa["question"],
            "answer":          qa["answer"],
            "paraphrases":     paraphrases,
        }

        with file_lock:
            with open(EXPAND_PATH, "a") as f:
                f.write(json.dumps(item) + "\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm(executor.map(_process, to_process), total=len(to_process)))


def main():
    qa_pairs = _load_qa_pairs()
    print(f"Total QA pairs after score filter: {len(qa_pairs)}")
    expand_questions(qa_pairs)
    print(f"Done. Saved to {EXPAND_PATH}")


if __name__ == "__main__":
    main()