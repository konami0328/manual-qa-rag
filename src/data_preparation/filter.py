import os
import json
import threading
import concurrent.futures
from tqdm import tqdm

from openai import OpenAI
from dotenv import load_dotenv

from src.client.mongodb_config import MongoConfig
from config import QA_CKPT_PATH, MAX_WORKERS, MIN_SCORE, FILTER_PATH

load_dotenv()

# --- Config ---
DEBUG        = True
DEBUG_SIZE   = 30


# --- Prompt ---
QA_QUALITY_PROMPT = """
You are an expert evaluator for a Tesla Model Y owner's manual Q&A dataset.
Score the following question-answer pair based on the source document.

**Scoring Criteria (1-5):**
5 - Perfectly grounded, complete, self-contained answer
4 - Mostly grounded, minor incompleteness
3 - Acceptable but partially incomplete or slightly off
2 - Missing key information or partially ungrounded
1 - Hallucinated, ungrounded, or references page numbers/sections

**A good question:**
- Asks about facts, procedures, warnings
- NOT a summarization request ("what does this section describe?")
- NOT about figures or images ("what does figure 4 show?")

**A good answer must:**
- Be fully supported by the source document
- Not contain information absent from the document
- Cover all key information relevant to the question
- NOT reference page numbers or other sections

**Output (JSON only, no preamble):**
{{"score": int, "reason": "one sentence max"}}

**Source document:**
<document>
{chunk}
</document>

**Question:**
<question>
{question}
</question>

**Answer:**
<answer>
{answer}
</answer>
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
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"LLM call failed after {max_retries} attempts: {e}")
                return None


def _load_checkpoint() -> set:
    """Load already-scored (source_chunk_id, question) pairs from checkpoint."""
    seen = set()
    if os.path.exists(FILTER_PATH):
        with open(FILTER_PATH) as f:
            for line in f:
                item = json.loads(line)
                seen.add((item["source_chunk_id"], item["question"]))
    return seen


def _load_qa_pairs(col) -> list[dict]:
    """Load and parse qa_raw.jsonl, fetch source chunk from MongoDB."""
    qa_pairs = []
    with open(QA_CKPT_PATH) as f:
        lines = f.readlines()

    if DEBUG:
        import random
        random.seed(42)
        lines = random.sample(lines, min(DEBUG_SIZE, len(lines)))
        print(f"DEBUG mode: processing {DEBUG_SIZE} random chunks")

    for line in lines:
        item = json.loads(line)
        source_chunk_id = item["source_chunk_id"]
        page            = item["page"]

        # parse raw_resp
        try:
            raw = item["raw_resp"].strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            pairs = json.loads(raw)
        except json.JSONDecodeError:
            print(f"Failed to parse raw_resp for chunk {source_chunk_id}, skipping.")
            continue

        if not pairs:
            continue

        # fetch source chunk from MongoDB
        record = col.find_one({"unique_id": source_chunk_id})
        if not record:
            print(f"Chunk {source_chunk_id} not found in MongoDB, skipping.")
            continue

        for qa in pairs:
            qa_pairs.append({
                "source_chunk_id": source_chunk_id,
                "page":            page,
                "question":        qa["question"],
                "answer":          qa["answer"],
                "chunk":           record["page_content"],
            })

    return qa_pairs


def score_qa(qa_pairs: list[dict]) -> None:
    """Score each QA pair concurrently, save all to FILTER_PATH (JSONL)."""
    seen      = _load_checkpoint()
    file_lock = threading.Lock()

    os.makedirs(os.path.dirname(FILTER_PATH), exist_ok=True)

    to_process = [
        qa for qa in qa_pairs
        if (qa["source_chunk_id"], qa["question"]) not in seen
    ]
    print(f"QA pairs to score: {len(to_process)} (skipping {len(seen)} already done)")

    def _process(qa: dict):
        prompt   = QA_QUALITY_PROMPT.format(
            chunk    = qa["chunk"],
            question = qa["question"],
            answer   = qa["answer"],
        )
        raw_resp = _call_llm(prompt)
        if raw_resp is None:
            return

        try:
            raw = raw_resp.strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            result = json.loads(raw)
        except json.JSONDecodeError:
            print(f"Failed to parse score response: {raw_resp}")
            return

        item = {
            "source_chunk_id": qa["source_chunk_id"],
            "page":            qa["page"],
            "question":        qa["question"],
            "answer":          qa["answer"],
            "score":           result["score"],
            "reason":          result["reason"],
        }

        with file_lock:
            with open(FILTER_PATH, "a") as f:
                f.write(json.dumps(item) + "\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm(executor.map(_process, to_process), total=len(to_process)))


def print_stats() -> None:
    """Print score distribution and keep/drop preview."""
    if not os.path.exists(FILTER_PATH):
        print("No scored pairs yet.")
        return
    all_pairs = []
    with open(FILTER_PATH) as f:
        for line in f:
            all_pairs.append(json.loads(line))

    total   = len(all_pairs)
    kept    = sum(1 for qa in all_pairs if qa["score"] >= MIN_SCORE)
    dropped = total - kept

    print(f"\n{'='*40}")
    print(f"Total QA pairs scored : {total}")
    print(f"Would keep (>= {MIN_SCORE})    : {kept}")
    print(f"Would drop (< {MIN_SCORE})     : {dropped} ({dropped/total*100:.1f}%)")
    print(f"{'='*40}")

    # score distribution
    print("\nScore distribution:")
    for score in range(1, 6):
        count = sum(1 for qa in all_pairs if qa["score"] == score)
        print(f"  Score {score}: {count} ({count/total*100:.1f}%)")


def main():
    col      = MongoConfig.get_collection("manual_text")
    qa_pairs = _load_qa_pairs(col)
    print(f"Total QA pairs loaded: {len(qa_pairs)}")

    score_qa(qa_pairs)
    print_stats()


if __name__ == "__main__":
    main()