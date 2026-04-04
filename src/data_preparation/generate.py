import os
import json
import threading
import concurrent.futures
from tqdm import tqdm

from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.documents import Document

from config import MINIMAL_CHUNK_SIZE, QA_CKPT_PATH, MAX_WORKERS
from src.client.mongodb_config import MongoConfig

load_dotenv()

# --- Config ---
DEBUG = True
DEBUG_SIZE = 10

# --- Prompt ---
LLM_GENERATE_QA_PROMPT = """
You are creating a Q&A dataset for a Tesla Model Y owner's manual retrieval system.

Generate 5 realistic question-answer pairs based on the document below. These questions should reflect what actual car owners would ask.

**Question Requirements:**
- Real user questions: "How do I...", "What happens if...", "Can I...", "Where is...", "When should..." etc.
- At least ONE must require synthesizing multiple sentences or steps (skip if document is too short)
- NO meta-questions about document structure or location

**Answer Requirements:**
- Complete and self-contained (2-4 sentences)
- NO references to other sections or page numbers

**Output Format (JSON only, no markdown, no preamble):**
[
  {{
    "question": "",
    "answer": ""
  }}
]

Return [] if the text is any of:
- A component list or table of contents
- A status indicator description
- An index or reference list

**Document:**
<document>
{document}
</document>
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
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"LLM call failed after {max_retries} attempts: {e}")
                return None


def _load_checkpoint() -> set:
    """Load already-processed source_chunk_ids from checkpoint file."""
    seen = set()
    if os.path.exists(QA_CKPT_PATH):
        with open(QA_CKPT_PATH) as f:
            for line in f:
                item = json.loads(line)
                seen.add(item["source_chunk_id"])
    return seen


def generate_qa(chunks: list[Document]) -> None:
    """Generate QA pairs for each chunk concurrently, save to JSONL checkpoint."""
    seen      = _load_checkpoint()
    file_lock = threading.Lock()

    os.makedirs(os.path.dirname(QA_CKPT_PATH), exist_ok=True)

    to_process = [c for c in chunks if c.metadata["unique_id"] not in seen]
    print(f"Chunks to process: {len(to_process)} (skipping {len(seen)} already done)")

    def _process(chunk: Document):
        prompt   = LLM_GENERATE_QA_PROMPT.format(document=chunk.page_content)
        raw_resp = _call_llm(prompt)
        if raw_resp is None:
            return

        item = {
            "source_chunk_id": chunk.metadata["unique_id"],
            "page":            chunk.metadata["page"],
            "raw_resp":        raw_resp,
        }

        with file_lock:
            with open(QA_CKPT_PATH, "a") as f:
                f.write(json.dumps(item) + "\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm(executor.map(_process, to_process), total=len(to_process)))


def main():
    col    = MongoConfig.get_collection("manual_text")
    chunks = [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in col.find()
        if len(d["page_content"].split()) >= MINIMAL_CHUNK_SIZE
    ]
    print(f"Total chunks after filter: {len(chunks)}")

    if DEBUG:
        # import random
        # random.seed(42)
        # chunks = random.sample(chunks, DEBUG_SIZE)
        chunks = chunks[:10]
        print(f"DEBUG mode: processing {DEBUG_SIZE} random chunks")

    generate_qa(chunks)
    print(f"Done. Saved to {QA_CKPT_PATH}")


if __name__ == "__main__":
    main()