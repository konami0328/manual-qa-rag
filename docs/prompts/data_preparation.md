# 1. Generate
```
"""
Input:  chunks from MongoDB (manual_text collection)
Output: QA pairs saved to QA_CKPT_PATH (JSONL)

Dependencies: openai, python-dotenv, langchain
"""

# --- Config ---
# MINIMAL_CHUNK_SIZE = 15          # words
# MAX_WORKERS        = 20
# DEBUG              = True
# DEBUG_SIZE         = 10
# QA_CKPT_PATH       = "data/qa_pairs/qa_raw.jsonl"

# --- Prompt ---

LLM_GENERATE_QA_PROMPT = """
You are creating a Q&A dataset for a Tesla Model Y owner's manual retrieval system.

Generate 5 realistic question-answer pairs based on the document below. These questions should reflect what actual car owners would ask.

**Question Requirements:**
- Real user questions: "How do I...", "What happens if...", "Can I...", "Where is...", "When should..." etc.
- NO meta-questions about document structure or location

**Answer Requirements:**
- Complete and self-contained (2-4 sentences)
- NO references to other sections or page numbers

**Output Format (JSON only, no markdown, no preamble):**
[
  {
    "question": "",
    "answer": ""
  }
]

Return [] if the text contains no actionable content.

**Document:**
<document>
{document}
</document>
"""

# --- Pipeline ---

# Step 1 — Load chunks from MongoDB
filter: len(page_content.split()) >= MINIMAL_CHUNK_SIZE
if DEBUG: chunks = chunks[:DEBUG_SIZE]

# Step 2 — Generate QA per chunk (concurrent, checkpoint)
def generate_qa(chunks) -> saves to QA_CKPT_PATH (JSONL)
  for each chunk: LLM → 5 (question, answer) pairs
  save: {source_chunk_id, page, raw_resp}
  skip if source_chunk_id already in checkpoint

# --- Main ---
def main():
  chunks = load from MongoDB + filter
  generate_qa(chunks)

if __name__ == "__main__":
  main()
```

# 2. Expand

# 3. 