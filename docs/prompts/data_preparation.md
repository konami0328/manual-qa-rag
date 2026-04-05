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
- At least ONE must require synthesizing multiple sentences or steps (skip if document is too short)
- NO meta-questions about document structure or location

**Answer Requirements:**
- Complete and self-contained
- NO references to other sections or page numbers
- Based ONLY on the provided document — do not add any information not present in the text

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

# 2. Filter
```
# filter.py

"""
Input:  qa_raw.jsonl (source_chunk_id, page, raw_resp)
Output: qa_filtered.jsonl (ALL pairs with scores, nothing dropped)

Dependencies: openai, python-dotenv, pymongo
"""

# --- Config ---
# MAX_WORKERS    = 20
# MIN_SCORE      = 3          # for stats preview only, no actual dropping here
# DEBUG          = True
# DEBUG_SIZE     = 10
# QA_CKPT_PATH   = "data/qa_pairs/qa_raw.jsonl"
# FILTER_PATH    = "data/qa_pairs/qa_filtered.jsonl"

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

**A good answer must:**
- Be fully supported by the source document
- Not contain information absent from the document
- Cover all key information relevant to the question
- NOT reference page numbers or other sections

**Output (JSON only, no preamble):**
{"score": int, "reason": "one sentence max"}

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

# --- Pipeline ---

# Step 1 — Load qa_raw.jsonl
# for each line: parse raw_resp → List[{question, answer}]
# fetch source chunk from MongoDB by source_chunk_id
# if DEBUG: process first DEBUG_SIZE chunks only

# Step 2 — Score each QA pair (concurrent, checkpoint)
# def score_qa(qa_pairs) -> saves to FILTER_PATH (JSONL)
#   for each (question, answer, chunk): LLM → {score, reason}
#   save: {source_chunk_id, page, question, answer, score, reason}
#   skip if (source_chunk_id, question) already in checkpoint

# Step 3 — Print stats (no dropping)
# total QA pairs scored
# would keep (score >= MIN_SCORE)
# would drop (score < MIN_SCORE) + percentage

# --- Main ---
# def main():
#   Step 1 → Step 2 → Step 3

# if __name__ == "__main__":
#   main()

# --- Output format ---
# {
#   "source_chunk_id": "...",
#   "page":            45,
#   "question":        "...",
#   "answer":          "...",
#   "score":           4,
#   "reason":          "..."
# }
```

# 3. Expand
```

```