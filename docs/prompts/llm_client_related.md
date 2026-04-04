# 1. GENERATE (USE LLM TO GENERATE TRAINING DATA AND FINETUNE)
```
"""
llm_generate.py
Input:  query (str), context (str)
Output: LLM response (str) — answer with citation numbers

Dependencies: openai, python-dotenv
"""

# --- Config (.env) ---
# OPENAI_API_KEY = "..."
# OPENAI_BASE_URL = "..."
# OPENAI_MODEL_NAME = "..."

# --- Prompt ---
# LLM_GENERATE_PROMPT:
LLM_CHAT_PROMPT = """
### Context
{context}

### Task
You are a specialized Q&A assistant for the Tesla Model Y User Manual. Using the information provided in the Context section, answer the following question: "{query}".

### Guidelines
1. **Accuracy**: Your answer must be precise and the sentences should flow naturally.
2. **Format**: Your output MUST strictly follow this format:
   {{Answer}} [{{Citation Number 1}}, {{Citation Number 2}}, ...]
3. **Grounding**: If the answer cannot be found in the provided Context, state "No Answer." Do not hallucinate or add any external information.
"""

# --- Function ---
def request_chat(query: str, context: str) -> str:
    # format prompt with query and context
    # call OpenAI API
    # return response string
```


# 2. CLEAN (USE LLM TO CLEAN RAW PDF TEXT)
```
"""
llm_clean.py
Input:  List[Document] — raw pages from load_pdf()
Output: List[Document] — cleaned pages, same structure

Dependencies: openai, python-dotenv
"""

# --- Config (.env) ---
# OPENAI_API_KEY    = "..."
# OPENAI_BASE_URL   = "..."
# OPENAI_MODEL_NAME = "..."
# MAX_WORKERS       = 20

# --- Prompt ---
LLM_CLEAN_PROMPT = """
You are a document formatting assistant for automotive owner's manuals.
Clean and split the following text extracted from a PDF page.

### Step 1 — Clean
1. Fix broken line breaks caused by column wrapping: if a line does not end with .!?, merge it with the next line using a space.
2. Remove redundant whitespace and artifacts from PDF extraction.
3. Remove any HTML tags (e.g. <p>, <br>, <div>).

### Step 2 — Split
Insert <<<SPLIT>>> immediately BEFORE each new code-like identifier or heading,
not after. The delimiter must appear on its own line before the new unit begins.
A new semantic unit begins whenever the topic, subject, or focus clearly shifts.
This can be signaled by a new code-like identifier (e.g. alphanumeric labels at the start of a line)
or a new heading/title (e.g. a short capitalized line that introduces a new topic).

### Rules
- Do NOT split within a single alert entry under any circumstance.
- Do NOT split numbered or bulleted list items — a list is a single semantic unit.
- Do NOT insert <<<SPLIT>>> at the very beginning or very end of the output.
- Do NOT rephrase, summarize, reorder, add, or remove any content.
- Preserve the original wording exactly.

### Input
{}

### Output
"""

# --- Function ---
def request_llm_clean(docs: List[Document]) -> List[Document]:
    # NOTE: only process first 10 docs for validation
    # use ThreadPoolExecutor(MAX_WORKERS) for concurrent API calls
    # for each doc: format prompt, call API, return cleaned text
    # preserve original metadata
    # return List[Document] with cleaned page_content
```
