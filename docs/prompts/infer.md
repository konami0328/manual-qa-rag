"""
infer.py
Input:  query (str) — from user input
Output: answer (str) — LLM response with citation numbers

Dependencies: retrieve_bm25, llm_generate
"""

# --- Config (config.py) ---
# TOPK = 5   # number of chunks to retrieve

# --- Pipeline ---

def infer(query: str) -> str:
    # 1. retrieve top-k chunks via BM25
    # 2. format context: "[1] chunk1\n[2] chunk2\n..."
    # 3. call request_chat(query, context)
    # 4. return response

def main():
    # load BM25 index from pkl (retrieve=True)
    # loop: input query → infer → print answer