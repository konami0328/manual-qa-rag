# 1. BGE Reranker (Cross-encoder reranker)
```
"""
rerank_bge.py
Input:  query (str), candidates List[Document] from HybridRetriever
Output: List[Document] — filtered and reranked by relevance score

Dependencies: FlagEmbedding, langchain
"""

# --- Config (config.py) ---
# RERANKER_MODEL_PATH = "models/BAAI/bge-reranker-v2-m3"
# RERANKER_THRESHOLD  = 0.0   # drop chunks below this score

class Reranker:
    def __init__(self):
        # load FlagReranker(RERANKER_MODEL_PATH, use_fp16=True)

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        # build pairs: [(query, doc.page_content), ...]
        # compute_score(pairs) → List[float]
        # zip docs with scores
        # filter: score > RERANKER_THRESHOLD
        # sort by score descending
        # return List[Document] with score stored in metadata
```

# 2. Qwen Reranker with vllm (LLM-based reranker)
