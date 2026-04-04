# 1. BM25 Retriever
```
"""
bm25_retriever.py
Input:  List[Document] (chunks from MongoDB) OR None (load from pkl)
Output: List[Document] (top-k relevant chunks for a given query)

Dependencies: rank_bm25, nltk, langchain, pickle
"""

# --- Config (config.py) ---
# BM25_PKL_FILE = os.path.join(ROOT, "data", "index", "bm25retriever.pkl")

# --- Tokenizer ---
# nltk English tokenizer + stopword filter
# from nltk.corpus import stopwords
# stopwords: nltk.corpus.stopwords.words("english")

# --- Pipeline ---

class BM25:
    def __init__(self, docs: List[Document]):
        # if BM25_PKL_FILE exists: load from pkl
        # else: build index from docs, save to BM25_PKL_FILE

    def _tokenize(self, text: str) -> List[str]:
        # lowercase + split
        # remove stopwords and punctuation
        # return List[str]

    def retrieve_topk(self, query: str, topk: int = 5) -> List[Document]:
        # tokenize query
        # return top-k Documents by BM25 score
```

# 2. BGE Retriever (Milvus)
```
"""
retrieve_bge.py
Input:  List[Document] (chunks from MongoDB) OR None (load from Milvus if exists)
Output: List[Document] (top-k relevant chunks for a given query)

Dependencies: FlagEmbedding, pymilvus, langchain
"""

# --- Config (config.py) ---
# EMBEDDING_MODEL_PATH = "models/BAAI/bge-m3"
# MILVUS_DB_FILE       = "data/index/milvus.db"
# MILVUS_COLLECTION    = "manual_index"
# DENSE_DIM            = 1024
# BGE_BATCH_SIZE       = 32

class BGERetriever:
    def __init__(self, docs: List[Document], force_rebuild: bool = False):
        # load bge-m3 model
        # connect to milvus-lite
        # if collection exists and not force_rebuild: skip build
        # else: _build(docs)

    def _build(self, docs: List[Document]) -> None:
        # drop collection if exists
        # create collection with dense + sparse schema
        # create indexes
        # encode all docs in batches → _encode()
        # insert into Milvus

    def _encode(self, texts: List[str]) -> dict:
        # batch encode with bge-m3
        # return {"dense": np.array, "sparse": list[dict]}

    def retrieve_topk(self, query: str, topk: int = 10) -> List[Document]:
        # encode query via encode_queries()
        # hybrid_search with RRFRanker
        # fetch page_content from MongoDB by unique_id
        # return List[Document]
```

# 3. Hybrid Retriever
```
"""
retrieve_hybrid.py
Input:  List[Document] (chunks from MongoDB)
Output: List[Document] (merged candidate pool for reranker)

Dependencies: retrieve_bm25, retrieve_bge
"""

# --- Config (config.py) ---
# TOPK = 10

class HybridRetriever:
    def __init__(self, docs: List[Document], force_rebuild: bool = False):
        # init BM25Retriever(docs)
        # init BGERetriever(docs, force_rebuild)

    def retrieve(self, query: str, topk: int = 10) -> List[Document]:
        # bge_results  = bge.retrieve_topk(query, topk)   ← dense+sparse RRF internally
        # bm25_results = bm25.retrieve_topk(query, topk)  ← keyword match
        # dedup + union by unique_id (bge first, bm25 appended)
        # return merged List[Document]
```