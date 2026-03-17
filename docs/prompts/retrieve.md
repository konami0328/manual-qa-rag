# 1. BM25 Retriever
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


# 2. 