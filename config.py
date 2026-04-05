import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Root
# =============================================================================
ROOT = "/mnt/e/ML_Engineer/manual-qa"

# =============================================================================
# 1. Parse
# =============================================================================
PDF_FILE         = os.path.join(ROOT, "data", "Owners_Manual.pdf")
CLEAN_DOCS_PATH  = os.path.join(ROOT, "data", "clean_docs.pkl")
PAGE_START       = 5
PAGE_END         = 313
PAGE_CROP_TOP    = 55
PAGE_CROP_BOTTOM = 25
MAX_WORKERS      = 20

# =============================================================================
# 2. Chunk
# =============================================================================
CHUNK_SIZE   = 256   # unused after LLM-based chunking, kept for reference
CHUNK_OVERLAP = 50   # unused after LLM-based chunking, kept for reference

# =============================================================================
# 3. Retrieve
# =============================================================================
BM25_PKL_FILE     = os.path.join(ROOT, "data", "index", "bm25retriever.pkl")
EMBEDDING_MODEL_PATH = "models/BAAI/bge-m3"
MILVUS_DB_FILE    = os.path.join(ROOT, "data", "index", "milvus.db")
MILVUS_COLLECTION = "manual_index"
DENSE_DIM         = 1024
BGE_BATCH_SIZE    = 32
TOPK              = 10

# =============================================================================
# 4. Rerank
# =============================================================================
RERANKER_MODEL_PATH = "models/AI-ModelScope/bge-reranker-v2-m3"
RERANKER_THRESHOLD  = 0.1

# =============================================================================
# 5. Data Generation
# =============================================================================
MINIMAL_CHUNK_SIZE = 15      # words; skip short chunks in QA generation
MAX_WORKERS_DATA   = 20      # concurrent LLM calls for data generation
QA_CKPT_PATH       = os.path.join(ROOT, "data", "qa_pairs", "qa_raw.jsonl")
FILTER_PATH        = os.path.join(ROOT, "data", "qa_pairs", "qa_filtered.jsonl")
EXPAND_PATH        = os.path.join(ROOT, "data", "qa_pairs", "qa_expand.jsonl")
TRAIN_PATH         = os.path.join(ROOT, "data", "qa_pairs", "train.json")
TEST_PATH          = os.path.join(ROOT, "data", "qa_pairs", "test.json")
MIN_SCORE          = 3       # scoring threshold
NEGATIVE_COUNT     = 1000
TRAIN_RATIO        = 0.9
NEG_TRAIN_RATIO    = 0.95