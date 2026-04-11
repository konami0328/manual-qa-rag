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
RERANKER_THRESHOLD  = 0

# =============================================================================
# 5. Data Generation
# =============================================================================
MINIMAL_CHUNK_SIZE = 15      # words; skip short chunks in QA generation
MAX_WORKERS_DATA   = 20      # concurrent LLM calls for data generation
QA_CKPT_PATH       = os.path.join(ROOT, "data", "qa_pairs", "qa_raw.jsonl")
FILTER_PATH        = os.path.join(ROOT, "data", "qa_pairs", "qa_filtered.jsonl")
EXPAND_PATH        = os.path.join(ROOT, "data", "qa_pairs", "qa_expand.jsonl")
TRAIN_PATH         = os.path.join(ROOT, "data", "qa_pairs", "train.jsonl")
VAL_PATH           = os.path.join(ROOT, "data", "qa_pairs", "val.jsonl")
TEST_PATH          = os.path.join(ROOT, "data", "qa_pairs", "test.jsonl")
MIN_SCORE          = 4       # scoring threshold
NEGATIVE_COUNT     = 5000
TRAIN_RATIO        = 0.7
VAL_RATIO          = 0.2

# =============================================================================
# 6. Retrieval Evaluation
# =============================================================================
EVAL_K_VALUES         = [1, 5, 10, 15, 20]
EVAL_RETRIEVAL_PATH   = os.path.join(ROOT, "eval", "retrieval", "results", "retrieval_results.csv")

# =============================================================================
# 7.1. Reranker Fine-tuning
# =============================================================================
MINE_TOPK             = 20  # fetch more candidates to keep neg pool non-empty after adjacency filtering
RERANKER_TRAIN_PATH   = os.path.join(ROOT, "data", "reranker", "train_triplets.jsonl")
RERANKER_VAL_PATH     = os.path.join(ROOT, "data", "reranker", "val_triplets.jsonl")
RERANKER_CKPT_DIR     = os.path.join(ROOT, "data", "reranker", "ckpt")
RERANKER_BEST_CKPT    = os.path.join(ROOT, "data", "reranker", "ckpt", "epoch2_valloss_0.06965")

LR           = 2e-4
BATCH_SIZE   = 16
NUM_EPOCHS   = 3
LORA_RANK    = 16
LORA_ALPHA   = 32   # 2 * rank, standard empirical value
LORA_DROPOUT = 0.1

GENERATION_THRESHOLD = -1.0
GENERATION_TOPK = 5

# =============================================================================
# 7.2. LLM Fine-tuning
# =============================================================================
LLM_TRAIN_PATH        = os.path.join(ROOT, "data", "llm", "train_samples.jsonl")
LLM_VAL_PATH          = os.path.join(ROOT, "data", "llm", "val_samples.jsonl")
LLM_CKPT_DIR          = os.path.join(ROOT, "data", "llm", "ckpt")
LLM_QUANTIZED_PATH    = os.path.join(ROOT, "data", "llm", "quantized")
LLM_MODEL_PATH        = "models/LLM-Research/Meta-Llama-3.1-8B-Instruct"

LLM_LORA_RANK         = 16
LLM_LORA_ALPHA        = 32
LLM_LORA_DROPOUT      = 0.1
LLM_LR                = 2e-4
LLM_BATCH_SIZE        = 4
LLM_NUM_EPOCHS        = 2
LLM_MAX_LENGTH        = 2048

# =============================================================================
# 8. Generation
# =============================================================================
VLLM_BASE_URL    = "http://localhost:8000/v1"
VLLM_MODEL_NAME  = os.path.join(ROOT, "models/LLM-Research/Meta-Llama-3.1-8B-Instruct")
VLLM_MAX_WORKERS = 1