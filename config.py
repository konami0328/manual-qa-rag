import os
from dotenv import load_dotenv

load_dotenv()

ROOT = "/mnt/e/ML_Engineer/manual-qa"
PDF_FILE = os.path.join(ROOT, "data", "Owners_Manual.pdf")
CLEAN_DOCS_PATH = os.path.join(ROOT, "data", "clean_docs.pkl")
BM25_PKL_FILE = os.path.join(ROOT, "data", "index", "bm25retriever.pkl")
EMBEDDING_MODEL_PATH = "models/BAAI/bge-m3"
MILVUS_DB_FILE       = "data/index/milvus.db"
MILVUS_COLLECTION    = "manual_index"
DENSE_DIM            = 1024
BGE_BATCH_SIZE       = 32

PAGE_START = 5
PAGE_END = 313
PAGE_CROP_TOP = 55
PAGE_CROP_BOTTOM = 25
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50
TOPK = 10

MAX_WORKERS = 20