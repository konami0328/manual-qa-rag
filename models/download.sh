# =============================================================================
# Embedding Models
# =============================================================================

# Primary: Milvus hybrid retrieval (dense + sparse)
modelscope download --model BAAI/bge-m3 --cache_dir=./

# Secondary: FAISS retrieval
modelscope download --model intfloat/e5-mistral-7b-instruct --cache_dir=./

# Semantic chunking clustering
modelscope download --model sentence-transformers/all-MiniLM-L6-v2 --cache_dir=./

# =============================================================================
# Reranker Models
# =============================================================================

# Cross-encoder reranker
modelscope download --model AI-ModelScope/bge-reranker-v2-m3 --cache_dir=./

# LLM-based reranker
modelscope download --model Qwen/Qwen3-Reranker-4B --cache_dir=./

# =============================================================================
# Generation Model
# =============================================================================

# LLM
modelscope download --model LLM-Research/Meta-Llama-3.1-8B-Instruct --cache_dir=./

# =============================================================================
# Evaluation Model
# =============================================================================

# Semantic similarity scoring
modelscope download --model sentence-transformers/all-mpnet-base-v2 --cache_dir=./