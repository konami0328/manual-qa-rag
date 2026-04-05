# =============================================================================
# Embedding Models
# =============================================================================

# Primary: Milvus hybrid retrieval (dense + sparse)
modelscope download --model BAAI/bge-m3 --cache_dir=./

# =============================================================================
# Reranker Models
# =============================================================================

# Cross-encoder reranker
modelscope download --model AI-ModelScope/bge-reranker-v2-m3 --cache_dir=./

# LLM-based reranker
modelscope download --model Qwen/Qwen3-Reranker-4B --cache_dir=./

=============================================================================
Generation Model
=============================================================================

# LLM
modelscope download --model LLM-Research/Meta-Llama-3.1-8B-Instruct --cache_dir=./