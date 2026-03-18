# RAG Pipeline Development Log

## Walking Skeleton
Minimal end-to-end pipeline to validate system connectivity:
- **Parse**: RecursiveCharacterTextSplitter, text only (images ignored)
- **Retrieve**: BM25 only
- **Generate**: LLaMA 3.1 8B (no fine-tuning)

---

## Parse Optimization

### Analysis
Chunk inspection output: `experiments/chunk_dump_<date>.txt`

### Issues & Fixes (in order)

| # | Issue | Fix | Status |
|---|-------|-----|--------|
| 1 | Cover/back matter included (TOC, index, etc.) | Filter by `PAGE_START` / `PAGE_END` in `load_pdf()` | ✅ |
| 2 | Header/footer text leaking into chunks | Crop page rect by `PAGE_CROP_TOP` / `PAGE_CROP_BOTTOM` via fitz | ✅ |
| 3 | Dirty formatting (`\n` artifacts from two-column PDF layout) | TRY 1: regex (newline preceded by `.!?` + followed by uppercase) — partially effective<br>**TRY 2: LLM-based cleaning with optimized prompt** ✅ | ✅ |
| 4.1 | Chunks mixing content from different sections | Semantic chunking (TODO) | 🔲 |
| 4.2 | Cross-page sentence splits | Overlap as mitigation; true fix requires cross-page context (TODO) | 🔲 |

### Refactor
`parse.py` split into two scripts:
- `parse.py`: `load_pdf()` → `llm_clean()` → pickle
- `chunk.py`: load pickle → `chunk()` → `save()` → MongoDB