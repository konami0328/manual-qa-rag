# RAG Pipeline Development Log

## 1. Walking Skeleton
Minimal end-to-end pipeline to validate system connectivity:
- **Parse**: `RecursiveCharacterTextSplitter`, text only (images ignored)
- **Retrieve**: BM25 only
- **Generate**: LLaMA 3.1 8B (no fine-tuning)

---

## 2. Parse Optimization

### Analysis Method
After each change, run `experiments/parse_analysis.py` to dump all chunks to `experiments/chunk_dump_<date>.txt` and inspect visually.

---

### Issues & Fixes (in order)

| # | Issue | Fix | Status |
|---|-------|-----|--------|
| 1 | Cover/back matter included (TOC, index, etc.) | Filter by `PAGE_START` / `PAGE_END` in `load_pdf()` | ✅ |
| 2 | Header/footer text leaking into chunks | Crop page rect by `PAGE_CROP_TOP` / `PAGE_CROP_BOTTOM` via fitz | ✅ |
| 3 | Dirty formatting (`\n` artifacts from two-column PDF layout) | TRY 1: regex (newline preceded by `.!?` + followed by uppercase) — partially effective. TRY 2: LLM-based cleaning (`llm_clean.py`, `temperature=0`) ✅ | ✅ |
| 4 | Chunks mixing content from different sections | See detailed notes below | ✅ |
| 5 | Cross-page splits | See detailed notes below | 🔲 |

---

### Issue 4: Chunks Mixing Content from Different Sections

**Problem:**
`RecursiveCharacterTextSplitter` splits purely by token count with no awareness of document structure. Results in chunks that mix unrelated content (e.g. a safety warning merged with an unrelated operation step, or two separate alert entries in one chunk).

**Attempted approaches:**

**TRY 1 — Semantic chunking** (`bge-m3` embeddings + cosine similarity breakpoints)
- Split pages by `\n\n` into paragraphs, compute embedding similarity between adjacent pairs, split at low-similarity boundaries
- Problem: adjacent entries share similar embeddings regardless of being different topics (e.g. two charging error alerts). Similarity score alone cannot detect structural boundaries. Also adds time per full run.
- Result: ❌ not reliable enough

**TRY 2 — LLM-based split** (update `llm_clean` prompt)
- `LLM_CLEAN_PROMPT` asks the LLM to clean the text AND insert `<<<SPLIT>>>` delimiter before each new semantic unit in a single pass
- `chunk_index.py` then simply splits on `<<<SPLIT>>>` — no embedding logic needed
- Why LLM over rules/regex: the manual contains many section types (alert entries, feature descriptions, numbered instructions, spec tables, warnings) with inconsistent structural signals. Rules cannot cover all cases reliably. LLM reads structure like a human.
- Result: ✅

**Prompt design decisions:**
- Semantic unit defined as: topic/subject/focus clearly shifts, signaled by a new code-like identifier (e.g. `PCS_a073`) OR a new heading/title
- Delimiter chose as `<<<SPLIT>>>` not `---SPLIT---`: LLM confused `---` with markdown horizontal rule and abbreviated the delimiter
- Added rule "do not split numbered or bulleted lists": LLM was treating each list item as a separate unit
- `temperature=0` for consistency across runs

**Pipeline after fix:**
```
load_pdf() → llm_clean_and_split() → pickle → split on <<<SPLIT>>> → save() → MongoDB
```

---

### Issue 5: Cross-Page Splits

**Problem:**
`load_pdf()` processes one page at a time. An alert entry that starts on page N and ends on page N+1 becomes two separate `Document` objects before any LLM or chunking step runs. No prompt or chunking strategy can fix a problem that originates at the parsing stage.

Observed example: `PCS_a073` header on page 287, full body on page 288 — two separate chunks.

**Investigated approaches:**

| Approach | Verdict |
|----------|---------|
| Hard rules (detect incomplete page by missing `What to do:` section) | ❌ Unreliable — page can end after alert title with no detectable incomplete sentence |
| LLM merge (detect and merge incomplete pages before cleaning) | ✅ Correct but adds pipeline complexity and an extra LLM call per page |
| Overlapping page pairs `[N, N+1]`, `[N+1, N+2]` | ❌ Shifts the problem rather than solving it; also doubles collection size |
| Increase `TOPK` | ✅ Pragmatic mitigation — BM25 will likely surface both halves for a relevant query |

**Decision: defer.**
TOPK=5 retrieval likely surfaces both halves of a split alert together in context. Validate with targeted queries on known split alerts before investing in a parse-level fix. Revisit only if answer quality is measurably degraded.

**If fix becomes necessary:** implement LLM merge as a `merge_pages()` step in `parse.py`, storing page range `[N, N+1]` in metadata instead of a single page number to avoid losing source traceability.

---

### Refactor
`parse.py` split into two scripts for independent re-runs:
- `parse.py`: `load_pdf()` → `llm_clean_and_split()` → pickle
- `chunk_index.py`: load pickle → split on `<<<SPLIT>>>` → `save()` → MongoDB

---

## 3. Retrieve Optimization

### Overview

BM25-only retrieval surfaces correct results but is purely keyword-based — semantically equivalent queries can miss relevant chunks. Upgraded to a three-signal hybrid pipeline.

---

### Architecture

```
query
  ├── BGE-M3 dense  → top-k ──┐
  ├── BGE-M3 sparse → top-k ──┤ RRF fusion (internal, Milvus)
  │                            └──────────────────────────────
  │                                         ↓
  │                                  dedup by unique_id
  │                                         ↓
  └── BM25          → top-k ────────── union merge
                                             ↓
                                    candidate pool (~14-20 chunks)
                                             ↓
                                         reranker
                                             ↓
                                        LLM generate
```

---

### Retriever: BGE-M3 (`retrieve_bge.py`)

**Why bge-m3 over separate embedder + BM25:**
`bge-m3` is a three-in-one model producing dense, sparse, and ColBERT vectors in a single forward pass. Using it for both dense and sparse eliminates the need for a separate embedder.

**Dense vs sparse metric types:**
- Dense → `COSINE`: bge-m3 dense vectors are not normalized by default
- Sparse → `IP`: sparse vectors are non-negative weights, cosine is not meaningful here. **Not tunable — IP is correct for sparse.**

**ColBERT excluded:**
Requires N vectors per doc (one per token) vs 1 for dense — ~50x storage increase. Milvus-lite has limited ColBERT support. Diminishing returns for single-domain manual corpus. Revisit if quality remains poor after reranker.

**`encode_queries` vs `encode`:**
bge-m3 applies an internal query prefix during `encode_queries()` — must use this for queries, `encode()` for documents. Using wrong method degrades retrieval quality.

**Index persistence:**
Collection build skipped if Milvus collection already exists (`force_rebuild=False`). Pass `force_rebuild=True` explicitly after re-chunking to keep index in sync with MongoDB.

**Batch size:** `BGE_BATCH_SIZE=32` — affects build speed and GPU memory only, not retrieval quality. Tune only if OOM.

---

### Fusion: RRF

**Why RRF over WeightedRanker:**
RRF is rank-based — no score normalization needed across different scales (BM25 scores vs cosine similarity vs IP). `WeightedRanker` requires score-scale calibration. RRF is the safer default.

**`k=60`:** Standard constant from original RRF paper. Results not sensitive to this value. Not worth tuning.

**Dense:sparse ratio:** Implicitly 1:1 with RRF. To control ratio explicitly, switch to `WeightedRanker(sparse_weight, dense_weight)` — defer until eval data exists.

---

### Hybrid: BM25 Union (`retrieve_hybrid.py`)

**Why BM25 as union instead of third RRF signal:**
BGE sparse already covers most of BM25's lexical matching (learned weights vs frequency-based). BM25 is added as a **safety net** — guaranteed inclusion of exact keyword matches that BGE sparse might miss, regardless of topk size. Unioning after RRF ensures BM25 results are never dropped by rank competition.

**Edge case — cross-page split (Issue 5 deferred):**
Page 45 (steps 1-3) and page 46 (step 4) are separate chunks due to page boundary. BGE hybrid search alone missed page 46 — dense embedding of a short single-step chunk is less representative, gets pushed down by longer semantically richer chunks. BM25 union recovers it via exact keyword match ("shoulder", "anchor", "button"). Validated with query "How to Adjust the Shoulder Anchor Height" — page 46 now appears in candidate pool.

**Dedup order:** BGE results first, BM25 appended. BGE-ranked order is preserved for the reranker; BM25-only chunks appended at end.

---

### Decisions deferred

| Decision | Reason |
|---|---|
| WeightedRanker sparse:dense ratio | No eval data yet to justify tuning |
| ColBERT retrieval | Storage cost, milvus-lite support, diminishing returns |
| Query decomposition for multi-faceted queries | Not in current query set; implement if needed |
| Score threshold on hybrid search | RRF scores not interpretable as thresholds; handled by reranker instead |
| Issue 5 cross-page merge | Requires `merge_pages()` in `parse.py`; deferred, mitigated by BM25 union |