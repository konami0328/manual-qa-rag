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
| 4 | Chunks mixing content from different sections | Semantic chunking | ✅ |
| 5 | Cross-page sentence splits | Overlap as mitigation; true fix requires cross-page context (TODO) | 🔲 |

### Refactor
`parse.py` split into two scripts:
- `parse.py`: `load_pdf()` → `llm_clean()` → pickle
- `chunk_index.py`: load pickle → `chunk()` → `save()` → MongoDB

---

## Issue 4: Chunks mixing content from different sections

### Problem
RecursiveCharacterTextSplitter splits purely by token count, ignoring semantic boundaries.
Results in chunks that mix unrelated content (e.g., safety warnings + operation steps).

### Solution: Semantic Chunking

**Approach:**
- Split by paragraph boundaries (`\n\n`) first to preserve document structure
- Compute embedding similarity (cosine) between adjacent paragraphs using bge-m3
- Split at low-similarity boundaries (eg., below 95th percentile) where topic shifts occur
- Constraint: max 5 chunks per page to avoid over-fragmentation

**Why this works:**
- Leverages cleaned text with preserved paragraph breaks
- Percentile-based threshold adapts to similarity distribution per page
- Respects natural topic boundaries rather than arbitrary token limits

**Implementation:**
- Model: BAAI/bge-m3 (local, ~560M params, ~2-3GB VRAM)
- Breakpoint: `percentile=95` (conservative, tune to lower if over-splitting)
- Merging: if chunks > 5, iteratively merge most similar neighbors

**Trade-offs:**
- Slower than token-based splitting (one-time embedding cost ~2-3 min on GPU)
- Variable chunk sizes (semantic boundaries ≠ fixed token length)
- May produce chunks >1000 tokens if page contains single cohesive topic

### Evaluation (Deferred)

**Original plan:** Compare Recursive vs Semantic chunking with Hit@3/MRR on 20% pages test data

**Decision: Skip baseline comparison**
- Different chunking methods → different chunk granularity → ambiguous ground truth
- Large chunks naturally have higher hit rates but more noise
- Fair comparison requires end-to-end QA evaluation (too complex at this stage)

**Current approach:**
- Implement semantic chunking directly
- Monitor retrieval quality via `infer.py` usage
- Revisit if Hit@3 drops significantly in production

**Status:** ✅ Implemented in `chunk_index.py`

Finetune the percentile
```
percentile=95:
════════════════════════════════════════════════════════════
                        BASIC STATS
════════════════════════════════════════════════════════════
Total docs   : 409
Avg words    : 348
Median words : 334
Max words    : 942
Min words    : 12
Pages        : 5 ~ 313

percentile=75:
════════════════════════════════════════════════════════════
                        BASIC STATS
════════════════════════════════════════════════════════════
Total docs   : 462
Avg words    : 308
Median words : 283
Max words    : 942
Min words    : 1
Pages        : 5 ~ 313

BAD EXAMPLE:
============================================================
[426/462]  page=283  words=446
────────────────────────────────────────────────────────────
ICR_a137
Cabin occupancy radar obstructed
Remove obstruction near the dome lights
What this alert means:
The In-Cabin Radar is currently blocked while all the doors are closed, which prevents it from operating correctly. This blockage is detected when all doors are closed and can occur due to:
• Objects attached to the headliner above the rear view mirror.
• Large objects in the cabin near the front of the vehicle.
A blocked In-Cabin Radar can lead to inaccurate readings. This may affect functions such as occupant detection, auto parking brake, and vehicle display status.
What to do:
To resolve this issue, relocate any obstruction away from the In-Cabin Radar. Once the obstruction is cleared, the blockage alert will automatically disappear.
If this alert continues to appear after removing any obstructions, please schedule a service appointment at your earliest convenience.
PCS_a016
Cannot charge - Poor grid power quality possible
Retry / Try other charge location or Supercharging
What this alert means:
Charging has stopped due to a condition that prevents your vehicle from charging with AC power. DC fast charging / Supercharging should still function as expected.
This may be due to power supply disturbances caused by the external charging equipment or by the electrical power grid. In some cases, this condition may be the result of using nearby electric devices that draw a lot of power.
If these possible causes can be ruled out, then a condition with your vehicle itself may also be affecting AC charging.
What to do:
If this alert is accompanied by another alert that specifies the condition affecting AC charging, start by investigating that alert.
Further troubleshooting tips based on equipment type:
• If using a Mobile Connector, try charging the vehicle with a different wall outlet.
◦ If the vehicle starts to charge, the issue was likely with the original wall outlet.
◦ If the vehicle still does not charge, the issue may be with the Mobile Connector.
• If using a Wall Connector, try charging the vehicle with different charging equipment like a Mobile Connector powered by a separate wall outlet.
◦ If the vehicle starts to charge, the issue was likely with the Wall Connector.
If the issue is with the original wall outlet or the Wall Connector, contact an electrician to inspect the wiring connection.
You can also try charging your vehicle using a Tesla Supercharger or Destination Charging location, all of which can be located through the map on your vehicle's touchscreen display. See Maps and Navigation on page 169 for more details.
If this alert persists when attempting to charge at multiple locations and with different charging equipment, it is recommended that you schedule service.
```

Ideally, page 283 should be splitted into 2 chunks?

I think the issue happened during the cleaning phase. We should update the original prompt to let the LLM insert \n\n wherever it feels a break is needed. That way, during semantic chunking, we can use these \n\n paragraphs as the base unit—each paragraph becomes a natural chunk unless the semantics are closely related.