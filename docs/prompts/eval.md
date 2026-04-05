## Evaluation

### Input
- `test.jsonl` — list of `{question, answer, unique_id}`
- MongoDB `manual_text` collection — load all docs to build retrievers

---

### Retrievers
| Name | Class | Notes |
|------|-------|-------|
| BM25 | `BM25Retriever` | keyword-based baseline |
| BGE | `BGERetriever` | dense + sparse RRF internally via Milvus |
| Hybrid | `HybridRetriever` | BGE + BM25 union, dedup by unique_id |
| Hybrid + Reranker | `HybridRetriever` + `Reranker` | reranked results treated as a ranked list, same k axis |

---

### Metrics
**Hit@k**
- 1 if ground truth `unique_id` appears in top-k results, else 0
- Averaged over all questions

```
top-k results: [chunk_A, chunk_truth, chunk_B, ...]
Hit@k = 1
```

**MRR (Mean Reciprocal Rank)**
- `1 / rank` if ground truth found, else 0
- Averaged over all questions
- Computed once per retriever (independent of k)

```
ground truth at rank 1 → 1/1 = 1.0
ground truth at rank 3 → 1/3 = 0.33
not found              → 0.0
```

---

### k sweep
- `EVAL_K_VALUES = [1, 3, 5, 10]`
- For each retriever: retrieve top-`max(k)` once, then slice to each k — avoids redundant retrieval calls
- For `Hybrid + Reranker`: retrieve top-`max(k)` candidates → rerank → slice to each k

---

### Output

**Printed table**
```
Retriever         Hit@1  Hit@3  Hit@5  Hit@10   MRR
BM25              0.42   0.61   0.68   0.74     0.51
BGE               0.55   0.72   0.78   0.83     0.64
Hybrid            0.57   0.74   0.80   0.85     0.67
Hybrid+Reranker   0.61   0.75   0.81   0.85     0.74
```

**CSV** — one row per (retriever, k)
```
retriever, k, hit@k, mrr
BM25, 1, 0.42, 0.51
BM25, 3, 0.61, 0.51
BM25, 5, 0.68, 0.51
BM25, 10, 0.74, 0.51
BGE, 1, 0.55, 0.64
...
```

---

### Pipeline
```
load test.json
load docs from MongoDB
init all 4 retrievers

for each question:
    for each retriever:
        retrieve top-max(k) results
        if Hybrid+Reranker: rerank
        for each k:
            compute Hit@k
        compute MRR

aggregate → print table → save CSV
```