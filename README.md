# cacd-dedup

**Cross-Attention Calibrated Deduplication (CACD)** — a chunk-filtering method for RAG that replaces fixed cosine-similarity thresholds with a cross-encoder pipeline producing a calibrated decision signal.

Standalone project, separated from `rag-bench-v4`. Keeps the same chunking strategies for evaluation, but replaces all 5 of the paper's filter methods (NoFilter / ExactNorm / MinHashLSH / Similarity / NERExact) with a single pipeline: **CACD**.

---

## 1. Problem CACD addresses

Cosine similarity on pooled embeddings reduces the "are these two chunks duplicates?" question to a single number compared against a fixed threshold (typically 0.8) with no principled justification. Benchmarking on `rag-bench-v4` (500 SQuAD documents) showed the concrete consequence:

| Filter | Strategy affected | Recall delta |
|---|---|---|
| NERExact | Contextual | **-0.143** |
| NERExact | TopicBased | **-0.131** |

Cause: chunks that look superficially similar (shared header text, shared named entities introduced by the chunking structure itself) get flagged as "duplicates" even though their actual content differs — a failure mode referred to here as **false redundancy**.

CACD addresses this by:
1. Using a **cross-encoder** (joint encoding of both chunks, not two independent vectors) to preserve token-level detail all the way to the final comparison step.
2. Extracting the **attention matrix** instead of collapsing to a single similarity score — this identifies exactly which parts of the two chunks actually correspond to each other.
3. Producing two complementary signals — a **calibrated duplicate probability** from the cross-encoder output, and a **Novel Information Score (NIS)** derived from attention entropy — combined through a 3-zone decision rule plus a length-aware guard, rather than a single hand-picked constant.

---

## 2. CACD Pipeline (4 stages)

```
New chunk
    |
    v
Stage 0 — Embedding
    all-MiniLM-L6-v2, 384-dim vector
    |
    v
Stage 1 — Coarse retrieval (HNSW, batch query)
    Retrieves top-K nearest neighbours already present in Qdrant
    (persistent index — not reset between ingest batches)
    Complexity: O(m log n)
    |
    v  (only K candidates, K=5 by default)
Stage 2 — Cross-attention redundancy scoring
    2a. Joint encoding: [CLS] new_chunk [SEP] candidate [SEP]
    2b. Extract attention matrix (last layer, averaged across heads)
    2c. Compute coverage signals (max-alignment, BERTScore-style)
    2d. Compute Novel Information Score (NIS) from attention B=>A entropy
    Model: cross-encoder/msmarco-MiniLM-L6-en-de-v1 (pretrained, no fine-tuning)
    Batched forward pass: all K candidates scored in a single GPU call.
    |
    v
Stage 3 — 3-zone decision + length-aware guard
    prob >= PROB_HIGH                => DROP (unless length guard applies)
    prob <= PROB_LOW                 => KEEP
    PROB_LOW < prob < PROB_HIGH      => NIS decides (NIS < threshold => DROP)
    chunk longer than LENGTH_GUARD chars is protected unless NIS < NIS_FLOOR
```

**Scope note**: this implementation only has a DROP branch — there is no Merge step. A chunk identified as redundant is excluded from the index entirely; partial-overlap cases (two chunks sharing roughly half their content) are resolved by keeping one and dropping the other rather than merging the non-overlapping portions. Comparing Drop vs. Merge is left for future work.

---

## 3. Chunking Strategies

| Strategy | Configs | Mechanism |
|---|---|---|
| FixedSize | size=200, size=400 | Fixed-length character windows |
| Recursive | size=200, size=400 | Paragraph => sentence => space => character split |
| Semantic | size=200, size=400 | Sequential breakpoint via cosine-distance percentile |
| Overlapping | size=400/overlap=200, size=800/overlap=400 | Sliding window, word-boundary aligned |
| AdaptiveEntropy | size=300, size=500 | Chunk size adapts to Shannon entropy |
| AdaptiveSentenceLen | target=4, target=6 sentences | Chunk size adapts to mean sentence length |
| HierarchicalParentChild | child=200/parent=600, child=400/parent=800 | Two-level parent+child, both indexed |
| Contextual | size=300, size=500 | Prepends header `[Context: title \| Part i/n]` |
| TopicBased | n_topics=4, n_topics=6 | K-means clustering on sentence embeddings |

=> **18 configs total** (9 strategies x 2 size variants), each running through the same CACD pipeline.

---

## 4. Evaluation Metrics

| Metric | Formula |
|---|---|
| Precision | \|Te ∩ Tr\| / \|Tr\| |
| Recall | \|Te ∩ Tr\| / \|Te\| |
| IoU | \|Te ∩ Tr\| / \|Te ∪ Tr\| |
| Index Size | chunk_count_after_filter + storage_mb |

Computed under two tokenization modes: `raw` (lowercase word tokens) and `preprocessed` (stopword removal + lemmatization via spaCy).

---

## 5. Setup and usage

### 5.1 Setup

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The first run downloads:
- `sentence-transformers/all-MiniLM-L6-v2` (~80MB) — Stage 0
- `cross-encoder/msmarco-MiniLM-L6-en-de-v1` (~90MB) — Stage 2

### 5.2 Verify installation

```bash
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('embed OK:', m.encode(['test']).shape)"
python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; m = AutoModelForSequenceClassification.from_pretrained('cross-encoder/msmarco-MiniLM-L6-en-de-v1'); print('cross-encoder OK')"
python -c "from qdrant_client import QdrantClient; c = QdrantClient(':memory:'); print('qdrant OK')"
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

### 5.3 Run benchmark

```bash
# Quick debug (20 docs, 30 questions)
python scripts/benchmark.py --max-docs 20 --max-questions 30

# Single strategy — Contextual is a good starting point: it is the case
# where the old NERExact baseline degraded Recall the most (-0.143)
python scripts/benchmark.py --strategy Contextual --max-docs 50 --max-questions 50

# Single config
python scripts/benchmark.py --config "Contextual_300_0"

# Full benchmark (all 18 configs)
python scripts/benchmark.py

# Background run
nohup python scripts/benchmark.py > results/bench.log 2>&1 &
tail -f results/bench.log
```

### 5.4 Inspect results

```bash
# View the results CSV
column -t -s',' results/benchmark_results.csv | less -S

# View the audit log (drop/keep decision for each chunk, with P(duplicate) and NIS)
column -t -s',' results/audit_Contextual_300_0.csv | less -S

# View attention heatmaps (disabled by default — see Section 8)
open results/heatmaps/Contextual_300_0/   # macOS
xdg-open results/heatmaps/Contextual_300_0/  # Linux
```

---

## 6. Output

### `results/benchmark_results.csv` — one row per config

| Column | Description |
|---|---|
| `config_name` | e.g. `Contextual_300_0` |
| `chunk_count_before_filter` / `chunk_count_after_filter` | Chunk count before/after CACD dedup |
| `filter_reduction_pct` | Percentage of chunks dropped |
| `ingest_time_s` | Total time: chunking + CACD (4 stages) + upsert |
| `storage_mb` / `storage_du` | Qdrant collection disk size |
| `precision_raw` / `recall_raw` / `iou_raw` | Token metrics, raw mode |
| `precision_pre` / `recall_pre` / `iou_pre` | Token metrics, preprocessed mode |
| `cacd_prob_high` / `cacd_prob_low` / `cacd_nis_threshold` | Decision thresholds actually used for this run |

### `results/audit_{config_name}.csv` — one row per chunk

Records the CACD decision for each chunk: `decision` (drop/keep), `reason` (which zone/guard triggered the decision), `best_p_duplicate`, `best_candidate_id`, `nis_b_given_a`, `coverage_a_to_b`, `coverage_b_to_a`, `redundancy_signal`.

### `results/heatmaps/{config_name}/*.png`

Attention-matrix heatmap between a new chunk and its nearest candidate — X axis is the candidate (B), Y axis is the new chunk (A); darker cells indicate higher attention weight. Disabled by default to reduce ingest time (see `save_heatmap` in `stage3_decision.py`, commented out at the call site); uncomment to re-enable, up to `max_heatmaps` per config.

---

## 7. Project structure

```
cacd-dedup/
├── configs/
│   └── settings.py              # CACD parameters + 18 chunking configs
├── src/
│   ├── ingestion/
│   │   ├── loader.py            # SQuAD 1.1 loader
│   │   ├── chunker.py           # 9 chunking strategies
│   │   ├── embedder.py          # all-MiniLM-L6-v2 (Stage 0, GPU-aware)
│   │   └── vector_store.py      # Qdrant embedded — persistent index
│   ├── dedup/
│   │   ├── stage1_coarse_retrieval.py   # Batch HNSW query
│   │   ├── stage2_cross_attention.py    # Cross-encoder + attention extraction (batched)
│   │   ├── calibration.py               # Bayes-optimal cutoff
│   │   └── stage3_decision.py           # 3-zone decision + length-aware guard
│   ├── retrieval/
│   │   └── retriever.py         # Dense cosine retrieval (final evaluation)
│   ├── evaluation/
│   │   └── metrics.py           # Precision/Recall/IoU
│   └── utils/
│       └── logger.py
├── scripts/
│   ├── benchmark.py                      # Main CLI
│   ├── experiment_calibration.py         # 5-pair sanity check (no benchmark dependency)
│   └── experiment_model_comparison.py    # 37-model cross-encoder comparison
├── data/
│   └── qdrant_storage/          # Qdrant collections
├── results/
│   ├── heatmaps/                # Attention heatmap PNGs per config (disabled by default)
│   ├── benchmark_results.csv
│   ├── per_question_*.csv
│   └── audit_*.csv
├── requirements.txt
└── README.md
```

---

## 8. Known limitations

- **DROP branch only, no Merge.** Partial-overlap cases (two chunks sharing roughly 50% of their content) are resolved binarily — one chunk is kept, the other dropped — risking loss of the unique information in the dropped chunk. This is a known limitation reserved for a future Merge experiment.
- **Cross-encoder is pretrained, not fine-tuned.** `msmarco-MiniLM-L6-en-de-v1` was trained for passage relevance ranking (MS MARCO), not binary duplicate classification. The model was selected via a 37-model comparison experiment (`scripts/experiment_model_comparison.py`) rather than fine-tuned on labeled duplicate pairs.
- **NIS_DROP_THRESHOLD saturates at 0.8 on SQuAD.** Values from 0.8 to 0.9 produce identical results because `LENGTH_GUARD` (300 characters) controls the majority of decisions once the probability threshold is satisfied. This has not been validated on datasets with different chunk-length distributions.
- **Heatmap generation is disabled by default** to reduce ingest time; re-enabling it (see Section 6) adds meaningful overhead per chunk during ingest.
- **Stage 1 (HNSW via Qdrant) runs on CPU only** — no GPU-accelerated ANN backend (e.g. cuVS/FAISS-GPU) is wired in, though the batch-query architecture would support one without changing the decision logic.
