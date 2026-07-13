# cacd-dedup

**Cross-Attention Calibrated Deduplication (CACD)** — a chunk-filtering method for RAG that replaces fixed cosine-similarity thresholds with a cross-encoder pipeline producing a calibrated decision signal.

Standalone project. Keeps the same chunking strategies used to evaluate the earlier baseline filters (NoFilter / ExactNorm / MinHashLSH / Similarity / NERExact, in the companion [rag_bench](https://github.com/lehuyphuong/rag_bench) repository) but replaces all five of those filter methods with a single pipeline: **CACD**.

---

## 1. Problem CACD addresses

Cosine similarity on pooled embeddings reduces the "are these two chunks duplicates?" question to a single number compared against a fixed threshold (typically 0.8) with no principled justification. Chunks that look superficially similar (shared header text, shared named entities introduced by the chunking structure itself) can get flagged as "duplicates" even though their actual content differs — a failure mode referred to here as **false-redundancy collapse**.

CACD addresses this by:
1. Using a **cross-encoder** (joint encoding of both chunks, not two independent vectors) to preserve token-level detail all the way to the final comparison step.
2. Extracting the **attention matrix** instead of collapsing to a single similarity score, to identify exactly which parts of the two chunks correspond to each other.
3. Producing two complementary signals, a **calibrated duplicate probability** from the cross-encoder output and a **Novel Information Score (NIS)** derived from attention entropy, combined through a 3-zone decision rule plus a length-aware guard rather than a single hand-picked constant.
4. Deciding via **majority vote** across several retrieved candidates, so one misleading nearest neighbor cannot flip the decision on its own.

---

## 2. CACD Pipeline (3 stages)

```
New chunk
    |
    v
Stage 0 — Embedding
    all-MiniLM-L6-v2, 384-dim vector
    |
    v
Stage 1 — Coarse retrieval (in-memory, exact top-K)
    Retrieves the K nearest chunks already decided KEEP, from a growing
    in-memory pool of embeddings (not an external vector store)
    Complexity: O(pool_size) per query
    |
    v  (only K candidates, K=5 by default)
Stage 2 — Cross-attention redundancy scoring
    2a. Joint encoding: [CLS] new_chunk [SEP] candidate [SEP]
    2b. Extract attention matrix (last layer, averaged across heads)
    2c. Compute coverage signals (max-alignment, BERTScore-style)
    2d. Compute Novel Information Score (NIS) from attention B=>A entropy
    Model: cross-encoder/msmarco-MiniLM-L6-en-de-v1 (pretrained, no fine-tuning)
    Batched forward pass: many (chunk, candidate) pairs scored per GPU call.
    |
    v
Stage 3 — 3-zone decision + length-aware guard, majority vote across K
    prob >= PROB_HIGH                => DROP (unless length guard applies)
    prob <= PROB_LOW                 => KEEP
    PROB_LOW < prob < PROB_HIGH      => NIS decides (NIS < threshold => DROP)
    chunk longer than LENGTH_GUARD chars is protected unless NIS < NIS_FLOOR
```

A chunk voted DROP is discarded. Qdrant is touched once per chunking
configuration, in a single bulk upsert after the whole ingest run, purely
so the final kept set is available for the retrieval-quality evaluation
step; it plays no role in the Stage 1-3 decision itself.

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

## 4. Setup and usage

### 4.1 Setup

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The first run downloads:
- `sentence-transformers/all-MiniLM-L6-v2` (~80MB) — Stage 0
- `cross-encoder/msmarco-MiniLM-L6-en-de-v1` (~90MB) — Stage 2

### 4.2 Verify installation

```bash
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('embed OK:', m.encode(['test']).shape)"
python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; m = AutoModelForSequenceClassification.from_pretrained('cross-encoder/msmarco-MiniLM-L6-en-de-v1'); print('cross-encoder OK')"
python -c "from qdrant_client import QdrantClient; c = QdrantClient(':memory:'); print('qdrant OK')"
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

### 4.3 Run benchmark

```bash
# Quick debug (20 docs, 30 questions)
python scripts/benchmark.py --max-docs 20 --max-questions 30

# Single strategy
python scripts/benchmark.py --strategy Contextual --max-docs 50 --max-questions 50

# Single config
python scripts/benchmark.py --config "Contextual_300_0"

# Full benchmark (all 18 configs)
python scripts/benchmark.py

# Keep Qdrant collections after each config instead of deleting them
python scripts/benchmark.py --keep-collections

# Background run
nohup python scripts/benchmark.py > results/bench.log 2>&1 &
tail -f results/bench.log
```

### 4.4 Inspect results

```bash
# View the results CSV
column -t -s',' results/benchmark_results.csv | less -S

# View the audit log (drop/keep decision for each chunk, with P(duplicate) and NIS)
column -t -s',' results/audit_Contextual_300_0.csv | less -S
```

---

## 5. Output

### `results/benchmark_results.csv` — one row per config

| Column | Description |
|---|---|
| `config_name` | e.g. `Contextual_300_0` |
| `chunk_count_before_filter` / `chunk_count_after_filter` | Chunk count before/after CACD dedup |
| `filter_reduction_pct` | Percentage of chunks dropped |
| `ingest_time_s` | Total time: chunking + embedding + CACD (Stage 1-3) + upsert |
| `storage_mb` / `storage_du` | Qdrant collection disk size |
| `precision_raw` / `recall_raw` / `iou_raw` | Token metrics, raw mode |
| `precision_pre` / `recall_pre` / `iou_pre` | Token metrics, preprocessed mode |
| `avg_retrieval_ms` | Mean retrieval latency at evaluation time |
| `n_questions` | Number of questions evaluated |
| `cacd_prob_high` / `cacd_prob_low` / `cacd_nis_threshold` | Decision thresholds actually used for this run |

### `results/per_question_{config_name}.csv` — one row per evaluated question

`config_name`, `question`, `doc_id`, `precision_raw`, `recall_raw`, `iou_raw`, `precision_pre`, `recall_pre`, `iou_pre`, `retrieval_ms`.

### `results/audit_{config_name}.csv` — one row per chunk

Records the CACD decision for each chunk: `chunk_id`, `decision` (drop/keep), `reason` (which zone/guard triggered the decision), `best_p_duplicate`, `best_candidate_id`, `nis_b_given_a`, `coverage_a_to_b`, `coverage_b_to_a`, `redundancy_signal`, `prob_high`, `prob_low`, `nis_threshold`.

---

## 6. Project structure

```
cacd-dedup/
├── configs/
│   └── settings.py              # CACD parameters + 18 chunking configs
├── src/
│   ├── ingestion/
│   │   ├── loader.py            # SQuAD 1.1 loader
│   │   ├── chunker.py           # 9 chunking strategies
│   │   ├── embedder.py          # all-MiniLM-L6-v2 (Stage 0, GPU-aware)
│   │   └── vector_store.py      # Qdrant wrapper (final storage only)
│   ├── dedup/
│   │   ├── stage1_inmemory_retrieval.py  # Stage 1: in-memory top-K search
│   │   ├── stage2_cross_attention.py     # Stage 2: cross-encoder + attention (batched)
│   │   ├── calibration.py                # Bayes-optimal cutoff
│   │   └── stage3_decision.py            # Stage 3: decision rule + pipeline orchestration
│   ├── retrieval/
│   │   └── retriever.py         # Dense cosine retrieval (final evaluation)
│   ├── evaluation/
│   │   ├── metrics.py           # Precision/Recall/IoU
│   │   └── generator.py         # Answer generation (optional, unused by default)
│   └── utils/
│       └── logger.py
├── scripts/
│   ├── benchmark.py                      # Main CLI
│   └── experiment_model_comparison.py    # 37-model cross-encoder comparison
├── data/
│   └── qdrant_storage/          # Qdrant collections (embedded mode)
├── results/
│   ├── benchmark_results.csv
│   ├── per_question_*.csv
│   └── audit_*.csv
├── requirements.txt
└── README.md
```