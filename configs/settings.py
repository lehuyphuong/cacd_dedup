"""
Central configuration for cacd-dedup.

Một project độc lập, tách riêng khỏi rag-bench-v4. Giữ nguyên 5
chunking strategies từ v4 để đánh giá, nhưng thay toàn bộ 5 filter
methods cũ (NoFilter/ExactNorm/MinHashLSH/Similarity/NERExact) bằng
một pipeline duy nhất: CACD (Cross-Attention Calibrated Deduplication).

CACD pipeline (4 stages, chỉ nhánh DROP — không Merge):
  Stage 0: Embedding          — all-MiniLM-L6-v2, 384-dim
  Stage 1: Coarse retrieval   — batch query Qdrant (HNSW), top-K candidates
  Stage 2: Cross-attention    — cross-encoder/ms-marco-MiniLM-L-6-v2,
                                 trích attention matrix, tính redundancy
                                 signal (max-alignment coverage)
  Stage 3: Decision           — calibrated probability + Bayes-optimal
                                 cutoff, chỉ nhánh DROP (Merge để dành
                                 cho experiment sau)

Evaluation metrics: giữ nguyên 4 metrics cũ từ v4 — Precision, Recall,
IoU, Index Size (chunk count + storage MB).
"""

from pathlib import Path

import torch

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT_DIR / "data"
RESULTS_DIR  = ROOT_DIR / "results"
QDRANT_PATH  = DATA_DIR / "qdrant_storage"
HEATMAP_DIR  = RESULTS_DIR / "heatmaps"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
QDRANT_PATH.mkdir(exist_ok=True)
HEATMAP_DIR.mkdir(exist_ok=True)

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET_NAME       = "rajpurkar/squad"
DATASET_SPLIT      = "validation"
# MAX_DOCUMENTS:       int | None = 500
# MAX_EVAL_QUESTIONS:  int | None = 200

MAX_DOCUMENTS:       int | None = None
MAX_EVAL_QUESTIONS:  int | None = None
# ── Embedding — all-MiniLM-L6-v2 (Stage 0) ───────────────────────────────────
EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_EMBED_DIM   = 384
# Batch size lớn hơn tận dụng GPU tốt hơn.
# GPU 8GB+: 512–1024. CPU: giữ 128.
EMBED_BATCH_SIZE = 512 if torch.cuda.is_available() else 128

# ── CACD — Stage 1 (Coarse retrieval) ────────────────────────────────────────
CACD_TOP_K_CANDIDATES = 5   # K ứng viên gần nhất lấy ra từ HNSW mỗi chunk

# ── CACD — Stage 2 (Cross-attention) ─────────────────────────────────────────
# Pretrained, KHÔNG fine-tune (theo quyết định của user). Model được chọn
# dựa trên kết quả research: cross-encoder/msmarco-MiniLM-L6-en-de-v1 là
# baseline phổ biến nhất trong literature (AugSBERT, nhiều paper rerank).
CACD_CROSS_ENCODER_MODEL = "cross-encoder/msmarco-MiniLM-L6-en-de-v1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── CACD — Stage 3 (Decision: Bayes-optimal cutoff) ──────────────────────────
# cutoff = cost_FP / (cost_FP + cost_FN)
#   cost_FP: chi phí khi NHẦM drop 1 chunk thật sự không trùng (mất thông tin)
#   cost_FN: chi phí khi NHẦM giữ 1 chunk thật sự trùng lặp (lãng phí storage)
#
# Mặc định đối xứng (cutoff=0.5). Tăng CACD_COST_FALSE_POSITIVE để hệ thống
# THẬN TRỌNG hơn khi drop (ưu tiên không mất thông tin hơn ưu tiên gọn nhẹ).
CACD_COST_FALSE_POSITIVE = 1.0
CACD_COST_FALSE_NEGATIVE = 1.0

# ── Vector store — Qdrant (embedded, no Docker) ──────────────────────────────
COLLECTION_PREFIX = "cacd_dedup"

# ── Retrieval (đánh giá cuối, top-k = 5 như v4) ──────────────────────────────
TOP_K = 5

# ── LLM (Ollama — optional, không dùng trong scope hiện tại) ────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL       = "mistral"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS  = 256

SYSTEM_PROMPT = """\
You are a precise question-answering assistant. Answer the question using \
ONLY the provided context passages. Be concise — answer in 1-3 sentences. \
If the answer is not in the context, respond with exactly: "I don't know."\
"""

# ── Chunking configurations (giữ nguyên 5 strategies từ v4) ─────────────────
#
# 10 chunking configs × 1 pipeline CACD = 10 configs tổng
#
# Strategy params:
#   chunk_size : target chunk size in characters (base unit)
#   overlap    : character overlap between chunks (0 for strategies that
#                don't use sliding windows)
#   extra      : strategy-specific overrides (optional)

def _make_configs() -> list[dict]:
    chunker_configs = [
        # ── FixedSize ≡ FixedToken (paper chunk_size=200,400) ───────────────
        {"strategy": "FixedSize",   "chunk_size": 200, "overlap": 0},
        {"strategy": "FixedSize",   "chunk_size": 400, "overlap": 0},

        # ── Recursive ≡ RecursiveToken (paper chunk_size=200,400) ────────────
        {"strategy": "RecursiveToken",   "chunk_size": 200, "overlap": 0},
        {"strategy": "RecursiveToken",   "chunk_size": 400, "overlap": 0},

        # ── Semantic ≡ ClusterSemantic (paper chunk_size=200,400) ─────────────
        {"strategy": "ClusterSemantic",    "chunk_size": 200, "overlap": 0,
         "extra": {"threshold_percentile": 95.0}},
        {"strategy": "ClusterSemantic",    "chunk_size": 400, "overlap": 0,
         "extra": {"threshold_percentile": 95.0}},

        # ── Overlapping (paper chunk_size=400/overlap=200, 800/overlap=400) ──
        {"strategy": "Overlapping", "chunk_size": 400, "overlap": 200},
        {"strategy": "Overlapping", "chunk_size": 800, "overlap": 400},

        # ── AdaptiveEntropy ─────────────────────────────────────────────────
        {"strategy": "AdaptiveEntropy",       "chunk_size": 300, "overlap": 0},
        {"strategy": "AdaptiveEntropy",       "chunk_size": 500, "overlap": 0},

        # ── AdaptiveSentenceLen ──────────────────────────────────────────────
        {"strategy": "AdaptiveSentenceLen",   "chunk_size": 4,   "overlap": 0,
         "extra": {"target_sentences": 4, "min_sentences": 2, "max_sentences": 8}},
        {"strategy": "AdaptiveSentenceLen",   "chunk_size": 6,   "overlap": 0,
         "extra": {"target_sentences": 6, "min_sentences": 3, "max_sentences": 12}},

        # ── HierarchicalParentChild ──────────────────────────────────────────
        {"strategy": "HierarchicalParentChild", "chunk_size": 200, "overlap": 0,
         "extra": {"parent_size": 600}},
        {"strategy": "HierarchicalParentChild", "chunk_size": 400, "overlap": 0,
         "extra": {"parent_size": 800}},

        # ── Contextual ───────────────────────────────────────────────────────
        {"strategy": "Contextual",            "chunk_size": 300, "overlap": 0},
        {"strategy": "Contextual",            "chunk_size": 500, "overlap": 0},

        # ── TopicBased ───────────────────────────────────────────────────────
        {"strategy": "TopicBased",            "chunk_size": 200, "overlap": 0,
         "extra": {"n_topics": 4, "min_chunk_size": 100}},
        {"strategy": "TopicBased",            "chunk_size": 400, "overlap": 0,
         "extra": {"n_topics": 6, "min_chunk_size": 200}},
    ]

    configs = []
    for chunker in chunker_configs:
        cfg = {
            "strategy":   chunker["strategy"],
            "chunk_size": chunker["chunk_size"],
            "overlap":    chunker["overlap"],
        }
        if "extra" in chunker:
            cfg["extra"] = chunker["extra"]
        configs.append(cfg)

    return configs


CHUNKING_CONFIGS = _make_configs()
