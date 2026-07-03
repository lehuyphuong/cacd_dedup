"""
CACD Benchmark — main entry point.

Chunk strategies : FixedSize, Recursive, Semantic, Overlapping,
                   AdaptiveEntropy, AdaptiveSentenceLen,
                   HierarchicalParentChild, Contextual, TopicBased
Dedup pipeline   : CACD (Cross-Attention Calibrated Deduplication)
                   Stage 0 Embedding => Stage 1 Coarse retrieval (HNSW) =>
                   Stage 2 Cross-attention scoring => Stage 3 Decision
                   (DROP branch only, Merge reserved for future work)
Eval metrics     : Precision, Recall, IoU, Index Size (chunk count + storage MB)

Usage:
    # Debug (small)
    python scripts/benchmark.py --max-docs 20 --max-questions 30

    # Single strategy
    python scripts/benchmark.py --strategy Contextual --max-docs 50 --max-questions 50

    # Single config
    python scripts/benchmark.py --config "Contextual_300_0"

    # Full benchmark (all 18 configs)
    python scripts/benchmark.py

    # Background
    nohup python scripts/benchmark.py > results/bench.log 2>&1 &
    tail -f results/bench.log
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.settings import CHUNKING_CONFIGS, EMBED_BATCH_SIZE, RESULTS_DIR, TOP_K
from src.dedup.stage3_decision import PROB_HIGH, PROB_LOW, NIS_DROP_THRESHOLD, run_cacd_dedup
from src.evaluation.metrics import compute_retrieval_metrics
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_chunks_batched, embed_texts
from src.ingestion.loader import load_squad
from src.ingestion.vector_store import (
    collection_name,
    collection_stats,
    delete_collection,
    ensure_collection,
)
from src.retrieval.retriever import retrieve
from src.utils.logger import configure_logging

logger = logging.getLogger(__name__)


# ── Config name ───────────────────────────────────────────────────────────────

def make_config_name(strategy: str, chunk_size: int, overlap: int) -> str:
    """Format: "{strategy}_{size}_{overlap}" — one CACD pipeline per config."""
    return f"{strategy}_{chunk_size}_{overlap}"


# ── CSV fields ────────────────────────────────────────────────────────────────

SUMMARY_FIELDS = [
    "config_name", "strategy", "chunk_size", "overlap",
    "chunk_count_before_filter",
    "chunk_count_after_filter",
    "filter_reduction_pct",
    "ingest_time_s",
    "storage_mb",
    "storage_du",
    "precision_raw", "recall_raw", "iou_raw",
    "precision_pre", "recall_pre", "iou_pre",
    "avg_retrieval_ms",
    "n_questions",
    "cacd_prob_high",
    "cacd_prob_low",
    "cacd_nis_threshold",
]

PER_Q_FIELDS = [
    "config_name", "question", "doc_id",
    "precision_raw", "recall_raw", "iou_raw",
    "precision_pre", "recall_pre", "iou_pre",
    "retrieval_ms",
]

AUDIT_FIELDS = [
    "chunk_id", "decision", "reason",
    "best_p_duplicate", "best_candidate_id",
    "nis_b_given_a", "coverage_a_to_b", "coverage_b_to_a",
    "redundancy_signal", "prob_high", "prob_low", "nis_threshold",
]


# ── Ingest (CACD) ─────────────────────────────────────────────────────────────

def run_ingest_cacd(
    documents:   list[dict],
    strategy:    str,
    chunk_size:  int,
    overlap:     int,
    config_name: str,
    embed_fn,
    extra:       dict | None = None,
) -> tuple[list[dict], list[dict], float, str, dict, list[dict]]:
    """
    Chunk => CACD dedup (Stage 1-3, drop only; chunks are inserted into
    Qdrant incrementally inside run_cacd_dedup) for one chunking config.

    Returns (chunks_before, chunks_after, ingest_time_s, cname, stats, audit_log).
    """
    t0 = time.perf_counter()

    # Step 1: Chunk
    chunks_raw, _ = chunk_documents(
        documents, strategy, chunk_size, overlap,
        embed_fn=embed_fn, extra=extra,
    )
    logger.info("  %d chunks before CACD dedup", len(chunks_raw))

    # Step 2: Embed all chunks (Stage 0)
    dense_vecs: list[list[float]] = []
    embedded_chunks: list[dict] = []
    for chunk, dv in embed_chunks_batched(chunks_raw, batch_size=EMBED_BATCH_SIZE):
        embedded_chunks.append(chunk)
        dense_vecs.append(dv)

    # Step 3: Create an empty collection; run_cacd_dedup inserts chunks
    # incrementally so each new chunk is checked against the already-indexed ones.
    cname = collection_name(strategy, chunk_size, overlap, "cacd")
    ensure_collection(cname, recreate=True)

    # Step 4: Run CACD (Stage 1 => 2 => 3 + Merge)
    # embed_fn is passed so that sentence-level merge can re-embed B_merged.
    kept_chunks, audit_log = run_cacd_dedup(
        embedded_chunks, dense_vecs, cname, config_name,
        embed_fn=embed_fn,
        save_heatmaps=True,
    )

    ingest_time = time.perf_counter() - t0
    stats = collection_stats(cname)

    logger.info(
        "  Ingest done: %.1fs | %d points | %.2f MB (%s)",
        ingest_time, stats["points_count"], stats["disk_mb"], stats["disk_size_du"],
    )
    return chunks_raw, kept_chunks, ingest_time, cname, stats, audit_log


# ── Evaluate ──────────────────────────────────────────────────────────────────

def run_eval(
    qa_pairs:   list[dict],
    documents:  list[dict],
    cname:      str,
    config_name: str,
) -> tuple[dict, list[dict]]:
    """
    Evaluate retrieval using Precision, Recall, IoU.

    Returns (summary, per_q_rows).
    """
    doc_lookup: dict[str, dict] = {d["doc_id"]: d for d in documents}

    acc: dict[str, list[float]] = {k: [] for k in [
        "precision_raw", "recall_raw", "iou_raw",
        "precision_pre", "recall_pre", "iou_pre",
        "retrieval_ms",
    ]}
    per_q_rows: list[dict] = []

    for i, qa in enumerate(qa_pairs):
        doc = doc_lookup.get(qa["doc_id"])
        if doc is None:
            continue

        reference_text    = doc["text"]
        retrieved, lat_ms = retrieve(qa["question"], cname, top_k=TOP_K)
        acc["retrieval_ms"].append(lat_ms)

        m_raw = compute_retrieval_metrics(reference_text, retrieved, mode="raw")
        acc["precision_raw"].append(m_raw["precision"])
        acc["recall_raw"].append(m_raw["recall"])
        acc["iou_raw"].append(m_raw["iou"])

        m_pre = compute_retrieval_metrics(reference_text, retrieved, mode="preprocessed")
        acc["precision_pre"].append(m_pre["precision"])
        acc["recall_pre"].append(m_pre["recall"])
        acc["iou_pre"].append(m_pre["iou"])

        per_q_rows.append({
            "config_name":   config_name,
            "question":      qa["question"],
            "doc_id":        qa["doc_id"],
            "precision_raw": m_raw["precision"],
            "recall_raw":    m_raw["recall"],
            "iou_raw":       m_raw["iou"],
            "precision_pre": m_pre["precision"],
            "recall_pre":    m_pre["recall"],
            "iou_pre":       m_pre["iou"],
            "retrieval_ms":  round(lat_ms, 2),
        })

        if (i + 1) % 20 == 0:
            logger.info(
                "  Evaluated %d/%d | recall_raw=%.3f | iou_raw=%.3f",
                i + 1, len(qa_pairs),
                sum(acc["recall_raw"]) / len(acc["recall_raw"]),
                sum(acc["iou_raw"])    / len(acc["iou_raw"]),
            )

    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

    summary = {k: avg(acc[k]) for k in [
        "precision_raw", "recall_raw", "iou_raw",
        "precision_pre", "recall_pre", "iou_pre",
    ]}
    summary["avg_retrieval_ms"] = avg(acc["retrieval_ms"])
    summary["n_questions"]      = len(per_q_rows)
    return summary, per_q_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(
        description="CACD dedup benchmark — cross-attention calibrated deduplication"
    )
    parser.add_argument("--max-docs",      type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument(
        "--strategy", type=str, default=None,
        choices=[
            "FixedSize", "Recursive", "Semantic", "Overlapping",
            "AdaptiveEntropy", "AdaptiveSentenceLen",
            "HierarchicalParentChild", "Contextual", "TopicBased",
        ],
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Run a single config by exact name, e.g. 'Contextual_300_0'",
    )
    parser.add_argument(
        "--keep-collections", action="store_true",
        help="Do not delete Qdrant collections after each config",
    )
    args = parser.parse_args()

    import configs.settings as S
    if args.max_docs:      S.MAX_DOCUMENTS       = args.max_docs
    if args.max_questions: S.MAX_EVAL_QUESTIONS  = args.max_questions

    documents, qa_pairs = load_squad()
    logger.info("Documents: %d | QA pairs: %d", len(documents), len(qa_pairs))
    logger.info(
        "CACD decision zones: prob_high=%.2f | prob_low=%.2f | nis_threshold=%.2f",
        PROB_HIGH, PROB_LOW, NIS_DROP_THRESHOLD,
    )

    embed_fn = lambda texts: embed_texts(texts)

    configs = list(CHUNKING_CONFIGS)

    if args.strategy:
        configs = [c for c in configs if c["strategy"] == args.strategy]

    if args.config:
        def _name(c):
            return make_config_name(c["strategy"], c["chunk_size"], c["overlap"])
        configs = [c for c in configs if _name(c) == args.config]
        if not configs:
            logger.error("Config '%s' not found.", args.config)
            sys.exit(1)

    logger.info("Running %d configs.", len(configs))

    summary_path = RESULTS_DIR / "benchmark_results.csv"
    is_new       = not summary_path.exists()
    summary_f    = open(summary_path, "a", newline="", encoding="utf-8")
    writer       = csv.DictWriter(summary_f, fieldnames=SUMMARY_FIELDS)
    if is_new:
        writer.writeheader()

    for cfg in configs:
        strategy   = cfg["strategy"]
        chunk_size = cfg["chunk_size"]
        overlap    = cfg["overlap"]
        extra      = cfg.get("extra", None)
        cname_str  = make_config_name(strategy, chunk_size, overlap)

        logger.info("=" * 65)
        logger.info("Config: %s", cname_str)
        logger.info("=" * 65)

        chunks_raw, kept_chunks, ingest_time, cname, stats, audit_log = run_ingest_cacd(
            documents, strategy, chunk_size, overlap,
            cname_str, embed_fn, extra=extra,
        )

        n_before      = len(chunks_raw)
        n_after       = len(kept_chunks)
        reduction_pct = round(100 * (n_before - n_after) / n_before, 2) if n_before else 0.0

        eval_summary, per_q = run_eval(qa_pairs, documents, cname, cname_str)

        # Per-question CSV
        per_q_path = RESULTS_DIR / f"per_question_{cname_str}.csv"
        with open(per_q_path, "w", newline="", encoding="utf-8") as f:
            pq_writer = csv.DictWriter(f, fieldnames=PER_Q_FIELDS)
            pq_writer.writeheader()
            pq_writer.writerows(per_q)

        # Audit log CSV — drop/keep decision for each chunk with CACD scores
        audit_path = RESULTS_DIR / f"audit_{cname_str}.csv"
        with open(audit_path, "w", newline="", encoding="utf-8") as f:
            a_writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
            a_writer.writeheader()
            for row in audit_log:
                a_writer.writerow({k: row.get(k, "") for k in AUDIT_FIELDS})

        row = {
            "config_name":               cname_str,
            "strategy":                  strategy,
            "chunk_size":                chunk_size,
            "overlap":                   overlap,
            "chunk_count_before_filter": n_before,
            "chunk_count_after_filter":  n_after,
            "filter_reduction_pct":      reduction_pct,
            "ingest_time_s":             round(ingest_time, 2),
            "storage_mb":                stats["disk_mb"],
            "storage_du":                stats["disk_size_du"],
            "cacd_prob_high":            round(PROB_HIGH, 4),
            "cacd_prob_low":             round(PROB_LOW, 4),
            "cacd_nis_threshold":        round(NIS_DROP_THRESHOLD, 4),
            **eval_summary,
        }
        writer.writerow(row)
        summary_f.flush()

        logger.info(
            "  DONE | chunks=%d=>%d (-%.1f%%) | %.1fs | %.2f MB (%s) | "
            "P=%.3f R=%.3f IoU=%.3f",
            n_before, n_after, reduction_pct,
            ingest_time, stats["disk_mb"], stats["disk_size_du"],
            eval_summary["precision_raw"],
            eval_summary["recall_raw"],
            eval_summary["iou_raw"],
        )

        if not args.keep_collections:
            delete_collection(cname)

    summary_f.close()
    logger.info("Results written to: %s", summary_path)
    logger.info("Heatmaps written to: %s", RESULTS_DIR / "heatmaps")


if __name__ == "__main__":
    main()
