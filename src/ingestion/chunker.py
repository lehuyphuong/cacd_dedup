"""
Chunking strategies — rag-bench v4.

Five new strategies replacing the v3 Fixed/Recursive/ClusterSemantic set:

  AdaptiveEntropy      : chunk size adapts to local text entropy (Shannon H).
                         High-entropy (information-dense) text → smaller chunks.
                         Low-entropy (repetitive) text → larger chunks.

  AdaptiveSentenceLen  : chunk size adapts to mean sentence length in each
                         sliding window. Short-sentence passages → finer chunks.
                         Long-sentence passages → coarser chunks.

  HierarchicalParentChild : two-level chunking. Large "parent" chunks are split
                         into smaller "child" chunks. Both levels are indexed;
                         child chunks carry a parent_id for provenance.

  Contextual           : each chunk is prepended with a one-sentence document
                         context summary (title + position), so the embedding
                         captures both local and global document context.

  TopicBased           : sentences are embedded and k-means clustered into
                         topic groups; consecutive sentences in the same cluster
                         are merged into one chunk. Chunk boundaries are placed
                         where the topic label changes.

Each chunker returns list[dict]:
  {
    "chunk_id"   : str,
    "doc_id"     : str,
    "title"      : str,
    "text"       : str,
    "char_start" : int,
    "char_end"   : int,
    # optional extras
    "parent_id"  : str | None,   # HierarchicalParentChild child chunks only
    "level"      : str | None,   # "parent" | "child" | None
  }
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _make_chunk(
    doc: dict,
    chunk_id: str,
    text: str,
    char_start: int,
    char_end: int,
    parent_id: str | None = None,
    level: str | None = None,
) -> dict:
    return {
        "chunk_id":   chunk_id,
        "doc_id":     doc["doc_id"],
        "title":      doc["title"],
        "text":       text.strip(),
        "char_start": char_start,
        "char_end":   char_end,
        "parent_id":  parent_id,
        "level":      level,
    }


def _shannon_entropy(text: str) -> float:
    """Shannon entropy (bits per character) of a text string."""
    if not text:
        return 0.0
    counts = Counter(text)
    total  = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _find_offset(full_text: str, fragment: str, start: int) -> int:
    """Find fragment in full_text starting from start; fallback to start."""
    idx = full_text.find(fragment[:40], start)
    return idx if idx != -1 else start


# ── 1. AdaptiveEntropy ────────────────────────────────────────────────────────

def adaptive_entropy_chunker(
    doc: dict,
    base_size: int = 400,
    min_size: int = 150,
    max_size: int = 800,
) -> list[dict]:
    """
    Adaptive chunking based on Shannon entropy.

    Algorithm:
      1. Slide a probe window (base_size chars) across the text.
      2. Compute entropy H of the window.
      3. Normalise H to [0,1] relative to the text's global entropy range.
      4. Target chunk size = lerp(max_size, min_size, normalised_H):
           high entropy (dense info) → smaller chunks (closer to min_size)
           low  entropy (repetitive) → larger  chunks (closer to max_size)
      5. Emit a chunk at the nearest sentence boundary to the target end.
    """
    text      = doc["text"]
    sentences = _sentence_split(text)
    if not sentences:
        return []

    # Pre-compute per-sentence entropy
    sent_entropies = [_shannon_entropy(s) for s in sentences]
    global_min = min(sent_entropies) if sent_entropies else 0.0
    global_max = max(sent_entropies) if sent_entropies else 1.0
    h_range    = global_max - global_min or 1.0

    chunks: list[dict] = []
    buf_sents: list[str] = []
    buf_entropy: list[float] = []
    cursor = 0
    chunk_idx = 0

    for sent, h in zip(sentences, sent_entropies):
        buf_sents.append(sent)
        buf_entropy.append(h)

        mean_h      = sum(buf_entropy) / len(buf_entropy)
        norm_h      = (mean_h - global_min) / h_range          # 0 = low, 1 = high
        target_size = int(max_size - norm_h * (max_size - min_size))

        buf_text = " ".join(buf_sents)
        if len(buf_text) >= target_size:
            start = _find_offset(text, buf_sents[0], cursor)
            end   = start + len(buf_text)
            chunks.append(_make_chunk(
                doc, f"{doc['doc_id']}_{chunk_idx}",
                buf_text, start, end,
            ))
            cursor    = end
            chunk_idx += 1
            buf_sents   = []
            buf_entropy = []

    # Flush remaining
    if buf_sents:
        buf_text = " ".join(buf_sents)
        start    = _find_offset(text, buf_sents[0], cursor)
        end      = start + len(buf_text)
        chunks.append(_make_chunk(
            doc, f"{doc['doc_id']}_{chunk_idx}",
            buf_text, start, end,
        ))

    return [c for c in chunks if c["text"]]


# ── 2. AdaptiveSentenceLen ────────────────────────────────────────────────────

def adaptive_sentence_len_chunker(
    doc: dict,
    target_sentences: int = 5,
    min_sentences: int = 2,
    max_sentences: int = 10,
    short_threshold: int = 60,
    long_threshold:  int = 120,
) -> list[dict]:
    """
    Adaptive chunking based on mean sentence length.

    Algorithm:
      1. Compute mean character length of sentences in a rolling window.
      2. Short-sentence passages (< short_threshold chars/sent):
           use fewer sentences per chunk (min_sentences) — finer granularity.
      3. Long-sentence passages (> long_threshold chars/sent):
           use more sentences per chunk (max_sentences) — coarser granularity.
      4. Medium passages: target_sentences per chunk.
    """
    text      = doc["text"]
    sentences = _sentence_split(text)
    if not sentences:
        return []

    chunks:    list[dict] = []
    cursor     = 0
    chunk_idx  = 0
    i          = 0

    while i < len(sentences):
        # Look-ahead window to estimate local sentence length
        window     = sentences[i : i + target_sentences]
        mean_len   = sum(len(s) for s in window) / max(len(window), 1)

        if mean_len < short_threshold:
            n = min_sentences
        elif mean_len > long_threshold:
            n = max_sentences
        else:
            # Linear interpolation in the medium range
            ratio = (mean_len - short_threshold) / (long_threshold - short_threshold)
            n     = int(min_sentences + ratio * (max_sentences - min_sentences))

        group    = sentences[i : i + n]
        buf_text = " ".join(group)
        start    = _find_offset(text, group[0], cursor)
        end      = start + len(buf_text)

        chunks.append(_make_chunk(
            doc, f"{doc['doc_id']}_{chunk_idx}",
            buf_text, start, end,
        ))
        cursor    = end
        chunk_idx += 1
        i         += n

    return [c for c in chunks if c["text"]]


# ── 3. HierarchicalParentChild ────────────────────────────────────────────────

def hierarchical_parent_child_chunker(
    doc: dict,
    parent_size: int = 800,
    child_size:  int = 200,
) -> list[dict]:
    """
    Two-level hierarchical chunking.

    Algorithm:
      1. Split document into large "parent" chunks (parent_size chars).
      2. Split each parent into smaller "child" chunks (child_size chars).
      3. Both parents and children are returned and indexed.
         Children carry a parent_id field for provenance.

    Retrieval operates on the full mixed index (parents + children together).
    The parent_id allows downstream tracing of which parent a child came from.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text = doc["text"]

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size, chunk_overlap=0,
        separators=["\n\n", "\n", ". ", " ", ""],
        strip_whitespace=True,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size, chunk_overlap=0,
        separators=["\n\n", "\n", ". ", " ", ""],
        strip_whitespace=True,
    )

    parent_texts = parent_splitter.split_text(text)
    all_chunks: list[dict] = []
    p_cursor   = 0
    p_idx      = 0

    for p_text in parent_texts:
        p_start    = _find_offset(text, p_text, p_cursor)
        p_end      = p_start + len(p_text)
        parent_id  = f"{doc['doc_id']}_p{p_idx}"

        # Parent chunk
        all_chunks.append(_make_chunk(
            doc, parent_id, p_text, p_start, p_end,
            parent_id=None, level="parent",
        ))

        # Child chunks
        child_texts = child_splitter.split_text(p_text)
        c_cursor    = p_start
        c_idx       = 0
        for c_text in child_texts:
            c_start = _find_offset(p_text, c_text, c_cursor - p_start)
            c_start += p_start
            c_end    = c_start + len(c_text)
            all_chunks.append(_make_chunk(
                doc,
                f"{parent_id}_c{c_idx}",
                c_text, c_start, c_end,
                parent_id=parent_id, level="child",
            ))
            c_cursor  = c_end
            c_idx    += 1

        p_cursor = p_end
        p_idx   += 1

    return [c for c in all_chunks if c["text"]]


# ── 4. Contextual ─────────────────────────────────────────────────────────────

def contextual_chunker(
    doc: dict,
    chunk_size: int = 400,
    overlap:    int = 0,
) -> list[dict]:
    """
    Contextual chunking: each chunk is prepended with a document-level
    context header so its embedding captures both local and global meaning.

    Context header format:
      "[Context: {title} | Part {i+1}/{n}] "

    This mirrors the "Contextual Retrieval" approach (Anthropic, 2024)
    referenced in the paper Section 2.1.

    The stored "text" field contains the context-enriched string so that
    the Qdrant embedding reflects the enriched representation.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text = doc["text"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        strip_whitespace=True,
    )
    raw_chunks = splitter.split_text(text)
    n_chunks   = len(raw_chunks)

    chunks:  list[dict] = []
    cursor   = 0
    for i, raw in enumerate(raw_chunks):
        start   = _find_offset(text, raw, cursor)
        end     = start + len(raw)

        # Prepend context header
        header     = f"[Context: {doc['title']} | Part {i+1}/{n_chunks}] "
        enriched   = header + raw

        chunks.append(_make_chunk(
            doc, f"{doc['doc_id']}_{i}",
            enriched, start, end,
        ))
        cursor = end

    return [c for c in chunks if c["text"]]


# ── 5. TopicBased ─────────────────────────────────────────────────────────────

def topic_based_chunker(
    doc: dict,
    n_topics:   int = 5,
    min_chunk_size: int = 100,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[dict]:
    """
    Topic-based chunking via k-means clustering on sentence embeddings.

    Algorithm:
      1. Split document into sentences.
      2. Embed each sentence.
      3. K-means cluster sentences into n_topics topic groups.
      4. Walk sentences in order; emit a new chunk when the topic label changes
         OR when the accumulated chunk exceeds min_chunk_size.
      5. Adjacent chunks of the same topic are merged before final output.

    Fallback: if embed_fn is None or n_topics >= n_sentences, falls back to
    AdaptiveSentenceLen chunking.
    """
    from sklearn.cluster import KMeans

    text      = doc["text"]
    sentences = _sentence_split(text)

    if embed_fn is None:
        logger.warning(
            "TopicBased: embed_fn not provided — falling back to AdaptiveSentenceLen."
        )
        return adaptive_sentence_len_chunker(doc)

    if len(sentences) < 3:
        logger.debug(
            "TopicBased: doc '%s' has only %d sentence(s) — too short for "
            "k-means clustering, falling back to AdaptiveSentenceLen.",
            doc.get("doc_id", "?"), len(sentences),
        )
        return adaptive_sentence_len_chunker(doc)

    k = min(n_topics, len(sentences))

    # Embed sentences
    vecs = np.array(embed_fn(sentences), dtype=np.float32)

    # K-means topic clustering
    km     = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(vecs)   # (N,) integer topic label per sentence

    # Group consecutive sentences with the same label
    chunks:    list[dict] = []
    chunk_idx  = 0
    cursor     = 0
    buf_sents: list[str] = [sentences[0]]
    buf_label  = int(labels[0])

    for sent, label in zip(sentences[1:], labels[1:]):
        label = int(label)
        buf_text = " ".join(buf_sents)

        # New chunk when topic changes AND buffer is big enough
        if label != buf_label and len(buf_text) >= min_chunk_size:
            start = _find_offset(text, buf_sents[0], cursor)
            end   = start + len(buf_text)
            chunks.append(_make_chunk(
                doc, f"{doc['doc_id']}_{chunk_idx}",
                buf_text, start, end,
            ))
            cursor    = end
            chunk_idx += 1
            buf_sents  = [sent]
            buf_label  = label
        else:
            buf_sents.append(sent)

    # Flush
    if buf_sents:
        buf_text = " ".join(buf_sents)
        start    = _find_offset(text, buf_sents[0], cursor)
        end      = start + len(buf_text)
        chunks.append(_make_chunk(
            doc, f"{doc['doc_id']}_{chunk_idx}",
            buf_text, start, end,
        ))

    return [c for c in chunks if c["text"]]


# ── Classic chunkers (v3-compatible) ──────────────────────────────────────────

def fixed_size_chunker(
    doc: dict,
    chunk_size: int = 500,
    overlap: int = 0,
) -> list[dict]:
    """
    FixedSize — cắt text theo số ký tự cố định, không quan tâm ranh giới
    câu hay từ. Đây là phương pháp đơn giản nhất, dùng làm baseline.

    overlap > 0 → tạo sliding window (mỗi chunk kế tiếp bắt đầu lùi lại
    `overlap` ký tự so với điểm kết thúc của chunk trước).
    """
    text   = doc["text"]
    chunks = []
    idx    = 0
    start  = 0
    step   = max(1, chunk_size - overlap)

    while start < len(text):
        end  = min(start + chunk_size, len(text))
        span = text[start:end].strip()
        if span:
            chunks.append(_make_chunk(
                doc, f"{doc['doc_id']}_{idx}", span, start, end,
            ))
            idx += 1
        start += step

    return chunks


def recursive_chunker(
    doc: dict,
    chunk_size: int = 500,
    overlap: int = 0,
) -> list[dict]:
    """
    Recursive / RecursiveCharacterTextSplitter — cắt theo ưu tiên phân
    cấp: paragraph (\\n\\n) → sentence (. ! ?) → space → ký tự.

    Đây là chiến lược mặc định của LangChain và được dùng làm baseline
    chính trong paper Berdyugina et al. (2026, arXiv:2604.24334).
    """
    separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def _split(text: str, seps: list[str]) -> list[str]:
        if not seps:
            return [text[i:i + chunk_size] for i in range(0, len(text), max(1, chunk_size - overlap))]
        sep = seps[0]
        parts = text.split(sep) if sep else list(text)
        result: list[str] = []
        buf = ""
        for part in parts:
            candidate = (buf + sep + part).strip() if buf else part.strip()
            if len(candidate) <= chunk_size:
                buf = candidate
            else:
                if buf:
                    result.append(buf)
                if len(part) > chunk_size:
                    result.extend(_split(part, seps[1:]))
                    buf = ""
                else:
                    buf = part.strip()
        if buf:
            result.append(buf)
        return result

    text   = doc["text"]
    pieces = _split(text, separators)

    # Merge tiny pieces into neighbours and apply overlap
    merged: list[str] = []
    buf = ""
    for p in pieces:
        if not p.strip():
            continue
        candidate = (buf + " " + p).strip() if buf else p.strip()
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            if buf:
                merged.append(buf)
            buf = p.strip()
    if buf:
        merged.append(buf)

    # Apply overlap: each chunk starts `overlap` chars before the end of prev
    chunks  = []
    carried = ""
    for i, piece in enumerate(merged):
        text_with_carry = (carried + " " + piece).strip() if carried else piece
        chunk_text = text_with_carry[:chunk_size].strip()
        if chunk_text:
            start = text.find(chunk_text[:30]) if chunk_text[:30] in text else 0
            chunks.append(_make_chunk(
                doc, f"{doc['doc_id']}_{i}", chunk_text, start, start + len(chunk_text),
            ))
        # Carry-over for overlap
        if overlap > 0 and chunk_text:
            carried = chunk_text[-overlap:].strip()
        else:
            carried = ""

    return chunks


def semantic_chunker(
    doc: dict,
    threshold_percentile: float = 95.0,
    embed_fn: Callable | None = None,
) -> list[dict]:
    """
    SemanticChunker / ClusterSemantic (v3) — cắt tại điểm có cosine
    distance giữa 2 câu liên tiếp vượt ngưỡng percentile.

    Đây là chiến lược "sequential breakpoint detection" — chỉ nhìn vào
    khoảng cách ngữ nghĩa CỤC BỘ (local) giữa 2 câu kề nhau, khác với
    TopicBased dùng global clustering.

    Fallback: nếu không có embed_fn, dùng RecursiveChunker.
    """
    if embed_fn is None:
        logger.debug("SemanticChunker: no embed_fn, falling back to RecursiveChunker")
        return recursive_chunker(doc, chunk_size=500, overlap=0)

    sentences = _sentence_split(doc["text"])
    if len(sentences) < 3:
        return recursive_chunker(doc, chunk_size=500, overlap=0)

    # Embed từng câu
    vecs = np.array(embed_fn(sentences), dtype=np.float32)

    # Chuẩn hóa L2
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    vecs  = vecs / norms

    # Cosine distance giữa câu liên tiếp
    distances = [
        float(1.0 - np.dot(vecs[i], vecs[i + 1]))
        for i in range(len(vecs) - 1)
    ]

    threshold = float(np.percentile(distances, threshold_percentile))

    # Cắt tại điểm distance > threshold
    chunks  = []
    buf     = [sentences[0]]
    chunk_i = 0
    text    = doc["text"]
    cursor  = 0

    for i, dist in enumerate(distances):
        if dist >= threshold:
            chunk_text = " ".join(buf).strip()
            if chunk_text:
                start = _find_offset(text, buf[0], cursor)
                end   = start + len(chunk_text)
                chunks.append(_make_chunk(
                    doc, f"{doc['doc_id']}_{chunk_i}", chunk_text, start, end,
                ))
                cursor   = end
                chunk_i += 1
            buf = [sentences[i + 1]]
        else:
            buf.append(sentences[i + 1])

    # Flush
    if buf:
        chunk_text = " ".join(buf).strip()
        if chunk_text:
            start = _find_offset(text, buf[0], cursor)
            end   = start + len(chunk_text)
            chunks.append(_make_chunk(
                doc, f"{doc['doc_id']}_{chunk_i}", chunk_text, start, end,
            ))

    return chunks if chunks else recursive_chunker(doc, chunk_size=500, overlap=0)


def overlapping_chunker(
    doc: dict,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[dict]:
    """
    Overlapping / SlidingWindow — giống FixedSize nhưng luôn có overlap
    giữa các chunk liên tiếp, đảm bảo không bị mất thông tin tại ranh
    giới chunk. Đây là một trong các baseline phổ biến nhất trong RAG.

    Khác với FixedSize(overlap>0): Overlapping cắt tại ranh giới từ
    (không cắt giữa từ) để chunk text vẫn có nghĩa.

    overlap phải < chunk_size. Mặc định overlap = chunk_size // 5.
    """
    if overlap >= chunk_size:
        overlap = chunk_size // 5
        logger.warning("overlap >= chunk_size, clamped to %d", overlap)

    text   = doc["text"]
    words  = text.split()
    chunks = []
    idx    = 0

    # Ước lượng số từ mỗi chunk theo tỷ lệ chars/words trung bình
    avg_word_len = len(text) / max(len(words), 1)
    words_per_chunk = max(1, int(chunk_size / avg_word_len))
    words_overlap   = max(0, int(overlap   / avg_word_len))
    step = max(1, words_per_chunk - words_overlap)

    i = 0
    while i < len(words):
        chunk_words = words[i:i + words_per_chunk]
        chunk_text  = " ".join(chunk_words).strip()
        if chunk_text:
            start = text.find(chunk_text[:40]) if chunk_text[:40] in text else 0
            end   = start + len(chunk_text)
            chunks.append(_make_chunk(
                doc, f"{doc['doc_id']}_{idx}", chunk_text, start, end,
            ))
            idx += 1
        i += step

    return chunks


# ── Registry ──────────────────────────────────────────────────────────────────

def get_chunker(
    strategy:   str,
    chunk_size: int,
    overlap:    int,
    embed_fn:   Callable | None = None,
    extra:      dict | None = None,
):
    """Return a callable(doc) → list[chunk] for the given strategy."""
    extra = extra or {}

    if strategy == "FixedSize":
        return lambda doc: fixed_size_chunker(
            doc, chunk_size=chunk_size, overlap=overlap,
        )
    elif strategy == "Recursive":
        return lambda doc: recursive_chunker(
            doc, chunk_size=chunk_size, overlap=overlap,
        )
    elif strategy == "Semantic":
        return lambda doc, _fn=embed_fn: semantic_chunker(
            doc,
            threshold_percentile=extra.get("threshold_percentile", 95.0),
            embed_fn=_fn,
        )
    elif strategy == "Overlapping":
        return lambda doc: overlapping_chunker(
            doc, chunk_size=chunk_size, overlap=overlap,
        )
    elif strategy == "AdaptiveEntropy":
        return lambda doc: adaptive_entropy_chunker(
            doc,
            base_size=chunk_size,
            min_size=extra.get("min_size", max(100, chunk_size // 3)),
            max_size=extra.get("max_size", chunk_size * 2),
        )
    elif strategy == "AdaptiveSentenceLen":
        return lambda doc: adaptive_sentence_len_chunker(
            doc,
            target_sentences=extra.get("target_sentences", 5),
            min_sentences=extra.get("min_sentences", 2),
            max_sentences=extra.get("max_sentences", 10),
        )
    elif strategy == "HierarchicalParentChild":
        return lambda doc: hierarchical_parent_child_chunker(
            doc,
            parent_size=extra.get("parent_size", chunk_size * 2),
            child_size=chunk_size,
        )
    elif strategy == "Contextual":
        return lambda doc: contextual_chunker(
            doc, chunk_size=chunk_size, overlap=overlap,
        )
    elif strategy == "TopicBased":
        # Explicitly capture embed_fn in default argument to avoid closure bug
        return lambda doc, _fn=embed_fn: topic_based_chunker(
            doc,
            n_topics=extra.get("n_topics", 5),
            min_chunk_size=extra.get("min_chunk_size", chunk_size // 2),
            embed_fn=_fn,
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")


def chunk_documents(
    documents:  list[dict],
    strategy:   str,
    chunk_size: int,
    overlap:    int,
    embed_fn:   Callable | None = None,
    extra:      dict | None = None,
) -> tuple[list[dict], float]:
    """
    Chunk all documents with the given strategy.
    Returns (chunks, elapsed_seconds).
    """
    chunker = get_chunker(strategy, chunk_size, overlap,
                          embed_fn=embed_fn, extra=extra)
    t0         = time.perf_counter()
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunker(doc))
    elapsed = time.perf_counter() - t0

    logger.info(
        "Chunked %d docs → %d chunks [%s size=%d overlap=%d] in %.2fs",
        len(documents), len(all_chunks),
        strategy, chunk_size, overlap, elapsed,
    )
    return all_chunks, elapsed
