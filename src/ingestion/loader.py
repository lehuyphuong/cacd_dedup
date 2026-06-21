"""
Load SQuAD 1.1 from HuggingFace datasets.

Each unique context passage becomes one "document" to be chunked and indexed.
Questions are used only at evaluation time.

Returns:
    documents : list[{doc_id, title, text}]
    qa_pairs  : list[{question, answers, doc_id, title}]
"""

from __future__ import annotations

import hashlib
import logging

from datasets import load_dataset

import configs.settings as settings

logger = logging.getLogger(__name__)


def _doc_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def load_squad() -> tuple[list[dict], list[dict]]:
    """
    Returns (documents, qa_pairs).

    documents : unique context passages, deduplicated by content hash.
    qa_pairs  : question + ground-truth answers + doc_id reference.

    Đọc MAX_DOCUMENTS / MAX_EVAL_QUESTIONS qua `settings.XXX` (không
    phải `from configs.settings import XXX`), để benchmark.py có thể
    override các giá trị này từ CLI (--max-docs/--max-questions) SAU
    khi module này đã được import — Python "from X import Y" copy giá
    trị tại thời điểm import, không tham chiếu động, nên override sau
    đó sẽ không có tác dụng nếu dùng cú pháp import cũ.
    """
    logger.info("Loading %s / %s ...", settings.DATASET_NAME, settings.DATASET_SPLIT)
    ds = load_dataset(settings.DATASET_NAME, split=settings.DATASET_SPLIT)

    seen_docs: dict[str, dict] = {}
    qa_pairs: list[dict] = []

    for row in ds:
        context = row["context"].strip()
        did = _doc_id(context)

        if did not in seen_docs:
            seen_docs[did] = {
                "doc_id": did,
                "title":  row["title"],
                "text":   context,
            }

        answers = [a for a in row["answers"]["text"] if a.strip()]
        if answers:
            qa_pairs.append({
                "question": row["question"].strip(),
                "answers":  answers,
                "doc_id":   did,
                "title":    row["title"],
            })

    documents = list(seen_docs.values())

    if settings.MAX_DOCUMENTS is not None:
        documents = documents[:settings.MAX_DOCUMENTS]
        kept_ids = {d["doc_id"] for d in documents}
        qa_pairs = [q for q in qa_pairs if q["doc_id"] in kept_ids]

    if settings.MAX_EVAL_QUESTIONS is not None:
        qa_pairs = qa_pairs[:settings.MAX_EVAL_QUESTIONS]

    logger.info(
        "Loaded %d unique documents, %d QA pairs.", len(documents), len(qa_pairs)
    )
    return documents, qa_pairs
