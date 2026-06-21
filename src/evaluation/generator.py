"""
Answer generation via Ollama (Mistral 7B).
Used only for generation metrics (EM / Token F1).
Embedding is handled entirely by sentence-transformers — Ollama not required
for retrieval or filtering steps.
"""

from __future__ import annotations

import logging

import httpx

from configs.settings import (
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

_CHAT_URL    = f"{OLLAMA_BASE_URL}/api/chat"
_HTTP_TIMEOUT = 120.0


def build_context(chunks: list[dict]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[{i}] ({chunk['title']})\n{chunk['text'].strip()}")
    return "\n\n".join(lines)


def generate_answer(question: str, chunks: list[dict]) -> str:
    """
    Generate an answer for `question` using `chunks` as context.
    Returns the generated answer string, or "" on failure.
    """
    context  = build_context(chunks)
    user_msg = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    payload = {
        "model":   LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "options": {
            "temperature": LLM_TEMPERATURE,
            "num_predict": LLM_MAX_TOKENS,
        },
        "stream": False,
    }

    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            resp = client.post(_CHAT_URL, json=payload)
            resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as exc:
        logger.warning("Generation failed: %s", exc)
        return ""
