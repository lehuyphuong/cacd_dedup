"""Structured logging setup for rag-bench v3."""

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    for noisy in ("httpx", "httpcore", "datasets", "urllib3",
                  "qdrant_client", "sentence_transformers", "torch"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
