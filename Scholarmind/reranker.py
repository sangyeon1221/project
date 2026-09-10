"""
reranker.py
-----------
Optional cross-encoder reranker for Ragxiv.

Enabled by setting RERANK_ENABLED=1 in the environment before launching.
When disabled (default), this module is imported but the CrossEncoder model
is NOT loaded — no memory or startup cost.

Model selection
---------------
Default: BAAI/bge-reranker-base (~278 MB, English-focused).
Override: set RERANK_MODEL=BAAI/bge-reranker-v2-m3 for a multilingual
          reranker that ranks Korean-heavy queries better, at the cost of
          a larger download and ~2× slower inference on CPU.

Latency
-------
bge-reranker-base on CPU: ~1–3 s for 20 (query, chunk) pairs.
This is the dominant extra cost when RERANK_ENABLED=1.  The precision
gain typically removes 1–2 hallucination-triggering chunks, which more
than repays the cost through shorter, more accurate LLM output.
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

RERANK_MODEL: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")


class Reranker:
    """
    Cross-encoder reranker backed by sentence-transformers CrossEncoder.

    The model is loaded once at construction time (heavy — ~278 MB for the
    default model).  RAGPipeline holds an Optional[Reranker] and only
    constructs it when RERANK_ENABLED=1.
    """

    def __init__(self, model_name: str = RERANK_MODEL) -> None:
        # Import here so the CrossEncoder download only happens when
        # RERANK_ENABLED=1; the module-level import is cost-free.
        from sentence_transformers import CrossEncoder
        logger.info("Loading reranker model: %s", model_name)
        self._model = CrossEncoder(model_name)
        logger.info("Reranker ready: %s", model_name)

    def rerank(self, query: str, chunks: list, top_k: int) -> list:
        """
        Re-score *chunks* against *query* and return the top-k by score.

        Args:
            query:  Raw user query string.
            chunks: List of chunk dicts (keys: chunk, metadata, score).
            top_k:  Number of chunks to keep.

        Returns:
            Up to *top_k* chunks sorted by cross-encoder score (best first).
        """
        if not chunks:
            return chunks
        pairs = [(query, c["chunk"]) for c in chunks]
        scores = self._model.predict(pairs)
        order = np.argsort(scores)[::-1][:top_k]
        return [chunks[i] for i in order]
