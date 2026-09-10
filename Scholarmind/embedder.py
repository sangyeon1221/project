"""
embedder.py
-----------
Thin wrapper around intfloat/multilingual-e5-small (384 dims).

multilingual-e5-small aligns Korean queries and English passages in a shared
embedding space, which is critical for Ragxiv: user queries arrive in Korean
while paper chunks are English text. BGE-small-en was English-only and produced
poor cross-lingual retrieval for Korean queries.

e5 prefix scheme (applied here transparently):
  - "query: "   prepended to every query vector (embed_query)
  - "passage: " prepended to every document chunk (embed_documents)
Omitting either prefix degrades retrieval quality significantly per the
intfloat/multilingual-e5 paper and model card.

First-run note: ~118 MB model download from Hugging Face on first import;
subsequent runs use the local HF cache. The one-time delay is not a bug.

EMBEDDING_DIM is 384 — identical to the previous BGE model — so FAISS,
vector_store.py, and chunk sizes are unchanged.
"""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME: str = "intfloat/multilingual-e5-small"
EMBEDDING_DIM: int = 384  # unchanged — e5-small is also 384-dim, FAISS stays the same

# e5 requires task prefixes: "query: " for queries, "passage: " for documents.
_QUERY_PREFIX: str = "query: "
_PASSAGE_PREFIX: str = "passage: "


class Embedder:
    """
    Singleton-safe text embedder backed by multilingual-e5-small.

    Use :meth:`embed_documents` for indexing chunks and
    :meth:`embed_query` for query vectors — they apply different e5 prefixes
    ("passage: " vs "query: ") that are required for correct retrieval quality.
    """

    def __init__(self) -> None:
        logger.info("Loading embedding model: %s", MODEL_NAME)
        self._model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model ready (dim=%d)", EMBEDDING_DIM)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of document chunks into L2-normalised vectors.

        Prepends the e5 "passage: " prefix to every chunk before encoding.
        This prefix is mandatory for e5 models — it shifts the vector into
        the passage region of the shared multilingual space so query vectors
        (which use "query: ") land nearby.

        Args:
            texts: Raw text chunks to embed.

        Returns:
            Float32 array of shape (len(texts), EMBEDDING_DIM).
        """
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        prefixed = [_PASSAGE_PREFIX + t for t in texts]
        vectors = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string into an L2-normalised vector.

        Prepends the e5 "query: " prefix automatically. Works for both Korean
        and English queries — multilingual-e5-small maps them into the same
        vector space as English passages.

        Args:
            query: Raw query string (Korean or English).

        Returns:
            Float32 array of shape (EMBEDDING_DIM,).
        """
        prefixed = _QUERY_PREFIX + query
        vector = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.astype(np.float32)
