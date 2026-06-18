"""
vector_store.py
---------------
In-memory FAISS vector store (CPU, IndexFlatIP).

Because document embeddings are L2-normalised by the Embedder, inner
product is equivalent to cosine similarity and IndexFlatIP gives exact
nearest-neighbour results without any approximation.
"""

import logging
from typing import Dict, List

import faiss
import numpy as np

from embedder import EMBEDDING_DIM

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Ephemeral FAISS index that holds chunks from a single paper session.

    Call :meth:`reset` before indexing a new paper so stale chunks from
    the previous session don't pollute retrieval results.
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self._dim = dim
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self._chunks: List[str] = []
        self._metadata: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of vectors currently stored."""
        return self._index.ntotal

    def add_chunks(
        self,
        chunks: List[str],
        embeddings: np.ndarray,
        metadata: List[Dict],
    ) -> None:
        """
        Index *chunks* into FAISS together with their *metadata*.

        Args:
            chunks:     Raw text strings — one per vector.
            embeddings: Float32 array of shape (len(chunks), dim).
            metadata:   One metadata dict per chunk (paper_id, title …).
        """
        if not chunks:
            return
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but "
                f"{embeddings.shape[0]} embeddings"
            )

        self._index.add(embeddings)
        self._chunks.extend(chunks)
        self._metadata.extend(metadata)
        logger.info("Indexed %d chunks (total %d)", len(chunks), self.size)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Return the top-K most relevant chunks for *query_vector*.

        Args:
            query_vector: L2-normalised float32 vector of shape (dim,).
            top_k:        Number of results to return.

        Returns:
            List of dicts with keys: chunk, metadata, score.
            Sorted descending by score (highest = most similar).
        """
        if self.size == 0:
            return []

        k = min(top_k, self.size)
        q = query_vector.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append(
                    {
                        "chunk": self._chunks[idx],
                        "metadata": self._metadata[idx],
                        "score": float(score),
                    }
                )
        return results

    def reset(self) -> None:
        """Wipe the index so a new paper can be loaded cleanly."""
        self._index = faiss.IndexFlatIP(self._dim)
        self._chunks = []
        self._metadata = []
        logger.info("VectorStore reset")
