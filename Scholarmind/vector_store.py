"""
vector_store.py
---------------
In-memory FAISS vector store (CPU, IndexFlatIP) with BM25 lexical index
fused via Reciprocal Rank Fusion.

Dense retrieval (e5 vectors) and lexical retrieval (BM25) complement each
other: dense captures semantic similarity across languages; BM25 captures
exact matches on acronyms, math symbols, and proper nouns that appear
verbatim in the query.  RRF(k=60) combines both rank lists without requiring
score normalisation.

For Korean queries against English chunks, BM25 contributes little (token
overlap is low) and dense dominates — RRF degrades gracefully in this case.
BM25 gains influence when the query contains English technical terms.
"""

import logging
import re
from typing import Dict, List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from embedder import EMBEDDING_DIM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BM25 tokeniser
# ---------------------------------------------------------------------------
_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


class VectorStore:
    """
    Ephemeral hybrid index (FAISS dense + BM25 lexical) for a single paper.

    Call :meth:`reset` before indexing a new paper so stale chunks from
    the previous session don't pollute retrieval results.
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self._dim = dim
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self._chunks: List[str] = []
        self._metadata: List[Dict] = []
        self._bm25: BM25Okapi | None = None

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
        Index *chunks* into FAISS and build the BM25 lexical index.

        Both indices are rebuilt over the complete chunk list every time so
        BM25 scores are always consistent with the dense index.

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

        # Build BM25 over the complete chunk list (fast — pure Python, non-neural)
        self._bm25 = BM25Okapi([_tokenize(c) for c in self._chunks])

        logger.info(
            "Indexed %d chunks (total %d); BM25 index built", len(chunks), self.size
        )

    def search(
        self,
        query_vector: np.ndarray,
        query_text: str = "",
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> List[Dict]:
        """
        Hybrid retrieval: dense FAISS + BM25, fused with RRF(k=60).

        Args:
            query_vector:  L2-normalised float32 vector of shape (dim,).
            query_text:    Raw query string for BM25 (optional; skipped if empty).
            top_k:         Final number of results to return.
            candidate_k:   Pool size for each ranker before fusion.

        Returns:
            List of dicts (chunk, metadata, score) sorted by fused RRF score.
        """
        if self.size == 0:
            return []

        cand = min(candidate_k, self.size)

        # Dense ranking
        q = query_vector.reshape(1, -1).astype(np.float32)
        _, didx = self._index.search(q, cand)
        dense = [int(i) for i in didx[0] if i >= 0]

        # Lexical ranking (skipped when query has no tokens or BM25 not built)
        sparse: List[int] = []
        if self._bm25 is not None and query_text.strip():
            toks = _tokenize(query_text)
            if toks:
                scores = self._bm25.get_scores(toks)
                sparse = [int(i) for i in np.argsort(scores)[::-1][:cand]]

        # Reciprocal Rank Fusion
        K = 60
        fused: Dict[int, float] = {}
        for rank, idx in enumerate(dense):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (K + rank + 1)
        for rank, idx in enumerate(sparse):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (K + rank + 1)

        top = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            {
                "chunk": self._chunks[i],
                "metadata": self._metadata[i],
                "score": float(s),
            }
            for i, s in top
        ]

    def head_chunks(self, n: int = 3) -> List[Dict]:
        """
        문서 앞부분에서 *n*개 청크를 문서 순서 그대로 반환.

        요약형 질의("요약해줘")처럼 벡터 유사도가 의미 있는 신호를 주지 못하는
        경우에 쓴다.  논문은 앞부분에 제목·초록·서론이 오므로, 유사도 상위 청크
        (부록의 실험 설정 등이 뽑힐 수 있음)보다 요약 근거로 훨씬 적합하다.

        Returns:
            List of dicts with keys: chunk, metadata, score (score는 항상 1.0).
        """
        return [
            {"chunk": self._chunks[i], "metadata": self._metadata[i], "score": 1.0}
            for i in range(min(n, len(self._chunks)))
        ]

    def reset(self) -> None:
        """Wipe both indices so a new paper can be loaded cleanly."""
        self._index = faiss.IndexFlatIP(self._dim)
        self._chunks = []
        self._metadata = []
        self._bm25 = None
        logger.info("VectorStore reset")
