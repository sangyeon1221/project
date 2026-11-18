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

Korean queries share no tokens with English chunks, so rag_pipeline passes
the paper's English search keywords along with the query (_bm25_text) —
otherwise BM25 would contribute nothing for Korean questions.
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


_TOC_LEADER = re.compile(r"\.{5,}\s*\d")

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

    @staticmethod
    def _is_toc(text: str) -> bool:
        """목차 청크인가 — '1.2. Introduction ........ 3'처럼 점선 줄이 3개 이상.
        (관찰: 요약 근거 5개 중 2개가 목차였고, 한계 질문에도 목차 속 'Future Research' 제목이 뽑힘)"""
        return len(_TOC_LEADER.findall(text)) >= 3

    def keyword_chunks(self, pattern, n: int = 3, bonus=None) -> List[Dict]:
        """
        정규식 *pattern*이 많이 나오는 청크 *n*개를 문서 순서대로 반환 (없으면 빈 리스트).
        *bonus*에 맞는 청크(예: 'Limitations' 절 제목)는 점수를 더 준다.
        """
        scored = []
        for i, text in enumerate(self._chunks):
            if self._is_toc(text):
                continue
            hits = len(pattern.findall(text))
            if not hits:
                continue
            if bonus is not None and bonus.search(text):
                hits += 3
            scored.append((hits, i))
        top = [i for _, i in sorted(scored, reverse=True)[:n]]
        # 절 제목이 청크 끝에 걸리면 본문은 다음 청크에 있다(관찰: 7장 제목만 근거로 들어가
        # 모델이 사례 설명을 한계로 바꿔 씀). 제목 청크는 다음 청크도 함께 넣는다(최대 n+1개).
        nxt = [i + 1 for i in top if bonus is not None and bonus.search(self._chunks[i])
               and i + 1 < len(self._chunks) and i + 1 not in top]
        top = sorted(top + nxt[:1])
        return [{"chunk": self._chunks[i], "metadata": self._metadata[i], "score": 1.0} for i in top]

    def head_chunks(self, n: int = 3) -> List[Dict]:
        """
        문서 앞부분에서 *n*개 청크를 문서 순서 그대로 반환.

        요약형 질의("요약해줘")처럼 벡터 유사도가 의미 있는 신호를 주지 못하는
        경우에 쓴다.  논문은 앞부분에 제목·초록·서론이 오므로, 유사도 상위 청크
        (부록의 실험 설정 등이 뽑힐 수 있음)보다 요약 근거로 훨씬 적합하다.

        Returns:
            List of dicts with keys: chunk, metadata, score (score는 항상 1.0).
        """
        idx = [i for i, t in enumerate(self._chunks) if not self._is_toc(t)][:n]
        if not idx:   # 전부 목차로 보이는 비정상 상황이면 앞부분 그대로
            idx = list(range(min(n, len(self._chunks))))
        return [
            {"chunk": self._chunks[i], "metadata": self._metadata[i], "score": 1.0}
            for i in idx
        ]

    def reset(self) -> None:
        """Wipe both indices so a new paper can be loaded cleanly."""
        self._index = faiss.IndexFlatIP(self._dim)
        self._chunks = []
        self._metadata = []
        self._bm25 = None
        logger.info("VectorStore reset")
