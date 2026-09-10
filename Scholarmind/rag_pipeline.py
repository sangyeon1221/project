"""
rag_pipeline.py
---------------
Orchestrator for the Ragxiv autonomous 2-Step Agentic Loop.

Step 1 — Intent & Search
    • Detect the user's language.
    • Refine the query into Semantic Scholar keywords via LLM.
    • Fetch candidate paper metadata from Semantic Scholar.
    • Apply reliability filter (STRICT / BALANCED / OFF) with automatic
      relaxation: STRICT → BALANCED → OFF until candidates remain.
    • Apply PDF pre-filter: remove papers with no resolvable PDF source
      before the LLM paper-selection step.

Step 2 — Selection & Deep RAG
    • LLM autonomously selects the best paper with visible reasoning.
    • Download the full PDF via the multi-source cascade in
      SemanticScholarTool (arXiv → openAccessPdf → Unpaywall → CORE).
    • Extract text and split into chunks.
    • Embed chunks → build FAISS index.
    • Retrieve top-K chunks for the query.
    • Generate the final answer with citations.

Integrated Fix — Keyword Memory Purge
    _last_keywords tracks the keyword string produced by the previous
    refine_query call.  On every new call, this string is passed to
    LLMClient.refine_query as avoid_tokens so the model is explicitly
    instructed not to recycle fragments from prior output.  _last_keywords
    is cleared in reset() and in index_paper() so a fully new search always
    starts fresh.

Each public method is intentionally small so app.py can interleave
Streamlit status updates between the steps.
"""

import logging
import os
import re
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from semantic_scholar_tool import SemanticScholarTool, S2RateLimitError
from embedder import Embedder
from llm_client import LLMClient
from pdf_extractor import extract_text as extract_pdf_text
from vector_store import VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking parameters
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = 1000       # characters
CHUNK_OVERLAP: int = 150     # characters
TOP_K: int = 3         # final chunks fed to LLM; precision raised by hybrid+compress so 3 > old 5
CANDIDATE_K: int = 20  # pool size for hybrid retrieval before rerank/compress

# Sentence splitter for context compression (Part 3)
_SENT = re.compile(r"(?<=[.!?。])\s+|\n+")

# 요약형 질의 감지 — "요약해줘" 같은 질문은 논문 내용과 의미적으로 무관해서
# 벡터 유사도 검색이 사실상 무작위 청크를 반환한다(관찰된 사례: 부록의 실험
# 하이퍼파라미터가 뽑혀서 그게 논문 주제인 것처럼 요약됨).  이런 질의는 유사도
# 검색 대신 문서 앞부분(제목/초록/서론)을 컨텍스트로 쓰는 편이 정확하다.
_SUMMARY_INTENT: tuple = (
    "요약", "정리해", "정리해줘", "무슨 내용", "어떤 논문", "무슨 논문",
    "summarize", "summarise", "summary", "overview",
    "what is this paper", "what's this paper", "tldr",
)


def _is_summary_intent(query: str) -> bool:
    """*query*가 논문 전체 요약을 요구하는 질의인지 판단."""
    q = query.lower()
    return any(kw in q for kw in _SUMMARY_INTENT)

# Defensive stopword strip for search_papers phrases — catches filler that
# slips past _REFINE_PROMPT (e.g. "study", "and") before querying S2.
_QUERY_STOPWORDS: frozenset = frozenset(
    {"and", "the", "of", "study", "research", "analysis", "about", "tell", "me"}
)


def _strip_stopwords(phrase: str) -> str:
    """Remove filler tokens from *phrase* before sending it to Semantic Scholar."""
    words = [w for w in phrase.split() if w.lower() not in _QUERY_STOPWORDS]
    return " ".join(words)

# ---------------------------------------------------------------------------
# Reliability filter configuration
# ---------------------------------------------------------------------------
# STRICT   — paper must have a DOI OR a named venue (formal publication signal).
# BALANCED — STRICT OR citation_count >= MIN_CITATIONS OR published within
#            RECENT_MONTHS (recency exemption: cutting-edge preprints are
#            allowed even if un-cited, but old un-cited un-published papers
#            are filtered out). Default.
# OFF      — no filtering; all S2 results are forwarded.
#
# When the active mode yields 0 candidates, the ladder relaxes automatically:
#   STRICT → BALANCED → OFF
# OFF never yields 0 on non-empty input, so a dead end only occurs when S2
# itself returned no results.

RELIABILITY_MODE: str = "BALANCED"   # "STRICT" | "BALANCED" | "OFF"
MIN_CITATIONS:    int = 5            # BALANCED: min citations for older papers
RECENT_MONTHS:    int = 6            # BALANCED: recency exemption window
MIN_CANDIDATE_POOL: int = 5          # pad the pool if fewer survive the phrase loop

_RELIABILITY_LADDER: List[str] = ["STRICT", "BALANCED", "OFF"]

# ---------------------------------------------------------------------------
# Follow-up detection keyword sets
# ---------------------------------------------------------------------------
# Checked before calling the LLM so obvious follow-up queries are handled
# instantly without an extra round-trip to Ollama.

_FOLLOWUP_KW_KO: frozenset = frozenset([
    "방금", "이 논문", "이논문", "앞서", "이것", "이 내용", "이내용",
    "요약해", "요약해줘", "요약하", "해석해", "번역해", "더 설명",
    "계속", "이 부분", "무슨 뜻", "다시 설명", "더 알려", "이게 뭔",
])

_FOLLOWUP_KW_EN: frozenset = frozenset([
    "this paper", "the paper", "summarize", "summarise",
    "translate", "explain this", "follow up", "more detail",
    "continue", "this section", "what does", "elaborate",
    "tell me more", "expand on",
])


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split *text* into overlapping character-level chunks.

    Args:
        text:          Full document text.
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of non-empty chunk strings.
    """
    if not text:
        return []
    chunks = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    """
    Stateful orchestrator that holds the embedder, vector store, and
    LLM client as long-lived resources across multiple queries.

    Typical call sequence from app.py
    ----------------------------------
    1. lang            = pipeline.detect_language(user_query)
    2. keywords        = pipeline.refine_query(user_query)
    3. papers, eff_kw  = pipeline.search_papers(keywords)
    4. paper, reason   = pipeline.select_paper(user_query, papers)
    5. n_chunks        = pipeline.index_paper(paper)
    6. thinking, ans   = pipeline.generate_answer(user_query, paper, lang)
    """

    def __init__(
        self,
        embedder: Embedder,
        llm: LLMClient,
        save_dir: str = "./pdfs",
    ) -> None:
        self._embedder = embedder
        self._llm = llm
        self._s2 = SemanticScholarTool(save_dir=save_dir)
        self._store = VectorStore()
        self._current_paper: Optional[Dict] = None
        self._last_keywords: Optional[str] = None   # memory purge guard

        # Optional cross-encoder reranker — loaded only when RERANK_ENABLED=1
        # to avoid the ~278 MB model download on default runs.
        if os.getenv("RERANK_ENABLED") == "1":
            from reranker import Reranker
            self._reranker: Optional[object] = Reranker()
        else:
            self._reranker = None

    # ------------------------------------------------------------------
    # Contextual follow-up detection
    # ------------------------------------------------------------------

    @property
    def current_paper(self) -> Optional[Dict]:
        """Currently indexed paper, or None if no paper has been loaded yet."""
        return self._current_paper

    def is_followup(self, user_query: str) -> bool:
        """
        Return True if *user_query* is a follow-up about the currently
        indexed paper, False if it introduces a new topic.

        Detection strategy (fastest-first):
          1. Guard — no paper indexed yet → always False.
          2. Keyword heuristic — deterministic, zero LLM cost.
          3. LLM classification — fallback for ambiguous queries.
        """
        if self._current_paper is None or self._store.size == 0:
            return False

        q_lower = user_query.lower()

        if any(kw in q_lower for kw in _FOLLOWUP_KW_KO):
            logger.info("Follow-up detected via KO keyword: '%s'", user_query[:50])
            return True
        if any(kw in q_lower for kw in _FOLLOWUP_KW_EN):
            logger.info("Follow-up detected via EN keyword: '%s'", user_query[:50])
            return True

        result = self._llm.classify_query(user_query, self._current_paper["title"])
        return result == "followup"

    # ------------------------------------------------------------------
    # Step 1 helpers
    # ------------------------------------------------------------------

    def detect_language(self, text: str) -> str:
        """Return 'ko' for Korean input, 'en' otherwise."""
        return LLMClient.detect_language(text)

    def refine_query(self, user_query: str) -> str:
        """
        Convert *user_query* into English arXiv search keywords.

        Passes _last_keywords as avoid_tokens to the LLM so previously
        generated fragments (e.g. "LLA", stale phrase components) are
        explicitly excluded from the new output.  Updates _last_keywords
        after each successful generation.
        """
        keywords = self._llm.refine_query(
            user_query, avoid_tokens=self._last_keywords
        )
        self._last_keywords = keywords
        return keywords

    def search_papers(
        self, keywords: str, max_results: int = 10
    ) -> Tuple[List[Dict], str]:
        """
        Fetch, filter, and return paper candidates from Semantic Scholar.

        Per-phrase search strategy (A):
          Each "|"-separated phrase is sent as a PLAIN-TEXT relevance query
          to S2 — no quoting, no arXiv-style OR booleans.  Phrases are tried
          in order (most-specific first), results are merged and deduped by
          paper id, capped at max_results.  Early-exit when the cap is reached.

        Thin-pool padding:
          If the merged pool is non-empty but smaller than MIN_CANDIDATE_POOL,
          a broader query (first 1-2 words of the most-specific phrase) is run
          once to pad the pool — prevents the LLM from "choosing" between only
          1-2 candidates on vague queries whose refined phrases are too narrow.

        Reliability filter (F/J):
          Starts at RELIABILITY_MODE and steps down STRICT → BALANCED → OFF
          until at least one candidate remains.  OFF always passes non-empty
          input, so a dead end only occurs when S2 itself returned nothing.

        PDF pre-filter (§4-1):
          Papers with no resolvable PDF source are removed before the LLM
          selection step so index_paper never hits a dead end.  If all papers
          are removed, the pre-filter is skipped (better to try than to always
          fail — _abstract_fallback guards answer quality).

        Abstract backfill (Part 2a):
          All intermediate self._s2.search(...) calls above pass backfill=False
          so the arXiv Atom batch call does not fire on every phrase/padding/
          fallback attempt.  Backfill runs exactly once, on the final filtered
          candidate list, right before returning.

        Returns:
            (papers, effective_keywords)
        """
        phrases = [p.strip() for p in keywords.split("|") if p.strip()]
        phrases = phrases[:3]   # cap: at most 3 S2 calls for the main loop

        seen_ids: set = set()
        merged: List[Dict] = []

        for phrase in phrases:
            if len(merged) >= max_results:
                break

            # Defensive: S2 is not boolean — never send a literal "OR". If the
            # model slipped one in despite the prompt rule, split into plain
            # sub-queries and try each in order (first non-empty wins).
            sub_queries = (
                [s.strip() for s in re.split(r"\s+OR\s+", phrase, flags=re.IGNORECASE) if s.strip()]
                if re.search(r"\bOR\b", phrase, re.IGNORECASE)
                else [phrase]
            )

            batch: List[Dict] = []
            for sub in sub_queries:
                clean_sub = _strip_stopwords(sub)
                batch = self._s2.search(clean_sub or sub, max_results=max_results, backfill=False)
                if batch:
                    break

            for paper in batch:
                if paper["id"] not in seen_ids and len(merged) < max_results:
                    seen_ids.add(paper["id"])
                    merged.append(paper)
            logger.info(
                "Phrase '%s' -> %d hits (%d total so far)", phrase, len(batch), len(merged)
            )

        if 0 < len(merged) < MIN_CANDIDATE_POOL:
            # Thin-but-nonzero pool: pad with a broader query on the single most
            # generic phrase (first 1-2 words of phrase[0]) rather than the full
            # compound phrase.  This targets invented enterprise-sounding compounds
            # (e.g. "AI Workflow Optimization") that individually return very few
            # hits — a broader query surfaces more real candidates for the LLM to
            # actually choose between.
            broad_terms = phrases[0].split()[:2] if phrases else keywords.split()[:2]
            broad_query = " ".join(broad_terms)
            logger.info(
                "Candidate pool thin (%d < %d) — padding with broader query '%s'",
                len(merged), MIN_CANDIDATE_POOL, broad_query,
            )
            extra = self._s2.search(broad_query, max_results=max_results, backfill=False)
            for paper in extra:
                if paper["id"] not in seen_ids and len(merged) < max_results:
                    seen_ids.add(paper["id"])
                    merged.append(paper)

        if not merged:
            # Progressive word-drop fallback on the first (most-specific) phrase
            first_phrase = phrases[0] if phrases else keywords
            kw_parts = first_phrase.split()
            for n in range(len(kw_parts) - 1, 0, -1):
                shorter = " ".join(kw_parts[:n])
                logger.warning(
                    "No results for '%s', retrying with '%s'", first_phrase, shorter
                )
                results = self._s2.search(shorter, max_results=max_results, backfill=False)
                if results:
                    merged = results
                    keywords = shorter
                    break

        if not merged:
            return [], keywords

        # Reliability filter with automatic relaxation ladder (F, J)
        filtered = self._reliability_filter(merged)

        # PDF pre-filter: only pass papers with a resolvable PDF source (§4-1)
        pdf_ready = self._pdf_prefilter(filtered)

        # Backfill abstracts exactly once, on the final candidate list — all
        # intermediate self._s2.search(...) calls above pass backfill=False
        # to avoid redundant arXiv batch calls on candidates that never
        # reach the LLM (Part 2a).
        self._s2.backfill_abstracts(pdf_ready)
        pdf_ready.sort(key=lambda p: 0 if p.get("abstract") else 1)

        return pdf_ready, keywords

    # ------------------------------------------------------------------
    # Step 2 helpers
    # ------------------------------------------------------------------

    def select_paper(
        self, query: str, papers: List[Dict]
    ) -> Tuple[Dict, str]:
        """
        LLM autonomously selects the best paper from *papers*.

        Returns:
            (paper_dict, reasoning_string)
        """
        return self._llm.select_best_paper(query, papers)

    def index_paper(self, paper: Dict) -> int:
        """
        Download the PDF, extract text, chunk it, embed, and build FAISS.

        Clears _last_keywords on entry: a new paper means a new topic, so
        the previous keyword string must not contaminate the next refine call.

        Args:
            paper: Metadata dict from search_papers().

        Returns:
            Number of chunks indexed (0 indicates a failure).
        """
        self._store.reset()
        self._current_paper = paper
        self._last_keywords = None      # purge keyword memory on new paper

        pdf_path = self._s2.download_pdf(paper)
        if pdf_path is None:
            logger.error("Could not download PDF for %s", paper["id"])
            return 0

        full_text = extract_pdf_text(pdf_path)

        # Cheap non-LLM sanity check: does the paper's own title actually appear
        # (loosely) in the extracted PDF text? Catches metadata/content mismatches
        # beyond the arXiv-ID/year heuristic in semantic_scholar_tool.py. This is a
        # WARNING signal only (PDF layout/OCR can cause false negatives) — it does
        # not block indexing.
        if full_text:
            title_words = [w.lower() for w in re.findall(r"\w+", paper["title"]) if len(w) > 3][:6]
            if title_words:
                head = full_text[:3000].lower()
                hits = sum(1 for w in title_words if w in head)
                if hits < max(1, len(title_words) // 3):
                    logger.warning(
                        "Title/content mismatch suspected for paper_id=%s title=%r "
                        "(%d/%d title words found in extracted text head)",
                        paper["id"], paper["title"], hits, len(title_words),
                    )
                    paper["metadata_suspect"] = True

        if not full_text:
            logger.error("Empty text from %s", pdf_path)
            return 0

        chunks = split_text(full_text)
        if not chunks:
            return 0

        # Embedding cache (Part 6) — skip re-embedding on repeat queries
        safe = paper["id"].replace("/", "_").replace(".", "_")
        emb_path = os.path.join(".", "cache", "emb", f"{safe}.npy")
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)

        if os.path.exists(emb_path):
            embeddings = np.load(emb_path)
            if embeddings.shape[0] == len(chunks):
                logger.info("Embedding cache hit: %s (%d chunks)", emb_path, len(chunks))
            else:
                logger.warning(
                    "Embedding cache shape mismatch (%d cached vs %d chunks) — recomputing",
                    embeddings.shape[0], len(chunks),
                )
                embeddings = self._embedder.embed_documents(chunks)
                np.save(emb_path, embeddings)
        else:
            embeddings = self._embedder.embed_documents(chunks)
            np.save(emb_path, embeddings)
            logger.info("Embedding cache saved: %s", emb_path)

        metadata = [
            {
                "paper_id": paper["id"],
                "title": paper["title"],
                "chunk_idx": i,
            }
            for i in range(len(chunks))
        ]
        self._store.add_chunks(chunks, embeddings, metadata)
        logger.info("Indexed %d chunks for '%s'", len(chunks), paper["title"][:50])
        return len(chunks)

    def index_with_fallback(
        self,
        papers: List[Dict],
        selected: Dict,
        max_attempts: int = 3,
    ) -> Tuple[Optional[Dict], int, bool]:
        """
        선택된 논문을 인덱싱하되, PDF 확보 실패 시 다음 후보로 자동 전환.

        기존 동작은 선택된 논문의 PDF가 실패하면 그대로 막다른 길이었다
        (관찰: arXiv ID 없는 저널 논문이 openAccessPdf 링크 하나에만 의존하다
        실패 → "Could not download or parse the PDF").  후보 목록이 이미 있으므로
        다음 후보로 넘어가는 편이 사용자 입장에서 훨씬 낫다.

        Args:
            papers:       search_papers가 반환한 전체 후보 목록.
            selected:     LLM이 고른 논문 (가장 먼저 시도).
            max_attempts: 최대 시도할 논문 수.

        Returns:
            (실제로 인덱싱된 논문, 청크 수, 대체 여부)
            모두 실패하면 (None, 0, False).
        """
        ordered = [selected] + [
            p for p in papers if p.get("id") != selected.get("id")
        ]
        for i, paper in enumerate(ordered[:max_attempts]):
            n_chunks = self.index_paper(paper)
            if n_chunks > 0:
                if i > 0:
                    logger.warning(
                        "선택 논문 PDF 실패 — 대체 논문으로 전환: '%s'",
                        paper["title"][:60],
                    )
                return paper, n_chunks, i > 0
            logger.warning(
                "PDF 확보 실패 (%d/%d): '%s'",
                i + 1, min(max_attempts, len(ordered)), paper["title"][:60],
            )
        return None, 0, False

    def generate_answer(
        self, query: str, paper: Dict, lang: str
    ) -> Tuple[str, str]:
        """
        Retrieve relevant chunks and generate the final answer.

        Three-tier response guarantee:
          1. Main call  → generate_rag_response
          2. If empty   → generate_fallback_summary (shorter prompt, 3 chunks)
          3. If still empty → _abstract_fallback (no LLM — raw abstract)

        Args:
            query: Original user question.
            paper: Selected paper (same dict used in index_paper).
            lang:  'en' or 'ko'.

        Returns:
            (thinking, answer) — thinking is "" with the XML-prompt format.
        """
        query_vec = self._embedder.embed_query(query)
        context_chunks = self._retrieve(query, query_vec)

        if not context_chunks:
            err = (
                "검색된 컨텍스트가 없습니다. 다른 질문을 시도해 주세요."
                if lang == "ko"
                else "No relevant context found. Please try a different question."
            )
            return "", err

        thinking, answer = self._llm.generate_rag_response(
            query, context_chunks, paper, lang
        )

        if not answer.strip():
            logger.warning(
                "Empty RAG response for '%s' — retrying with fallback prompt",
                query[:50],
            )
            thinking, answer = self._llm.generate_fallback_summary(
                query, context_chunks, paper, lang
            )

        if not answer.strip():
            logger.warning("Fallback prompt also empty — serving abstract summary")
            answer = self._abstract_fallback(paper, lang)

        return thinking, answer

    def stream_answer(self, query: str, paper: Dict, lang: str):
        """
        Streaming counterpart to generate_answer. Uses the identical retrieval
        path (_retrieve: hybrid search → optional rerank → compression) so the
        context is identical to the non-streaming call. If retrieval finds
        nothing, yields the localized error string once. Does NOT implement the
        3-tier fallback chain itself — app.py calls generate_answer as the
        fallback if the stream produces no usable text.
        """
        query_vec = self._embedder.embed_query(query)
        context_chunks = self._retrieve(query, query_vec)
        if not context_chunks:
            yield (
                "검색된 컨텍스트가 없습니다. 다른 질문을 시도해 주세요."
                if lang == "ko"
                else "No relevant context found. Please try a different question."
            )
            return
        yield from self._llm.stream_rag_response(query, context_chunks, paper, lang)

    # ------------------------------------------------------------------
    # Unified retrieval path (Parts 3 + 4)
    # ------------------------------------------------------------------

    def _retrieve(self, query: str, query_vec: np.ndarray) -> List[Dict]:
        """
        Hybrid retrieve → optional rerank → sentence-level compress.

        1. BM25 + dense fusion via RRF (candidate_k=CANDIDATE_K).
        2. Cross-encoder rerank if RERANK_ENABLED=1 (optional).
        3. Sentence-level compression using the e5 embedder (no LLM call).

        예외 — 요약형 질의:
          "요약해줘"처럼 논문 내용과 의미적으로 무관한 질의는 유사도 검색이
          무작위에 가까운 청크를 반환한다(관찰: 부록의 실험 하이퍼파라미터가
          뽑혀 그것이 논문 주제인 양 요약됨).  이 경우 유사도 검색을 건너뛰고
          문서 앞부분(제목/초록/서론)을 쓴다.  압축도 생략한다 — 압축은 질의
          유사도로 문장을 고르는데, 요약 질의엔 그 신호가 없기 때문이다.
        """
        if _is_summary_intent(query):
            head = self._store.head_chunks(TOP_K)
            logger.info(
                "요약형 질의 감지 — 유사도 검색 대신 문서 앞부분 %d청크 사용", len(head)
            )
            return head

        chunks = self._store.search(
            query_vec,
            query_text=query,
            top_k=CANDIDATE_K,
            candidate_k=CANDIDATE_K,
        )
        if self._reranker is not None:
            chunks = self._reranker.rerank(query, chunks, top_k=TOP_K)
        else:
            chunks = chunks[:TOP_K]
        return self._compress_context(query_vec, chunks)

    def _compress_context(
        self,
        query_vec: np.ndarray,
        chunks: List[Dict],
        max_sents: int = 3,
        threshold: float = 0.30,
    ) -> List[Dict]:
        """
        Keep only the most query-relevant sentences in each chunk.

        Uses the already-loaded e5 embedder — no extra model or LLM call.
        Cuts prompt length 30–50%, reducing CPU prefill time.
        Logs total char count before and after for visibility.
        """
        before = sum(len(c["chunk"]) for c in chunks)
        out = []
        for c in chunks:
            sents = [
                s.strip()
                for s in _SENT.split(c["chunk"])
                if len(s.strip()) > 15
            ]
            if not sents:
                out.append(c)
                continue
            vecs = self._embedder.embed_documents(sents)   # (n, 384), normalised
            sims = vecs @ query_vec                         # query_vec is normalised
            order = np.argsort(sims)[::-1]
            keep = [
                int(i) for i in order if sims[i] >= threshold
            ][:max_sents] or [int(order[0])]
            keep.sort()   # restore reading order
            out.append({**c, "chunk": " ".join(sents[i] for i in keep)})
        after = sum(len(c["chunk"]) for c in out)
        logger.info(
            "Context compression: %d → %d chars (%.0f%% reduction)",
            before, after, 100 * (1 - after / before) if before else 0,
        )
        return out

    @staticmethod
    def _abstract_fallback(paper: Dict, lang: str) -> str:
        """
        Construct a minimal answer from the paper abstract when both the
        main RAG call and the simplified retry return empty strings.
        """
        title = paper["title"]
        abstract = paper.get("abstract", "")[:600].replace("\n", " ")
        year = paper["published"][:4]
        authors = ", ".join(paper["authors"][:3])
        if len(paper["authors"]) > 3:
            authors += " et al."
        if lang == "ko":
            return (
                f"LLM 응답을 생성하는 데 문제가 발생했습니다. "
                f"대신 논문 초록을 제공합니다.\n\n"
                f"**{title}** — {authors} ({year})\n\n{abstract}"
            )
        return (
            f"The model response could not be generated. "
            f"Here is the paper abstract instead.\n\n"
            f"**{title}** — {authors} ({year})\n\n{abstract}"
        )

    # ------------------------------------------------------------------
    # Reliability filter (F, J)
    # ------------------------------------------------------------------

    def _reliability_filter(self, papers: List[Dict]) -> List[Dict]:
        """
        Filter *papers* by reliability, stepping down the ladder
        STRICT → BALANCED → OFF until at least one paper passes.

        OFF always passes non-empty input, so this only returns [] when
        *papers* itself is empty (which search_papers guards against).
        """
        if RELIABILITY_MODE in _RELIABILITY_LADDER:
            start = _RELIABILITY_LADDER.index(RELIABILITY_MODE)
        else:
            logger.warning(
                "Unknown RELIABILITY_MODE '%s', defaulting to BALANCED", RELIABILITY_MODE
            )
            start = _RELIABILITY_LADDER.index("BALANCED")

        for level_idx in range(start, len(_RELIABILITY_LADDER)):
            mode = _RELIABILITY_LADDER[level_idx]
            filtered = [p for p in papers if self._passes_reliability(p, mode)]
            if filtered:
                if level_idx > start:
                    logger.info(
                        "Reliability filter relaxed from %s to %s: %d/%d papers passed",
                        RELIABILITY_MODE, mode, len(filtered), len(papers),
                    )
                else:
                    logger.info(
                        "Reliability filter (%s): %d/%d papers passed",
                        mode, len(filtered), len(papers),
                    )
                return filtered

        logger.warning("Reliability filter: no papers passed (input list was empty)")
        return []

    def _passes_reliability(self, paper: Dict, mode: str) -> bool:
        """Return True if *paper* meets the reliability threshold for *mode*."""
        if mode == "OFF":
            return True

        has_doi   = bool(paper.get("doi"))
        has_venue = bool(paper.get("venue"))
        strict_ok = has_doi or has_venue

        if mode == "STRICT":
            return strict_ok

        # BALANCED: STRICT OR enough citations OR recently published
        if strict_ok:
            return True
        if (paper.get("citation_count") or 0) >= MIN_CITATIONS:
            return True

        # Recency exemption (D) — guard unparseable dates with try/except (J)
        try:
            pub_date = datetime.strptime(paper.get("published", ""), "%Y-%m-%d").date()
            age_months = (date.today() - pub_date).days / 30.5
            if age_months <= RECENT_MONTHS:
                return True
        except (ValueError, TypeError):
            pass  # unparseable date (e.g. "n.d.-01-01") → treated as old, no exemption

        return False

    # ------------------------------------------------------------------
    # PDF pre-filter (§4-1)
    # ------------------------------------------------------------------

    def _pdf_prefilter(self, papers: List[Dict]) -> List[Dict]:
        """
        Remove papers with no resolvable PDF source before LLM selection.

        Uses is_pdf_likely_available() — a metadata-only check, no download.
        If all papers are removed, the original unfiltered list is returned
        as a fallback so the caller always gets something to work with.
        """
        available = [p for p in papers if self._s2.is_pdf_likely_available(p)]
        if not available:
            logger.warning(
                "PDF pre-filter: all %d papers removed — falling back to unfiltered",
                len(papers),
            )
            return papers
        logger.info(
            "PDF pre-filter: %d/%d papers have resolvable PDFs",
            len(available), len(papers),
        )
        return available

    def reset(self) -> None:
        """Wipe the vector store, clear the current paper, and purge keyword memory."""
        self._store.reset()
        self._current_paper = None
        self._last_keywords = None
