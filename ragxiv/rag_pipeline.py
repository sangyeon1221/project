"""
rag_pipeline.py
---------------
Orchestrator for the Ragxiv autonomous 2-Step Agentic Loop.

Step 1 — Intent & Search
    • Detect the user's language.
    • Refine the query into English arXiv keywords via LLM.
    • Fetch candidate paper metadata from arXiv.

Step 2 — Selection & Deep RAG
    • LLM autonomously selects the best paper with visible reasoning.
    • Download the full PDF.
    • Extract text and split into chunks.
    • Embed chunks → build FAISS index.
    • Retrieve top-K chunks for the query.
    • Generate the final answer with citations.

Integrated Fix — Keyword Memory Purge
    _last_keywords tracks the keyword string produced by the previous
    refine_query call.  On every new call, this string is passed to
    LLMClient.refine_query as avoid_tokens so the 1B model is explicitly
    instructed not to recycle fragments from prior output (e.g. "LLA Agent",
    stale phrase components).  _last_keywords is cleared in reset() and in
    index_paper() so a fully new search always starts fresh.

Each public method is intentionally small so app.py can interleave
Streamlit status updates between the steps.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from arxiv_tool import ArxivTool
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
TOP_K: int = 5  # safe experiment: raise to 6 within 4096 ctx if grounding on vague queries needs improvement

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
        self._arxiv = ArxivTool(save_dir=save_dir)
        self._store = VectorStore()
        self._current_paper: Optional[Dict] = None
        self._last_keywords: Optional[str] = None   # memory purge guard

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
        Fetch paper metadata from arXiv for *keywords*.

        Per-phrase search strategy:
          1. Split the "|"-separated keyword string into individual phrases
             (cap at 3 to bound latency).
          2. For each phrase: try exact-quoted first (good for canonical compound
             terms like "neuro-symbolic"); if zero hits, retry the same phrase
             unquoted (good for loose descriptive phrases like "hallucination
             medical AI"). OR-phrases skip quoting and go straight to unquoted.
             Worst case: 2 arXiv calls per phrase, 6 total; early-exit when
             merged reaches max_results keeps the common case fast.
          3. Merge in phrase order, deduplicate by paper id, cap at max_results.
          4. Progressive word-drop fallback fires only when ALL phrase searches
             yield nothing (same behaviour as before for truly unknown terms).

        Returns:
            (papers, effective_keywords)
        """
        phrases = [p.strip() for p in keywords.split("|") if p.strip()]
        phrases = phrases[:3]  # cap: at most 3 phrases; worst case 2 calls each = 6 arXiv calls

        seen_ids: set = set()
        merged: List[Dict] = []

        for phrase in phrases:
            if len(merged) >= max_results:
                break

            has_or = " OR " in phrase.upper()
            # OR-phrases stay unquoted (boolean); others try exact-quoted first,
            # then fall back to the same phrase unquoted — catches loose descriptive
            # phrases like "hallucination medical AI" that return 0 exact hits.
            attempts = [phrase] if has_or else [f'"{phrase}"', phrase]

            batch: List[Dict] = []
            for q in attempts:
                batch = self._arxiv.search(q, max_results=max_results)
                if batch:
                    break  # stop at the first attempt that yields hits

            for paper in batch:
                if paper["id"] not in seen_ids and len(merged) < max_results:
                    seen_ids.add(paper["id"])
                    merged.append(paper)
            logger.info(
                "Phrase '%s' -> %d hits (%d total so far)", phrase, len(batch), len(merged)
            )

        if merged:
            return merged, keywords

        # Progressive word-drop fallback — unchanged from original
        kw_parts = keywords.split()
        for n in range(len(kw_parts) - 1, 0, -1):
            shorter = " ".join(kw_parts[:n])
            logger.warning("No results for '%s', retrying with '%s'", keywords, shorter)
            results = self._arxiv.search(shorter, max_results=max_results)
            if results:
                return results, shorter

        return [], keywords

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

        pdf_path = self._arxiv.download_pdf(paper)
        if pdf_path is None:
            logger.error("Could not download PDF for %s", paper["id"])
            return 0

        full_text = extract_pdf_text(pdf_path)
        if not full_text:
            logger.error("Empty text from %s", pdf_path)
            return 0

        chunks = split_text(full_text)
        if not chunks:
            return 0

        embeddings = self._embedder.embed_documents(chunks)
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
        context_chunks = self._store.search(query_vec, top_k=TOP_K)

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

    def reset(self) -> None:
        """Wipe the vector store, clear the current paper, and purge keyword memory."""
        self._store.reset()
        self._current_paper = None
        self._last_keywords = None
