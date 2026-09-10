"""
app.py
------
ScholarMind — Streamlit dashboard.

Layout overview
---------------
Sidebar  : title · How It Works · Parameters · System Status · New Search
Main     : "ScholarMind" title · caption · chat interface

Agent flow
----------
A context-aware routing step runs before the main agentic loop:

  [Follow-up path]  Is the user asking about the current paper?
                    → skip Semantic Scholar search + PDF download, go
                      straight to FAISS RAG.
  [New-topic path]  Standard 2-step loop:
                    Step 1 → refine query → search Semantic Scholar
                    Step 2 → select paper → download PDF → index → answer
"""

import logging
import sys

import streamlit as st

sys.path.insert(0, ".")

from embedder import EMBEDDING_DIM, MODEL_NAME as EMBED_MODEL
from llm_client import (
    LLMClient,
    MODEL_NAME as LLM_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT,
    TEMPERATURE,
)
from rag_pipeline import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CANDIDATE_POOL, RAGPipeline, TOP_K
from semantic_scholar_tool import S2RateLimitError
from embedder import Embedder
from pdf_extractor import clean_llm_output
from llm_client import normalize_bullets

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ScholarMind",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — minimalist, generous whitespace
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    h1 { letter-spacing: -0.5px; }
    [data-testid="stSidebar"] h1 {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    [data-testid="stChatMessage"] {
        border-radius: 10px;
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.5rem;
    }
    hr { opacity: 0.18; }
    [data-testid="stStatusWidget"] { border-radius: 8px; }
    code {
        font-size: 0.82rem;
        padding: 1px 4px;
        border-radius: 3px;
    }
    .citation-card {
        background: rgba(100,100,200,0.07);
        border-left: 3px solid #6c63ff;
        padding: 0.6rem 0.9rem;
        border-radius: 0 6px 6px 0;
        margin-top: 1rem;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached heavy resources (loaded once per server process)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading embedding model...")
def _get_embedder() -> Embedder:
    return Embedder()


@st.cache_resource(show_spinner=False)
def _get_llm() -> LLMClient:
    return LLMClient()


def _get_pipeline() -> RAGPipeline:
    """Return the pipeline stored in session state (created once per session)."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline(
            embedder=_get_embedder(),
            llm=_get_llm(),
        )
    return st.session_state.pipeline


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _source_label(paper: dict) -> str:
    """Return the correct citation label — paper['id'] is a Semantic Scholar
    hash, not an arXiv id, so it must never be shown as 'arXiv:<id>'."""
    if paper.get("arxiv_id"):
        return f"arXiv:{paper['arxiv_id']}"
    if paper.get("doi"):
        return f"DOI:{paper['doi']}"
    return "View on Semantic Scholar"


_init_state()

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("ScholarMind")

    with st.expander("How It Works"):
        st.markdown(
            """
**Autonomous 2-Step Agent + Follow-up Detection**

**Step 0 — Context Check**
If a paper is already indexed and your query is a follow-up
(e.g. "방금 논문 요약해줘", "summarize this"), the agent
**skips the Semantic Scholar search and PDF download** and answers
directly from the existing FAISS index.

**Step 1 — Intent & Search (new topics)**
1. Language auto-detected (EN / KO).
2. LLM decomposes query into concise CS-specific search keywords.
3. Up to 10 paper abstracts are fetched from Semantic Scholar, filtered
   by reliability (DOI/venue/citations) and PDF availability.

**Step 2 — Selection & Deep RAG**
4. LLM autonomously picks the best paper.
5. Full PDF downloaded via a multi-source cascade (arXiv ·
   openAccessPdf · Unpaywall · CORE) and split into 1 000-char chunks.
6. Chunks embedded (multilingual-e5-small) and indexed in FAISS (CPU).
7. Top-K chunks retrieved (hybrid BM25 + dense, compressed) and passed to the LLM.
8. Final cited answer generated in your query language.

*No manual paper selection required.*
            """
        )

    with st.expander("Parameters"):
        st.markdown(
            f"""
| Parameter | Value |
|---|---|
| **Top-K** | {TOP_K} |
| **기본 온도 (utility)** | {TEMPERATURE} |
| **온도 — RAG 답변** | 0.1 |
| **온도 — 논문 선택** | 0.3 |
| **Chunk Size** | {CHUNK_SIZE:,} chars |
| **Chunk Overlap** | {CHUNK_OVERLAP:,} chars |
            """
        )

    st.divider()

    st.markdown("**Ensure Ollama is Running**")

    llm_check = _get_llm()
    ollama_ok = llm_check.is_available()
    available_models = llm_check.list_models() if ollama_ok else []
    model_present = any(LLM_MODEL in m for m in available_models)

    if ollama_ok:
        st.success("Ollama server reachable", icon="✅")
        if model_present:
            st.success(f"`{LLM_MODEL}` is loaded", icon="✅")
        else:
            st.warning(
                f"`{LLM_MODEL}` not found.\n\nRun: `ollama pull {LLM_MODEL}`",
                icon="⚠️",
            )
    else:
        st.error(
            "Cannot reach Ollama.\n\nStart it with: `ollama serve`",
            icon="🔴",
        )

    with st.expander("Advanced config"):
        st.markdown(
            f"""
- **LLM model:** `{LLM_MODEL}`
- **Embedding model:** `{EMBED_MODEL}`
- **Embedding dim:** `{EMBEDDING_DIM}`
- **Ollama base URL:** `{OLLAMA_BASE_URL}`
- **Ollama timeout:** `{OLLAMA_TIMEOUT} s`
- **Vector backend:** FAISS CPU `IndexFlatIP`
- **PDF source:** Semantic Scholar + cascade (arXiv · openAccessPdf · Unpaywall · CORE)
- **PDF cache:** `./pdfs/`
- **Rate limit:** 3 s between Semantic Scholar calls
            """
        )

    st.divider()

    if st.button("🗑️ New Search", use_container_width=True):
        st.session_state.messages = []
        _get_pipeline().reset()
        st.rerun()

# ---------------------------------------------------------------------------
# MAIN AREA
# ---------------------------------------------------------------------------

st.title("ScholarMind")
st.caption("Your Private Research Intelligence on Local Hardware")
st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("질문을 입력하세요  ·  Enter your question")

if user_input:
    if not _get_llm().is_available():
        st.error(
            "Ollama is not running. Please start it with `ollama serve` and refresh.",
            icon="🔴",
        )
        st.stop()

    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant turn
    with st.chat_message("assistant"):
        pipeline = _get_pipeline()
        lang = pipeline.detect_language(user_input)

        # Shared output variables — set in whichever branch runs
        selected_paper = None
        full_answer = ""

        # ----------------------------------------------------------------
        # Context-aware routing
        # Keyword heuristics run first (zero LLM cost).  LLM classification
        # fires only for ambiguous queries.  If a paper is already indexed
        # and the user is continuing that conversation, bypass the entire
        # Semantic Scholar search + PDF download pipeline.
        # ----------------------------------------------------------------
        followup = pipeline.is_followup(user_input)

        if followup:
            # Follow-up path — reuse existing FAISS index
            selected_paper = pipeline.current_paper
            title_preview = (
                selected_paper["title"][:75] + "..."
                if len(selected_paper["title"]) > 75
                else selected_paper["title"]
            )

            with st.status("Follow-up query detected", expanded=True) as status:
                st.write("🔄 **Follow-up** — Reusing the currently indexed paper.")
                st.write(f"→ Paper: **{title_preview}**")
                st.write("🔍 Searching existing FAISS index for relevant passages...")
                status.update(label="Ready to answer...", state="complete")

        else:
            # New-topic path — full 2-step agentic loop
            with st.status("Agent is working...", expanded=True) as status:

                # Step 1a: Query refinement
                st.write("🔍 **Step 1** — Refining query into search keywords...")
                keywords = pipeline.refine_query(user_input)
                st.write(f"→ Keywords: `{keywords}`")

                # Step 1b: Semantic Scholar search
                st.write("📄 Searching Semantic Scholar (rate limit: 3 s/request)...")
                try:
                    papers, effective_kw = pipeline.search_papers(keywords, max_results=10)
                except S2RateLimitError:
                    # 429는 "결과 없음"이 아니라 API 차단 — 질문을 바꾸라고 안내하면 안 된다.
                    status.update(label="API 요청 한도 초과", state="error")
                    st.error(
                        "**Semantic Scholar API 요청 한도(429)에 도달했습니다.**\n\n"
                        "질문의 문제가 아니라 API 호출이 차단된 상태입니다. "
                        "`S2_API_KEY` 환경변수가 설정되지 않으면 전 세계 익명 사용자가 "
                        "공유하는 요청 풀을 사용하게 되어 자주 차단됩니다.\n\n"
                        "**해결 방법:** 무료 API 키를 발급받아 `S2_API_KEY` 환경변수에 "
                        "설정한 뒤 앱을 재시작하세요 — "
                        "https://www.semanticscholar.org/product/api#api-key-form",
                        icon="🚦",
                    )
                    st.stop()

                if not papers:
                    status.update(label="No papers found", state="error")
                    st.error(
                        "Semantic Scholar returned no results even after keyword fallback. "
                        "Please try rephrasing your question.",
                        icon="❌",
                    )
                    st.stop()

                if effective_kw != keywords:
                    st.write(f"→ Fallback keywords used: `{effective_kw}`")
                st.write(f"→ Found **{len(papers)}** candidate papers.")
                if len(papers) < MIN_CANDIDATE_POOL:
                    st.caption(
                        "Candidate pool may be small for broad queries — "
                        "try a more specific term for more options."
                    )

                # Step 2a: Paper selection
                st.write("🧠 **Step 2** — LLM is analysing abstracts...")
                selected_paper, reasoning = pipeline.select_paper(user_input, papers)
                title_short = (
                    selected_paper["title"][:80] + "..."
                    if len(selected_paper["title"]) > 80
                    else selected_paper["title"]
                )
                st.write(f"→ Selected: **{title_short}**")
                st.write(f"→ Reasoning: *{reasoning}*")

                # Step 2b: Download and index PDF (실패 시 다음 후보로 자동 전환)
                st.write("📥 Downloading PDF and building FAISS index...")
                indexed_paper, n_chunks, substituted = pipeline.index_with_fallback(
                    papers, selected_paper
                )

                if n_chunks == 0:
                    status.update(label="PDF indexing failed", state="error")
                    st.error(
                        "Could not download or parse the PDF for any of the top "
                        "candidate papers. Try asking about a different topic.",
                        icon="❌",
                    )
                    st.stop()

                if substituted:
                    # 원래 고른 논문의 PDF를 못 구해서 다른 논문으로 바뀐 경우 —
                    # 인용 카드가 실제 인덱싱된 논문을 가리키도록 교체하고 사용자에게 알림
                    selected_paper = indexed_paper
                    sub_title = (
                        indexed_paper["title"][:80] + "..."
                        if len(indexed_paper["title"]) > 80
                        else indexed_paper["title"]
                    )
                    st.write(
                        f"⚠️ 원래 선택한 논문의 PDF를 확보하지 못해 "
                        f"다음 후보로 전환했습니다: **{sub_title}**"
                    )

                st.write(f"→ Indexed **{n_chunks:,}** chunks into FAISS.")
                status.update(label="Ready to answer...", state="complete")

        # ----------------------------------------------------------------
        # Display — common to both follow-up and new-topic paths
        # ----------------------------------------------------------------
        st.divider()

        # Stream the answer as it's generated; falls back to the non-streaming
        # 3-tier chain (main → simplified → abstract) on an empty/failed stream.
        full_answer = st.write_stream(pipeline.stream_answer(user_input, selected_paper, lang))
        if not full_answer or not full_answer.strip():
            _, full_answer = pipeline.generate_answer(user_input, selected_paper, lang)
            if not full_answer.strip():
                lang_check = pipeline.detect_language(user_input)
                full_answer = (
                    "⚠️ 응답을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요."
                    if lang_check == "ko"
                    else "⚠️ The model returned an empty response. Please try again."
                )
            st.markdown(full_answer)
        full_answer = normalize_bullets(clean_llm_output(full_answer))

        # Citation card
        if selected_paper:
            year = selected_paper["published"][:4]
            authors_str = ", ".join(selected_paper["authors"][:3])
            if len(selected_paper["authors"]) > 3:
                authors_str += " et al."
            source_label = _source_label(selected_paper)
            if selected_paper.get("metadata_suspect"):
                st.caption(
                    "⚠️ Source metadata (author/year) may be inconsistent for this paper — "
                    "please verify against the original publisher or arXiv page before citing."
                )
            st.markdown(
                f"""
<div class="citation-card">
📚 <strong>Source</strong><br>
<em>{selected_paper["title"]}</em><br>
{authors_str} · {year}<br>
<a href="{selected_paper["url"]}" target="_blank">{source_label}</a>
</div>
                """,
                unsafe_allow_html=True,
            )

        # Persist to chat history
        full_content = full_answer
        if selected_paper:
            full_content += (
                f"\n\n---\n📚 **Source:** *{selected_paper['title']}* "
                f"— {authors_str} ({year}) "
                f"[[{source_label}]({selected_paper['url']})]"
            )
        st.session_state.messages.append(
            {"role": "assistant", "content": full_content}
        )
