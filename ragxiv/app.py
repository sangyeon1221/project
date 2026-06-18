"""
app.py
------
Ragxiv — Streamlit dashboard.

Layout overview
---------------
Sidebar  : title · How It Works · Parameters · System Status · New Search
Main     : "Ragxiv" title · caption · chat interface

Agent flow
----------
Fix 4 adds a context-aware routing step before the main agentic loop:

  [Follow-up path]  Is the user asking about the current paper?
                    → skip arXiv + PDF download, go straight to FAISS RAG.
  [New-topic path]  Standard 2-step loop:
                    Step 1 → refine query → search arXiv
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
from rag_pipeline import CHUNK_OVERLAP, CHUNK_SIZE, RAGPipeline, TOP_K
from embedder import Embedder

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ragxiv",
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


_init_state()

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("arXiv Research Assistant")

    with st.expander("How It Works"):
        st.markdown(
            """
**Autonomous 2-Step Agent + Follow-up Detection**

**Step 0 — Context Check (Fix 4)**
If a paper is already indexed and your query is a follow-up
(e.g. "방금 논문 요약해줘", "summarize this"), the agent
**skips arXiv search and PDF download** and answers directly
from the existing FAISS index.

**Step 1 — Intent & Search (new topics)**
1. Language auto-detected (EN / KO).
2. LLM decomposes query into 3 CS-specific arXiv keywords.
3. Up to 10 paper abstracts are fetched from arXiv.

**Step 2 — Selection & Deep RAG**
4. LLM autonomously picks the best paper.
5. Full PDF downloaded and split into 1 000-char chunks.
6. Chunks embedded (BGE-small) and indexed in FAISS (CPU).
7. Top-5 chunks retrieved and passed to the LLM.
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
- **PDF cache:** `./pdfs/`
- **Rate limit:** 3 s between arXiv calls
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

st.title("Ragxiv")
st.caption("Your Private ArXiv Intelligence on Local Hardware")
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
        thinking = ""
        full_answer = ""

        # ----------------------------------------------------------------
        # Fix 4: Context-aware routing
        # Keyword heuristics run first (zero LLM cost).  LLM classification
        # fires only for ambiguous queries.  If a paper is already indexed
        # and the user is continuing that conversation, bypass the entire
        # arXiv search + PDF download pipeline.
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
                st.write("✍️ Generating answer...")
                status.update(label="LLM is reasoning...", state="running")
                thinking, full_answer = pipeline.generate_answer(
                    user_input, selected_paper, lang
                )
                status.update(label="Analysis complete!", state="complete")

        else:
            # New-topic path — full 2-step agentic loop
            with st.status("Agent is working...", expanded=True) as status:

                # Step 1a: Query refinement
                st.write("🔍 **Step 1** — Refining query into search keywords...")
                keywords = pipeline.refine_query(user_input)
                st.write(f"→ Keywords: `{keywords}`")

                # Step 1b: arXiv search
                st.write("📄 Searching arXiv (rate limit: 3 s/request)...")
                papers, effective_kw = pipeline.search_papers(keywords, max_results=10)

                if not papers:
                    status.update(label="No papers found", state="error")
                    st.error(
                        "arXiv returned no results even after keyword fallback. "
                        "Please try rephrasing your question.",
                        icon="❌",
                    )
                    st.stop()

                if effective_kw != keywords:
                    st.write(f"→ Fallback keywords used: `{effective_kw}`")
                st.write(f"→ Found **{len(papers)}** candidate papers.")

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

                # Step 2b: Download and index PDF
                st.write("📥 Downloading PDF and building FAISS index...")
                n_chunks = pipeline.index_paper(selected_paper)

                if n_chunks == 0:
                    status.update(label="PDF indexing failed", state="error")
                    st.error(
                        "Could not download or parse the PDF. "
                        "Try asking about a different topic.",
                        icon="❌",
                    )
                    st.stop()

                st.write(f"→ Indexed **{n_chunks:,}** chunks into FAISS.")

                # Step 2c: Generate answer (Fix 5 Korean anchor applied inside llm_client)
                st.write("✍️ Generating answer from retrieved context...")
                status.update(label="LLM is reasoning...", state="running")
                thinking, full_answer = pipeline.generate_answer(
                    user_input, selected_paper, lang
                )
                status.update(label="Analysis complete!", state="complete")

        # ----------------------------------------------------------------
        # Display — common to both follow-up and new-topic paths
        # ----------------------------------------------------------------
        st.divider()

        # Agent reasoning block (collapsed by default)
        if thinking.strip():
            with st.expander("💭 Agent Reasoning", expanded=False):
                st.markdown(
                    thinking,
                    help="The model's internal chain-of-thought before writing the answer.",
                )
            st.write("")

        # Final answer — guard against blank buffer reaching the UI
        if not full_answer.strip():
            lang = pipeline.detect_language(user_input)
            full_answer = (
                "⚠️ 응답을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요."
                if lang == "ko"
                else "⚠️ The model returned an empty response. Please try again."
            )
        st.markdown(full_answer)

        # Citation card
        if selected_paper:
            year = selected_paper["published"][:4]
            authors_str = ", ".join(selected_paper["authors"][:3])
            if len(selected_paper["authors"]) > 3:
                authors_str += " et al."
            st.markdown(
                f"""
<div class="citation-card">
📚 <strong>Source</strong><br>
<em>{selected_paper["title"]}</em><br>
{authors_str} · {year}<br>
<a href="{selected_paper["url"]}" target="_blank">arXiv:{selected_paper["id"]}</a>
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
                f"[[arXiv]({selected_paper['url']})]"
            )
        st.session_state.messages.append(
            {"role": "assistant", "content": full_content}
        )
