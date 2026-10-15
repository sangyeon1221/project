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
import os
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
from semantic_scholar_tool import RATE_LIMIT_DELAY, RATE_LIMIT_DELAY_KEYED, S2RateLimitError
from embedder import Embedder
from pdf_extractor import clean_llm_output
from llm_client import normalize_bullets, strip_answer_echo

logging.basicConfig(level=logging.INFO)

# 화면 안내용 — S2 키가 있으면 1.1초, 없으면 3초 간격 (semantic_scholar_tool 참고)
_S2_DELAY = RATE_LIMIT_DELAY_KEYED if os.environ.get("S2_API_KEY") else RATE_LIMIT_DELAY

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


# 앱을 켜둔 채 소스를 고치면 Streamlit은 바뀐 모듈을 다시 불러오지만, 캐시해 둔
# 객체는 옛 코드로 만들어진 채 남는다. 새 rag_pipeline이 옛 LLMClient에 새 인자를
# 넘겨 TypeError가 난 적이 있다(서버를 다시 켜야 풀렸음). 소스 파일 수정 시각을
# 캐시 키에 넣어, 코드가 바뀌면 객체도 새로 만든다. 실사용(코드 변경 없음)에는 영향 없다.
_HERE = os.path.dirname(os.path.abspath(__file__))


def _mtime(*names: str) -> float:
    return max(os.path.getmtime(os.path.join(_HERE, n)) for n in names)


_PIPELINE_SOURCES = ("rag_pipeline.py", "llm_client.py", "semantic_scholar_tool.py",
                     "vector_store.py", "pdf_extractor.py", "embedder.py")


@st.cache_resource(show_spinner="Loading embedding model...")
def _get_embedder(version: float) -> Embedder:
    # version은 캐시 키로만 쓴다(밑줄로 시작하면 Streamlit이 키에서 빼므로 붙이지 않음)
    return Embedder()


@st.cache_resource(show_spinner=False)
def _get_llm(version: float) -> LLMClient:
    return LLMClient()


def _get_pipeline() -> RAGPipeline:
    """Return the pipeline stored in session state (created once per session)."""
    version = _mtime(*_PIPELINE_SOURCES)
    if st.session_state.get("pipeline_version") != version:
        st.session_state.pipeline = RAGPipeline(
            embedder=_get_embedder(_mtime("embedder.py")),
            llm=_get_llm(_mtime("llm_client.py")),
        )
        st.session_state.pipeline_version = version
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
**Step 0 — 질문 확인**
- 이미 읽은 논문에 대한 후속 질문("방금 논문 요약해줘")이면 검색 없이 바로 답합니다.
- "다른 논문 찾아줘"처럼 새 논문을 원하면 새로 검색합니다.
- "컴퓨터"처럼 한 단어뿐이면 무엇이 궁금한지 먼저 묻습니다.

**Step 1 — 검색**
1. LLM이 질문을 영어 검색어 3개로 바꿉니다 (약어 보존, 정의 질문이면 survey 검색어 추가).
2. Semantic Scholar에서 최대 10편을 찾습니다. "최신"이면 연도로 거릅니다.
3. 신뢰도(DOI·학회·인용)와 PDF 확보 가능성으로 후보를 거릅니다.

**Step 2 — 선택과 읽기**
4. LLM이 초록을 읽고 한 편을 고릅니다. PDF를 못 받으면 질문과 가장 가까운 다음 후보로 넘어갑니다.
5. PDF를 1,000자 단위로 나누고 참고문헌을 빼서 FAISS + BM25로 인덱싱합니다.
6. 관련 문단 5개를 골라 질문과 관련된 문장만 남기고, 초록과 함께 LLM에 넘깁니다.
7. 답변 후 중복 줄을 지우고, 수치를 원문과 다르게 옮긴 곳이 있으면 경고합니다.
8. 답변 아래 후보 목록에서 다른 논문을 골라 같은 질문에 다시 답하게 할 수 있습니다.
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

    llm_check = _get_llm(_mtime("llm_client.py"))
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
- **Rate limit:** {_S2_DELAY:g} s between Semantic Scholar calls
            """
        )

    st.divider()

    if st.button("🗑️ New Search", use_container_width=True):
        st.session_state.messages = []
        st.session_state.candidates = None
        _get_pipeline().reset()
        st.rerun()

# ---------------------------------------------------------------------------
# MAIN AREA
# ---------------------------------------------------------------------------

st.title("ScholarMind")
st.caption("Your Private Research Intelligence on Local Hardware")
st.divider()


# ---------------------------------------------------------------------------
# 후보 논문 목록 — 자동으로 고른 한 편에 답한 뒤, 검색된 후보를 함께 보여주고
# "이 논문으로 다시 답하기"로 바꿀 수 있게 한다. 자동 선택이 빗나갔을 때(관찰:
# Nature 논문이 후보 2번째였는데 표현이 겹친 MLZero가 뽑힘) 클릭 한 번으로 바로잡는다.
# 여러 편을 각각 요약하는 방식은 논문마다 다운로드·인덱싱·답변이 반복되어
# 느린 와이파이에서 1분을 넘기기 쉬워 택하지 않았다. 고르는 데는 TLDR로 충분하다.
# ---------------------------------------------------------------------------


def _candidates_md(papers: list, current_id: str) -> str:
    """대화 기록에 남길 후보 목록(버튼 없는 텍스트판)."""
    rows = []
    for p in papers:
        title = p["title"].replace("[", "(").replace("]", ")")
        line = f"- {p.get('published', '')[:10]} · [{title}]({p.get('url', '')})"
        if p.get("venue"):
            line += f" · {p['venue']}"
        line += f" · 인용 {p.get('citation_count') or 0}"
        if p.get("id") == current_id:
            line += " ← 답변에 사용"
        rows.append(line)
    return "\n".join(rows)


def _choose_switch(paper_id: str) -> None:
    st.session_state.switch_to = paper_id


def _render_candidate_panel() -> None:
    """가장 최근 검색의 후보 목록 — 각 논문에 '이 논문으로 다시 답하기' 버튼."""
    cands = st.session_state.get("candidates")
    if not cands:
        return
    papers, current = cands["papers"], cands["current_id"]
    order_note = "최신순" if cands.get("by_date") else "검색 관련도순"
    with st.expander(
        f"🔎 검색된 후보 논문 {len(papers)}편 ({order_note}) — 질문: \"{cands['question']}\"",
        expanded=cands.get("expanded", False),
    ):
        st.caption("PDF를 구할 수 있는 논문만 보여줍니다. 다른 논문을 고르면 같은 질문에 다시 답합니다.")
        for i, p in enumerate(papers):
            left, right = st.columns([5, 1.2], vertical_alignment="center")
            title = p["title"].replace("[", "(").replace("]", ")")
            meta = f"{p.get('published', '')[:10]}"
            if p.get("venue"):
                meta += f" · {p['venue']}"
            meta += f" · 인용 {p.get('citation_count') or 0}"
            mark = " · ✅ **현재 답변**" if p.get("id") == current else ""
            left.markdown(f"**[{title}]({p.get('url', '')})**  \n{meta}{mark}")
            if p.get("tldr"):
                left.caption(f"TLDR: {p['tldr']}")
            right.button(
                "이 논문으로 답하기", key=f"switch_{cands['id']}_{i}",
                disabled=p.get("id") == current, on_click=_choose_switch, args=(p.get("id"),),
                use_container_width=True,
            )


def _render_answer(pipeline: RAGPipeline, question: str, paper: dict, lang: str,
                   weak_match: bool = False) -> str:
    """
    답변을 스트리밍으로 보여주고 후처리·경고·출처 카드를 붙인다.
    새 검색, 후속 질문, '다시 답하기'가 모두 이 함수를 쓴다.
    대화 기록에 남길 텍스트를 돌려준다.
    """
    st.divider()

    # 논문 선택 단계에서 LLM이 "질문을 직접 다루는 논문이 없다"고 밝힌 경우,
    # 그 신호를 답변 앞에 경고로 보여준다. 신조어 약어(예: MCP)처럼 모델이
    # 모르는 용어에서 자주 생기므로 정식 명칭으로 다시 묻도록 안내한다.
    if weak_match:
        st.warning(
            "질문을 직접 다루는 논문을 찾지 못했습니다. 아래 답변은 가장 가까운 "
            "논문을 바탕으로 합니다. 약어 대신 정식 명칭으로 다시 질문하거나, 아래 "
            "후보 목록에서 다른 논문을 골라보세요. (예: MCP → Model Context Protocol)",
            icon="⚠️",
        )

    # Stream the answer as it's generated; falls back to the non-streaming
    # 3-tier chain (main → simplified → abstract) on an empty/failed stream.
    # 스트리밍이 끝난 뒤 후처리로 바뀐 게 있으면 같은 자리(answer_slot)에 다시 그린다.
    answer_slot = st.empty()
    with answer_slot:
        full_answer = st.write_stream(pipeline.stream_answer(question, paper, lang))
    if isinstance(full_answer, str) and full_answer.strip():
        # 스트리밍 경로도 일반 경로와 같은 후처리를 거친다(번호 목록 → 불릿,
        # 앞머리에 따라 쓴 "[최종 답변]:" 제거, 중복 줄 제거).
        tidy = pipeline.dedup_bullets(
            strip_answer_echo(normalize_bullets(clean_llm_output(full_answer)))
        )
        if tidy != full_answer:
            full_answer = tidy
            answer_slot.markdown(full_answer)
    if not full_answer or not full_answer.strip():
        _, full_answer = pipeline.generate_answer(question, paper, lang)
        if not full_answer.strip():
            full_answer = (
                "⚠️ 응답을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요."
                if lang == "ko"
                else "⚠️ The model returned an empty response. Please try again."
            )
        st.markdown(full_answer)
    full_answer = normalize_bullets(clean_llm_output(full_answer))

    # 답변의 "X% 향상/감소"가 원문에서는 오류율 같은 지표 값 자체인 경우 경고한다.
    # (소형 모델이 "gains over the state of the art: 21.2% top-1 error"를
    # "기존 대비 21.2% 향상"으로 옮기는 오류를 프롬프트로는 막지 못했다)
    misreads = pipeline.metric_misreads(full_answer)
    if misreads:
        nums = ", ".join(f"{n}%" for n, _ in misreads)
        note = (
            f"⚠️ 수치 확인 필요: 답변의 {nums}는 원문에서 개선 폭이 아니라 지표 값 "
            f"자체로 보입니다. 원문: \"{misreads[0][1]}\""
        )
        st.warning(note)
        full_answer += "\n\n" + note

    # Citation card
    year = paper["published"][:4]
    authors_str = ", ".join(paper["authors"][:3])
    if len(paper["authors"]) > 3:
        authors_str += " et al."
    source_label = _source_label(paper)
    if paper.get("metadata_suspect"):
        st.caption(
            "⚠️ Source metadata (author/year) may be inconsistent for this paper — "
            "please verify against the original publisher or arXiv page before citing."
        )
    st.markdown(
        f"""
<div class="citation-card">
📚 <strong>Source</strong><br>
<em>{paper["title"]}</em><br>
{authors_str} · {year}<br>
<a href="{paper["url"]}" target="_blank">{source_label}</a>
</div>
        """,
        unsafe_allow_html=True,
    )
    return (
        f"{full_answer}\n\n---\n📚 **Source:** *{paper['title']}* "
        f"— {authors_str} ({year}) [[{source_label}]({paper['url']})]"
    )


# ---------------------------------------------------------------------------
# 대화 기록
# ---------------------------------------------------------------------------

_live_cands = (st.session_state.get("candidates") or {}).get("id")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 지난 검색의 후보 목록은 접어서 보여준다. 가장 최근 검색의 목록은
        # 아래 패널(버튼 포함)에 있으므로 여기서는 생략한다.
        if msg.get("candidates_md") and msg.get("candidates_id") != _live_cands:
            with st.expander("이때 검색된 후보 논문"):
                st.markdown(msg["candidates_md"])

user_input = st.chat_input("질문을 입력하세요  ·  Enter your question")
# 되묻기 버튼을 누르면 그 선택지가 다음 실행의 질문이 된다(on_click으로 저장).
# "그대로 검색"이면 되묻기를 건너뛴다.
_skip_clarify = False
if not user_input and "pending_query" in st.session_state:
    user_input = st.session_state.pop("pending_query")
    _skip_clarify = st.session_state.pop("skip_clarify", False)
_switch_to = None if user_input else st.session_state.pop("switch_to", None)


def _choose_clarification(query: str, skip: bool) -> None:
    st.session_state.pending_query = query
    st.session_state.skip_clarify = skip


if (user_input or _switch_to) and not _get_llm(_mtime("llm_client.py")).is_available():
    st.error(
        "Ollama is not running. Please start it with `ollama serve` and refresh.",
        icon="🔴",
    )
    st.stop()

# ---------------------------------------------------------------------------
# 후보 목록에서 다른 논문을 골랐을 때 — 같은 질문에 그 논문으로 다시 답한다
# ---------------------------------------------------------------------------
if _switch_to and st.session_state.get("candidates"):
    cands = st.session_state.candidates
    paper = next((p for p in cands["papers"] if p.get("id") == _switch_to), None)
    if paper:
        pipeline = _get_pipeline()
        question, lang = cands["question"], cands["lang"]
        request = f"↻ **{paper['title']}** 논문으로 다시 답해줘 — \"{question}\""
        st.session_state.messages.append({"role": "user", "content": request})
        with st.chat_message("user"):
            st.markdown(request)
        with st.chat_message("assistant"):
            with st.status("선택한 논문을 읽는 중...", expanded=True) as status:
                st.write("📥 PDF를 받아 인덱싱합니다...")
                n_chunks = pipeline.switch_paper(paper)
                if n_chunks == 0:
                    status.update(label="PDF를 받지 못했습니다", state="error")
                    fail = ("이 논문의 PDF를 받지 못했습니다. 원래 논문은 그대로 유지되니, "
                            "후보 목록에서 다른 논문을 골라주세요.")
                    st.error(fail, icon="❌")
                    st.session_state.messages.append({"role": "assistant", "content": fail})
                    st.stop()
                st.write(f"→ Indexed **{n_chunks:,}** chunks into FAISS.")
                status.update(label="Ready to answer...", state="complete")
            content = _render_answer(pipeline, question, paper, lang)
        cands["current_id"] = paper.get("id")
        st.session_state.messages.append({"role": "assistant", "content": content})

# ---------------------------------------------------------------------------
# 새 질문
# ---------------------------------------------------------------------------
if user_input:
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
        weak_match = False
        papers = None   # 새로 검색한 경우의 후보 — 답변 아래 목록에 쓴다

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

        elif not _skip_clarify and pipeline.is_vague_query(user_input):
            # 한 단어뿐인 질문 — 검색 전에 무엇이 궁금한지 되묻는다.
            # ("컴퓨터" → computer architecture로 검색 → 이미지 인식 논문이 뽑힌 사례)
            ask = (
                f"**'{user_input.strip()}'**만으로는 어떤 논문을 찾아야 할지 정하기 어렵습니다. "
                "궁금한 점을 조금 더 구체적으로 적어주시거나 아래에서 골라주세요."
                if lang == "ko" else
                f"**'{user_input.strip()}'** alone is too broad to pick a paper. "
                "Please add what you want to know, or choose below."
            )
            st.markdown(ask)
            options = pipeline.clarify_options(user_input, lang)
            for col, (i, (label, query)) in zip(st.columns(len(options)), enumerate(options)):
                col.button(
                    label, key=f"clarify_{len(st.session_state.messages)}_{i}",
                    on_click=_choose_clarification,
                    args=(query, i == len(options) - 1),   # 마지막 = 그대로 검색
                    use_container_width=True,
                )
            st.session_state.messages.append({"role": "assistant", "content": ask})
            st.stop()

        else:
            # New-topic path — full 2-step agentic loop
            with st.status("Agent is working...", expanded=True) as status:

                # Step 1a: Query refinement
                st.write("🔍 **Step 1** — Refining query into search keywords...")
                keywords = pipeline.refine_query(user_input)
                st.write(f"→ Keywords: `{keywords}`")
                recency_filter = pipeline.search_year
                if recency_filter and pipeline.wants_newest:
                    st.write(
                        f"→ 가장 최신 논문 요청 — {recency_filter[:4]}년 논문부터 찾고, "
                        "관련 있는 후보 중 가장 최근 논문을 고릅니다."
                    )
                elif recency_filter:
                    st.write(f"→ 최신 논문 요청 — {recency_filter[:4]}년 이후 논문만 검색합니다.")

                # Step 1b: Semantic Scholar search
                st.write(f"📄 Searching Semantic Scholar (rate limit: {_S2_DELAY:g} s/request)...")
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

                if recency_filter and pipeline.search_year != recency_filter:
                    widened = (
                        f"{pipeline.search_year[:4]}년 이후로 넓혀"
                        if pipeline.search_year else "연도 제한 없이"
                    )
                    st.write(
                        f"→ {recency_filter[:4]}년 이후 논문이 없어 {widened} 다시 검색했습니다."
                    )
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
                weak_match = pipeline.is_weak_match(reasoning)

                # Step 2b: Download and index PDF (실패 시 다음 후보로 자동 전환)
                st.write("📥 Downloading PDF and building FAISS index...")
                indexed_paper, n_chunks, substituted = pipeline.index_with_fallback(
                    papers, selected_paper, query=user_input
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
                        f"질문과 가장 가까운 다음 후보로 전환했습니다: **{sub_title}**"
                    )

                st.write(f"→ Indexed **{n_chunks:,}** chunks into FAISS.")
                status.update(label="Ready to answer...", state="complete")

        content = _render_answer(pipeline, user_input, selected_paper, lang, weak_match)

    message = {"role": "assistant", "content": content}
    if papers:
        # 새로 검색한 경우 — 후보 목록을 패널(버튼)과 대화 기록(텍스트) 양쪽에 남긴다.
        # 후보 목록이 요청 자체에 대한 답인 "논문 찾아줘"·최신 요청이면 펼쳐서 보여준다.
        shown = pipeline.candidates_for_display(papers)
        cand_id = st.session_state.get("candidates_seq", 0) + 1
        st.session_state.candidates_seq = cand_id
        st.session_state.candidates = {
            "id": cand_id, "question": user_input, "lang": lang, "papers": shown,
            "current_id": selected_paper.get("id"),
            "by_date": bool(pipeline.search_year),
            "expanded": pipeline.is_paper_request(user_input) or bool(pipeline.search_year),
        }
        message["candidates_md"] = _candidates_md(shown, selected_paper.get("id"))
        message["candidates_id"] = cand_id
    st.session_state.messages.append(message)

# 가장 최근 검색의 후보 목록 — 항상 대화 맨 아래에 둔다
_render_candidate_panel()
