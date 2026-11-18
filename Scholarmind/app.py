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

import html
import json
import logging
import os
import re
import sys
from datetime import datetime

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
from rag_pipeline import (CHUNK_OVERLAP, CHUNK_SIZE, MIN_CANDIDATE_POOL, SUMMARY_LIMIT_EN, SUMMARY_LIMIT_KO,
                          TOP_K, RAGPipeline, year_label)
from semantic_scholar_tool import RATE_LIMIT_DELAY, RATE_LIMIT_DELAY_KEYED, S2RateLimitError
from embedder import Embedder
from pdf_extractor import clean_llm_output
from llm_client import normalize_bullets, strip_answer_echo
from timing import StageTimer

logging.basicConfig(level=logging.INFO)

# 화면 안내용 — S2 키가 있으면 1.1초, 없으면 3초 간격 (semantic_scholar_tool 참고)
_S2_DELAY = RATE_LIMIT_DELAY_KEYED if os.environ.get("S2_API_KEY") else RATE_LIMIT_DELAY

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ScholarMind",
    page_icon=":material/auto_stories:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — minimalist, generous whitespace
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    h1 { letter-spacing: -0.8px; }
    .block-container { max-width: 820px; padding-top: 2.5rem; }
    [data-testid="stSidebar"] h1 {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    [data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.25rem 0;
        margin-bottom: 1.5rem;
    }
    /* Material Symbols — Streamlit이 내장한 글꼴을 HTML 카드 안에서도 쓴다 */
    .msr {
        font-family: 'Material Symbols Rounded';
        font-weight: normal; font-style: normal; font-size: 1.15em;
        line-height: 1; vertical-align: -0.2em; margin-right: 0.3rem;
        -webkit-font-feature-settings: 'liga'; font-feature-settings: 'liga';
    }
    hr { opacity: 0.18; }
    [data-testid="stStatusWidget"] { border-radius: 8px; }
    code {
        font-size: 0.82rem;
        padding: 1px 4px;
        border-radius: 3px;
    }
    .citation-card {
        background: #fdeee9;
        padding: 0.75rem 1rem;
        border-radius: 14px;
        margin-top: 1rem;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .citation-card .msr { color: #E8553D; }
    .citation-card .meta { color: #8f8a83; font-size: 0.82rem; }
    .evidence {
        background: #f3f1ed;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        margin: 0.25rem 0 0.6rem 0;
        font-size: 0.82rem;
        line-height: 1.55;
        white-space: pre-wrap;
    }
    .hero { text-align: center; margin: 18vh 0 1.75rem 0; }
    .hero .logo { font-family: 'Material Symbols Rounded'; font-size: 2.6rem; line-height: 1; color: #E8553D;
                  -webkit-font-feature-settings: 'liga'; font-feature-settings: 'liga'; }
    .hero h1 { font-size: 2.6rem; font-weight: 700; letter-spacing: -1.2px; margin: 0.35rem 0 0.4rem 0; padding: 0; }
    .hero p { color: #8f8a83; font-size: 1rem; margin: 0; }
    .topbar { display: flex; align-items: center; gap: 0.4rem; font-weight: 700; font-size: 1.05rem; margin-bottom: 1rem; }
    .topbar .msr { color: #E8553D; }
    /* 첫 화면 입력창 — 화면 가운데, 크게 */
    .st-key-hero_q [data-testid="stChatInput"] { border-radius: 22px; }
    .st-key-hero_q textarea { min-height: 3.2rem; font-size: 1rem; }
    /* 예시 질문 카드 */
    .stButton > button { border-radius: 12px; }
    hr { opacity: 0.12; }
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
    return "Semantic Scholar에서 보기"


_init_state()

_AVATAR = {"assistant": ":material/school:", "user": ":material/person:"}


def _chat(role: str):
    """대화 말풍선 — 아바타는 Google Material Symbols 아이콘."""
    return st.chat_message(role, avatar=_AVATAR.get(role))


def _reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.candidates = None
    st.session_state.upload = None
    st.session_state.upload_key = None
    st.session_state.uploader_n = st.session_state.get("uploader_n", 0) + 1  # 업로드 칸 비우기
    _get_pipeline().reset()


def _ask_example(query: str) -> None:
    st.session_state.pending_query = query
    st.session_state.skip_clarify = False


def _save_feedback(idx: int) -> None:
    """답변 아래 👍/👎를 logs/feedback.jsonl에 남긴다 — 답변 품질 평가 자료."""
    value = st.session_state.get(f"fb_{idx}")
    msgs = st.session_state.messages
    if value is None or idx >= len(msgs):
        return
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "question": msgs[idx].get("question", ""),
        "paper": msgs[idx].get("paper_title", ""),
        "good": value == 1,
    }
    try:
        os.makedirs(os.path.join(_HERE, "logs"), exist_ok=True)
        with open(os.path.join(_HERE, "logs", "feedback.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _feedback(idx: int) -> None:
    st.feedback("thumbs", key=f"fb_{idx}", on_change=_save_feedback, args=(idx,))


_FOLLOWUPS_KO = (("요약", ":material/summarize:", "이 논문 요약해줘"),
                 ("실험 결과", ":material/science:", "이 논문의 실험 결과는?"),
                 ("한계", ":material/report:", "이 논문의 한계는?"))
_FOLLOWUPS_EN = (("Summary", ":material/summarize:", "Summarize this paper"),
                 ("Results", ":material/science:", "What are the results of this paper?"),
                 ("Limitations", ":material/report:", "What are the limitations of this paper?"))


def _followups(idx: int, question: str) -> None:
    """답변 아래 이어서 물을 질문 버튼 — 누르면 바로 질문한다."""
    ko = any("가" <= ch <= "힣" for ch in question)
    for col, (i, (label, icon, q)) in zip(st.columns(3), enumerate(_FOLLOWUPS_KO if ko else _FOLLOWUPS_EN)):
        col.button(label, key=f"fu_{idx}_{i}", icon=icon, use_container_width=True, help=q,
                   on_click=_ask_example, args=(q,))


def _render_evidence(chunks: list) -> None:
    """모델이 실제로 읽은 원문 문단 — 답변이 원문에 있는 내용인지 직접 확인할 수 있다."""
    if not chunks:
        return
    with st.expander(f"근거 문단 보기 · 모델이 읽은 원문 {len(chunks)}개", icon=":material/format_quote:"):
        for i, text in enumerate(chunks, 1):
            body = html.escape(text[:900] + ("…" if len(text) > 900 else ""))
            st.markdown(f"**[{i}]**<div class='evidence'>{body}</div>", unsafe_allow_html=True)


_LABELS = r"(?:핵심 아이디어|방법|결과|한계|Key idea|Method|Results|Limitations)"
_SUMMARY_LABEL = re.compile(rf"^(\s*(?:-\s*)?)(?:\*\*)?({_LABELS})(?:\*\*)?\s*:(?:\*\*)?\s*(?:\2\s*:\s*)?", re.MULTILINE)


def _bold_summary_labels(text: str) -> str:
    """구조화 요약의 항목 이름(핵심 아이디어/방법/결과/한계)을 굵게, '한계: 한계:' 같은 반복은 하나로."""
    text = _SUMMARY_LABEL.sub(lambda m: f"- **{m.group(2)}**: ", text)
    return _trim_summary(text)


# "명시적으로 제시되지 않았으나 ~가 필요할 수 있다"도 잡는다(관찰: 예전 식은 이 표현을 놓침)
_NOT_STATED = re.compile(r"명시[^.]{0,15}?(?:않|없)|not stated", re.IGNORECASE)


def _trim_summary(text: str) -> str:
    """
    2.4B 모델은 "1–2문장"·"한계를 추측하지 말 것"을 자주 어긴다(관찰: 한계가 없다고
    쓴 뒤 "~할 수 있다"는 추측을 덧붙임). 프롬프트로 못 막는 부분을 후처리로 고정한다:
    항목마다 앞 두 문장만 남기고, 한계가 '명시되지 않음'이면 그 문장만 남긴다.
    """
    out = []
    for line in text.split("\n"):
        if line.startswith("- **") and "**: " in line:
            head, body = line.split("**: ", 1)
            label = head[4:]
            if label in ("한계", "Limitations") and _NOT_STATED.search(body):
                # 파이프라인이 붙이는 고정 문장과 같게 ('한계' 버튼 안내 포함)
                body = (SUMMARY_LIMIT_KO if label == "한계" else SUMMARY_LIMIT_EN).split(": ", 1)[1]
            else:
                sents = re.split(r"(?<=[.!?])\s+", body.strip())
                # 출력 길이 상한(SUMMARY_MAX_TOKENS)에 걸려 잘린 마지막 문장은 뺀다
                if len(sents) > 1 and not re.search(r"[.!?]$", sents[-1]):
                    sents = sents[:-1]
                body = " ".join(sents[:2])
            line = f"{head}**: {body}"
        out.append(line)
    return "\n".join(out)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("ScholarMind")
    st.caption("로컬 환경에서 실행되는 논문 검색 시스템")

    if st.button("새 대화", icon=":material/edit_square:", use_container_width=True, type="primary"):
        _reset_conversation()
        st.rerun()

    st.divider()
    st.markdown("**:material/upload_file: 논문 PDF로 질문하기**")
    uploaded = st.file_uploader(
        "논문 PDF 올리기", type=["pdf"], label_visibility="collapsed",
        key=f"uploader_{st.session_state.get('uploader_n', 0)}",
    )
    st.caption("PDF를 올리면 검색 없이 그 논문에 대해 답합니다.")

    st.divider()
    st.markdown("**:material/route: 이렇게 동작해요**")
    st.markdown(
        "1. 한국어 질문을 영어 검색어로 바꿔 논문 찾기\n"
        "2. 가장 맞는 논문의 원문 PDF 읽기\n"
        "3. 원문 문단을 근거로 출처와 함께 답변 생성"
    )

    llm_check = _get_llm(_mtime("llm_client.py"))
    ollama_ok = llm_check.is_available()
    available_models = llm_check.list_models() if ollama_ok else []
    model_present = any(LLM_MODEL in m for m in available_models)

    st.divider()
    if not ollama_ok:
        st.error("로컬 모델 서버(Ollama)가 꺼져 있습니다.\n\n`ollama serve`로 켜 주세요.", icon=":material/error:")
    elif not model_present:
        st.warning(f"`{LLM_MODEL}` 모델이 없습니다.\n\n`ollama pull {LLM_MODEL}`", icon=":material/warning:")
    else:
        st.caption(f":material/check_circle: 로컬 모델 연결됨 · {LLM_MODEL}")

    with st.expander("기타 정보", icon=":material/code:"):
        st.markdown(
            f"""
| 항목 | 값 |
|---|---|
| 언어 모델 | `{LLM_MODEL}` |
| 임베딩 | `{EMBED_MODEL}` ({EMBEDDING_DIM}차원) |
| 근거 문단 수 | 단순 질문 3 · 복잡한 질문 {TOP_K} |
| 문단 크기 | {CHUNK_SIZE:,}자 (겹침 {CHUNK_OVERLAP:,}자) |
| 답변 온도 | 0.1 (논문 선택 0.3, 기타 {TEMPERATURE}) |
| 검색 | FAISS `IndexFlatIP` + BM25, RRF 융합 |
| PDF 출처 | arXiv · openAccessPdf · Unpaywall · CORE |
| Ollama | `{OLLAMA_BASE_URL}` · 제한 {OLLAMA_TIMEOUT}초 |
| 검색 간격 | {_S2_DELAY:g}초 |
            """
        )

# ---------------------------------------------------------------------------
# MAIN AREA
# ---------------------------------------------------------------------------

# 빈 화면(첫 질문 전)인지 — 이때는 가운데 로고와 입력창을 보여준다
_EMPTY = (not st.session_state.messages and "pending_query" not in st.session_state
          and "switch_to" not in st.session_state)
_topbar = st.empty()


def _show_topbar() -> None:
    _topbar.markdown("<div class='topbar'><span class='msr'>auto_stories</span>ScholarMind</div>",
                     unsafe_allow_html=True)


if not _EMPTY:
    _show_topbar()

# ---------------------------------------------------------------------------
# PDF 업로드 — 새 파일이 올라오면 그 논문을 현재 논문으로 인덱싱한다
# ---------------------------------------------------------------------------
if uploaded is not None:
    _key = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("upload_key") != _key:
        st.session_state.upload_key = _key
        with st.spinner("PDF를 읽고 문단으로 나누는 중..."):
            _paper, _n = _get_pipeline().index_uploaded_pdf(uploaded.getvalue(), uploaded.name)
        if _n:
            st.session_state.upload = _paper
            st.session_state.candidates = None
            st.session_state.messages.append({
                "role": "assistant",
                "content": f":material/description: **{_paper['title']}** 을(를) 읽었습니다 ({_n}개 문단). "
                           "이 논문에 대해 무엇이든 물어보세요. 요약도 할 수 있습니다.",
            })
        else:
            st.session_state.upload = None
            st.error("PDF에서 글자를 읽지 못했습니다. 스캔 이미지로만 된 PDF는 지원하지 않습니다.", icon=":material/error:")

_hero = st.empty()   # 질문이 시작되면 비운다
_hero_q = None
_HERO_SHOWN = _EMPTY and not st.session_state.messages   # 업로드로 메시지가 막 생긴 경우 제외
if _HERO_SHOWN:
  with _hero.container():
    st.markdown(
        "<style>[data-testid='stBottom'] { display: none; }</style>"
        "<div class='hero'><div class='logo'>auto_stories</div><h1>ScholarMind</h1>"
        "<p>질문을 하면 논문을 찾아 읽고 출처와 함께 답합니다</p></div>",
        unsafe_allow_html=True,
    )
    # 컨테이너 안의 chat_input은 화면 아래가 아니라 이 자리에 놓인다
    _hero_q = st.chat_input("궁금한 논문의 주제를 물어보세요   ·   예: MCP가 뭐야?  RAG에 대해 설명해줘",
                            key="hero_q")
def _leave_hero() -> None:
    """첫 화면을 떠난다 — 가운데 블록을 비우고(아래 입력창이 다시 보임) 작은 로고를 그린다."""
    if _HERO_SHOWN:
        _hero.empty()
        _show_topbar()


if user_input_pending := (st.session_state.get("pending_query") or st.session_state.get("switch_to")):
    _leave_hero()


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


def _save_candidates(pipeline, papers: list, question: str, lang: str, current_id) -> tuple:
    """가장 최근 검색의 후보를 패널용으로 저장한다. (보여줄 후보, 후보 id)를 돌려준다."""
    shown = pipeline.candidates_for_display(papers)
    cand_id = st.session_state.get("candidates_seq", 0) + 1
    st.session_state.candidates_seq = cand_id
    st.session_state.candidates = {
        "id": cand_id, "question": question, "lang": lang, "papers": shown,
        "current_id": current_id,
        "by_date": bool(pipeline.search_year),
        # 후보 목록이 요청 자체에 대한 답인 "논문 찾아줘"·최신 요청이면 펼쳐서 보여준다
        "expanded": pipeline.is_paper_request(question) or bool(pipeline.search_year),
    }
    return shown, cand_id


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
        f"다른 논문으로 답하기 · 후보 {len(papers)}편 ({order_note}) — 질문: \"{cands['question']}\"",
        expanded=cands.get("expanded", False), icon=":material/swap_horiz:",
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


def _timed(stream, timer: StageTimer):
    """첫 글자가 나오는 순간을 기록한다 — 그때까지가 근거 검색과 프롬프트 읽기 시간이다."""
    first = True
    for piece in stream:
        if first:
            timer.lap("근거 검색 · 첫 글자")
            first = False
        yield piece


def _show_timing(timer: StageTimer, question: str) -> str:
    """단계별 시간표를 답변 아래에 보여주고 logs/timings.jsonl에 남긴다. 기록용 표를 돌려준다."""
    table = timer.markdown()
    with st.expander(f"단계별 소요 시간 · 총 {timer.total:.1f}초", icon=":material/timer:"):
        st.markdown(table)
    timer.save(question)
    return table


def _render_answer(pipeline: RAGPipeline, question: str, paper: dict, lang: str,
                   weak_match: bool = False, timer: StageTimer = None) -> str:
    """
    답변을 스트리밍으로 보여주고 후처리·경고·출처 카드를 붙인다.
    새 검색, 후속 질문, '다시 답하기'가 모두 이 함수를 쓴다.
    (대화 기록에 남길 답변 텍스트, 출처 카드 HTML)을 돌려준다.
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
            icon=":material/warning:",
        )

    # Stream the answer as it's generated; falls back to the non-streaming
    # 3-tier chain (main → simplified → abstract) on an empty/failed stream.
    # 스트리밍이 끝난 뒤 후처리로 바뀐 게 있으면 같은 자리(answer_slot)에 다시 그린다.
    answer_slot = st.empty()
    with answer_slot:
        stream = pipeline.stream_answer(question, paper, lang)
        full_answer = st.write_stream(_timed(stream, timer) if timer else stream)
    if isinstance(full_answer, str) and full_answer.strip():
        # 스트리밍 경로도 일반 경로와 같은 후처리를 거친다(번호 목록 → 불릿,
        # 앞머리에 따라 쓴 "[최종 답변]:" 제거, 중복 줄 제거).
        tidy = pipeline.dedup_bullets(
            strip_answer_echo(normalize_bullets(clean_llm_output(full_answer)))
        )
        tidy = _bold_summary_labels(tidy)
        if tidy != full_answer:
            full_answer = tidy
            answer_slot.markdown(full_answer)
    if not full_answer or not full_answer.strip():
        _, full_answer = pipeline.generate_answer(question, paper, lang)
        if not full_answer.strip():
            full_answer = (
                "응답을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요."
                if lang == "ko"
                else "The model returned an empty response. Please try again."
            )
        st.markdown(full_answer)
    full_answer = normalize_bullets(clean_llm_output(full_answer))
    if timer:
        timer.lap("답변 출력")

    # 답변의 "X% 향상/감소"가 원문에서는 오류율 같은 지표 값 자체인 경우 경고한다.
    # (소형 모델이 "gains over the state of the art: 21.2% top-1 error"를
    # "기존 대비 21.2% 향상"으로 옮기는 오류를 프롬프트로는 막지 못했다)
    misreads = pipeline.metric_misreads(full_answer)
    if misreads:
        nums = ", ".join(f"{n}%" for n, _ in misreads)
        note = (
            f":material/warning: 수치 확인 필요: 답변의 {nums}는 원문에서 개선 폭이 아니라 지표 값 "
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
            ":material/warning: 이 논문의 저자·연도 정보가 원문과 다를 수 있습니다. "
            "인용하기 전에 출판사나 arXiv 페이지에서 확인해 주세요."
        )
    if paper.get("uploaded"):
        meta = f"업로드한 PDF · {html.escape(paper.get('filename', ''))}"
        link = ""
    else:
        bits = [authors_str, year]
        if paper.get("venue"):
            bits.append(paper["venue"])
        if paper.get("citation_count") is not None:
            bits.append(f"인용 {paper['citation_count']:,}회")
        meta = " · ".join(html.escape(str(b)) for b in bits if b)
        link = f'<br><a href="{paper["url"]}" target="_blank">{source_label}</a>'
    card = (
        f'<div class="citation-card"><span class="msr">school</span><strong>출처</strong><br>'
        f'<em>{html.escape(paper["title"])}</em><br><span class="meta">{meta}</span>{link}</div>'
    )
    st.markdown(card, unsafe_allow_html=True)
    _render_evidence([c["chunk"] for c in getattr(pipeline, "last_evidence", [])])
    # 카드도 함께 돌려준다 — 대화 기록에서 같은 카드로 다시 그린다(예전엔 기록에서만 한 줄 텍스트로 보임)
    return full_answer, card


# ---------------------------------------------------------------------------
# 대화 기록
# ---------------------------------------------------------------------------

_live_cands = (st.session_state.get("candidates") or {}).get("id")
for _idx, msg in enumerate(st.session_state.messages):
    with _chat(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("card"):
            st.markdown(msg["card"], unsafe_allow_html=True)
        if msg.get("evidence"):
            _render_evidence(msg["evidence"])
        # 지난 검색의 후보 목록은 접어서 보여준다. 가장 최근 검색의 목록은
        # 아래 패널(버튼 포함)에 있으므로 여기서는 생략한다.
        if msg.get("timing_md"):
            with st.expander("단계별 소요 시간", icon=":material/timer:"):
                st.markdown(msg["timing_md"])
        if msg.get("candidates_md") and msg.get("candidates_id") != _live_cands:
            with st.expander("이때 검색된 후보 논문", icon=":material/history:"):
                st.markdown(msg["candidates_md"])
        if msg["role"] == "assistant" and msg.get("question"):
            _feedback(_idx)
            if _idx == len(st.session_state.messages) - 1:
                _followups(_idx, msg["question"])

# 아래 입력창은 항상 만든다(첫 화면에서는 위 스타일로 숨김). 첫 화면에선 가운데 입력창 값을 쓴다.
_bottom_q = st.chat_input(
    "업로드한 논문에 대해 물어보세요" if st.session_state.get("upload")
    else "궁금한 논문 주제를 한국어로 물어보세요"
)
user_input = _hero_q or _bottom_q
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
        "로컬 모델 서버(Ollama)가 꺼져 있습니다. `ollama serve`로 켠 뒤 새로고침해 주세요.",
        icon=":material/error:",
    )
    st.stop()

# ---------------------------------------------------------------------------
# 후보 목록에서 다른 논문을 골랐을 때 — 같은 질문에 그 논문으로 다시 답한다
# ---------------------------------------------------------------------------
if _switch_to and st.session_state.get("candidates"):
    cands = st.session_state.candidates
    paper = next((p for p in cands["papers"] if p.get("id") == _switch_to), None)
    if paper:
        timer = StageTimer("다시 답하기")
        pipeline = _get_pipeline()
        question, lang = cands["question"], cands["lang"]
        request = f"↻ **{paper['title']}** 논문으로 다시 답해줘 — \"{question}\""
        st.session_state.messages.append({"role": "user", "content": request})
        with _chat("user"):
            st.markdown(request)
        with _chat("assistant"):
            with st.status("선택한 논문을 읽는 중...", expanded=True) as status:
                st.write("원문 PDF를 받아 문단으로 나누는 중...")
                n_chunks = pipeline.switch_paper(paper)
                timer.lap("PDF·인덱싱")
                if n_chunks == 0:
                    status.update(label="PDF를 받지 못했습니다", state="error")
                    fail = ("이 논문의 PDF를 받지 못했습니다. 원래 논문은 그대로 유지되니, "
                            "후보 목록에서 다른 논문을 골라주세요.")
                    st.error(fail, icon=":material/error:")
                    st.session_state.messages.append({"role": "assistant", "content": fail})
                    st.stop()
                st.write(f"→ {n_chunks:,}개 문단으로 나눠 인덱싱 · {timer.laps[-1][1]:.1f}초")
                status.update(label="답변 준비 완료", state="complete")
            content, card = _render_answer(pipeline, question, paper, lang, timer=timer)
            timing_md = _show_timing(timer, question)
            _feedback(len(st.session_state.messages))
            _followups(len(st.session_state.messages), question)
        cands["current_id"] = paper.get("id")
        st.session_state.messages.append(
            {"role": "assistant", "content": content, "card": card, "timing_md": timing_md,
             "question": question, "paper_title": paper["title"],
             "evidence": [c["chunk"] for c in getattr(pipeline, "last_evidence", [])]})

# ---------------------------------------------------------------------------
# 새 질문
# ---------------------------------------------------------------------------
if user_input:
    _leave_hero()
    timer = StageTimer("새 검색")
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with _chat("user"):
        st.markdown(user_input)

    # Assistant turn
    with _chat("assistant"):
        pipeline = _get_pipeline()
        lang = pipeline.detect_language(user_input)

        # Shared output variables — set in whichever branch runs
        selected_paper = None
        weak_match = False
        papers = None   # 새로 검색한 경우의 후보 — 답변 아래 목록에 쓴다
        sub_note = None  # PDF를 못 받아 다른 논문으로 바뀐 경우의 안내
        pdf_failed = False

        # ----------------------------------------------------------------
        # Context-aware routing
        # Keyword heuristics run first (zero LLM cost).  LLM classification
        # fires only for ambiguous queries.  If a paper is already indexed
        # and the user is continuing that conversation, bypass the entire
        # Semantic Scholar search + PDF download pipeline.
        # ----------------------------------------------------------------
        # 업로드한 PDF가 있으면 모든 질문을 그 논문에 대한 질문으로 본다 ("새 대화"로 해제)
        _up = st.session_state.get("upload")
        _cur = pipeline.current_paper
        followup = bool(_up and _cur and _cur.get("uploaded")) or pipeline.is_followup(user_input)
        timer.lap("질문 판별")
        if followup:
            timer.kind = "후속 질문"

        if followup:
            # Follow-up path — reuse existing FAISS index
            selected_paper = pipeline.current_paper
            title_preview = (
                selected_paper["title"][:75] + "..."
                if len(selected_paper["title"]) > 75
                else selected_paper["title"]
            )

            with st.status("이어지는 질문 — 읽은 논문으로 바로 답합니다", expanded=True) as status:
                st.write(":material/reply: 검색 없이 지금 읽은 논문을 다시 씁니다.")
                st.write(f"→ 논문: **{title_preview}**")
                st.write("관련 문단을 찾는 중...")
                status.update(label="답변 준비 완료 · 검색 생략", state="complete")

        elif pipeline.current_paper is None and pipeline.refers_to_current_paper(user_input):
            # 아직 읽은 논문이 없는데 "이 논문 요약해줘" — 엉뚱한 검색 대신 안내한다
            note = ("아직 읽은 논문이 없습니다. 먼저 궁금한 주제를 질문하시거나, "
                    "왼쪽에서 논문 PDF를 올린 뒤 다시 물어봐 주세요."
                    if lang == "ko" else
                    "No paper has been read yet. Ask about a topic first, or upload a PDF on the left.")
            st.markdown(note)
            st.session_state.messages.append({"role": "assistant", "content": note})
            st.stop()

        elif not _skip_clarify and pipeline.is_topicless_query(user_input):
            # 주제가 빠진 질문 — 검색 전에 분야를 되묻는다
            ask = (f"'**{user_input.strip()}**'에는 어떤 분야인지가 빠져 있어서 맞는 논문을 고르기 어렵습니다. "
                   "궁금한 분야를 함께 적어 주시거나 아래에서 골라주세요. (예: RAG 최신 연구 동향 알려줘)"
                   if lang == "ko" else
                   f"'**{user_input.strip()}**' has no topic. Add a field, or choose below.")
            st.markdown(ask)
            options = pipeline.topic_options(user_input, lang)
            for col, (i, (label, query)) in zip(st.columns(len(options)), enumerate(options)):
                col.button(label, key=f"topic_{len(st.session_state.messages)}_{i}",
                           on_click=_choose_clarification, args=(query, True), use_container_width=True)
            st.session_state.messages.append({"role": "assistant", "content": ask})
            st.stop()

        elif not _skip_clarify and pipeline.is_vague_query(user_input):
            # 한 단어뿐인 질문 — 검색 전에 무엇이 궁금한지 되묻는다.
            # ("컴퓨터" → computer architecture로 검색 → 이미지 인식 논문이 뽑힌 사례)
            ask = (
                f"'**{user_input.strip()}**'만으로는 어떤 논문을 찾아야 할지 정하기 어렵습니다. "
                "궁금한 점을 조금 더 구체적으로 적어주시거나 아래에서 골라주세요."
                if lang == "ko" else
                f"'**{user_input.strip()}**' alone is too broad to pick a paper. "
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
            with st.status("논문을 찾아 읽는 중...", expanded=True) as status:

                # Step 1a: Query refinement
                st.write(":material/search: **검색어 만들기** — 질문을 영어 검색어로 바꾸는 중...")
                keywords = pipeline.refine_query(user_input)
                timer.lap("검색어 생성")
                st.write(f"→ 검색어: `{keywords}` · {timer.laps[-1][1]:.1f}초")
                recency_filter = pipeline.search_year
                if recency_filter and pipeline.wants_newest:
                    st.write(
                        f"→ 가장 최신 논문 요청 — {recency_filter[:4]}년 논문부터 찾고, "
                        "관련 있는 후보 중 가장 최근 논문을 고릅니다."
                    )
                elif recency_filter:
                    st.write(f"→ 연도 조건 — {year_label(recency_filter)} 논문만 검색합니다.")

                # Step 1b: Semantic Scholar search
                st.write(":material/travel_explore: **논문 검색** — Semantic Scholar에서 찾는 중...")
                try:
                    papers, effective_kw = pipeline.search_papers(keywords, max_results=10)
                    timer.lap("논문 검색")
                except S2RateLimitError:
                    # 429는 "결과 없음"이 아니라 API 차단 — 질문을 바꾸라고 안내하면 안 된다.
                    status.update(label="API 요청 한도 초과", state="error")
                    if os.environ.get("S2_API_KEY"):
                        st.error(
                            "**논문 검색 서비스(Semantic Scholar)가 잠시 요청을 막았습니다 (429).**\n\n"
                            "API 키는 정상입니다. 키의 한도가 초당 1회 정도라 질문이 몰리면 잠깐 막힙니다. "
                            "**20~30초 뒤에 같은 질문을 다시 입력해 주세요.** 한 번 검색한 질문은 저장돼서 "
                            "다음부터는 이 오류 없이 바로 진행됩니다.",
                            icon=":material/block:",
                        )
                        st.stop()
                    st.error(
                        "**Semantic Scholar API 요청 한도(429)에 도달했습니다.**\n\n"
                        "질문의 문제가 아니라 API 호출이 차단된 상태입니다. "
                        "`S2_API_KEY` 환경변수가 설정되지 않으면 전 세계 익명 사용자가 "
                        "공유하는 요청 풀을 사용하게 되어 자주 차단됩니다.\n\n"
                        "**해결 방법:** 무료 API 키를 발급받아 `S2_API_KEY` 환경변수에 "
                        "설정한 뒤 프로그램을 다시 실행하세요 — "
                        "https://www.semanticscholar.org/product/api#api-key-form",
                        icon=":material/block:",
                    )
                    st.stop()

                if not papers:
                    status.update(label="논문을 찾지 못했습니다", state="error")
                    st.error(
                        "대체 검색어로도 논문을 찾지 못했습니다. 질문을 바꿔서 다시 물어봐 주세요.",
                        icon=":material/error:",
                    )
                    st.stop()

                if recency_filter and pipeline.search_year != recency_filter:
                    widened = (
                        f"{year_label(pipeline.search_year)}로 넓혀"
                        if pipeline.search_year else "연도 제한 없이"
                    )
                    st.write(
                        f"→ {year_label(recency_filter)} 논문이 없어 {widened} 다시 검색했습니다."
                    )
                if effective_kw != keywords:
                    st.write(f"→ 대체 검색어 사용: `{effective_kw}`")
                st.write(f"→ 후보 **{len(papers)}**편 · {timer.laps[-1][1]:.1f}초")
                if len(papers) < MIN_CANDIDATE_POOL:
                    st.caption("후보가 적습니다. 더 구체적인 용어로 물어보면 후보가 늘어납니다.")

                # Step 2a: Paper selection
                st.write(":material/fact_check: **논문 고르기** — 초록을 읽고 가장 맞는 논문을 고르는 중...")
                selected_paper, reasoning = pipeline.select_paper(user_input, papers)
                timer.lap("논문 선택")
                title_short = (
                    selected_paper["title"][:80] + "..."
                    if len(selected_paper["title"]) > 80
                    else selected_paper["title"]
                )
                st.write(f"→ 선택: **{title_short}** · {timer.laps[-1][1]:.1f}초")
                st.write(f"→ 고른 이유: *{reasoning}*")
                weak_match = pipeline.is_weak_match(reasoning)

                # Step 2b: Download and index PDF (실패 시 다음 후보로 자동 전환)
                st.write(":material/description: **원문 읽기** — PDF를 받아 문단으로 나누는 중...")
                indexed_paper, n_chunks, substituted = pipeline.index_with_fallback(
                    papers, selected_paper, query=user_input
                )
                timer.lap("PDF·인덱싱")

                if n_chunks == 0:
                    status.update(label="PDF를 받지 못했습니다", state="error")
                    pdf_failed = True
                else:
                    if substituted:
                        # 원래 고른 논문의 PDF를 못 구해서 다른 논문으로 바뀐 경우 —
                        # 인용 카드가 실제 인덱싱된 논문을 가리키도록 교체하고 사용자에게 알림
                        orig_title = selected_paper["title"]
                        selected_paper = indexed_paper
                        sub_title = (
                            indexed_paper["title"][:80] + "..."
                            if len(indexed_paper["title"]) > 80
                            else indexed_paper["title"]
                        )
                        # 접힌 상태 상자 안에만 쓰면 못 보고 지나친다(관찰) — 답변 위에 경고로도 보여준다
                        sub_note = (
                            # 굵은 글씨 바로 뒤에 한글이 오면 제목이 '...'로 끝날 때 **가 그대로 보인다(관찰) → 뒤에 공백
                            f"처음 고른 **{orig_title}** 논문은 PDF를 받지 못해(사이트가 다운로드를 막음) "
                            f"다음 후보인 **{sub_title}** 논문으로 답합니다. 아래 후보 목록에서 다른 논문을 고를 수 있습니다."
                        )
                        st.write(f":material/warning: {sub_note}")

                    st.write(f"→ {n_chunks:,}개 문단으로 나눠 인덱싱 · {timer.laps[-1][1]:.1f}초")
                    status.update(label="답변 준비 완료", state="complete")

        if pdf_failed:
            # 예전엔 여기서 멈춰 후보 목록이 안 보였다 — 다른 논문을 고를 길이 없었음(관찰)
            fail = ("상위 후보 논문들의 PDF를 받지 못했습니다(사이트가 다운로드를 막음). "
                    "아래 후보 목록에서 다른 논문을 골라 주세요.")
            st.error(fail, icon=":material/error:")
            _save_candidates(pipeline, papers, user_input, lang, None)
            st.session_state.messages.append({"role": "assistant", "content": fail})
            _render_candidate_panel()
            st.stop()
        if sub_note:
            st.warning(sub_note, icon=":material/warning:")
        content, card = _render_answer(pipeline, user_input, selected_paper, lang, weak_match, timer)
        if sub_note:
            content = f":material/warning: {sub_note}\n\n{content}"
        timing_md = _show_timing(timer, user_input)
        _feedback(len(st.session_state.messages))
        _followups(len(st.session_state.messages), user_input)

    message = {"role": "assistant", "content": content, "card": card, "timing_md": timing_md,
               "question": user_input, "paper_title": selected_paper["title"],
               "evidence": [c["chunk"] for c in getattr(pipeline, "last_evidence", [])]}
    if papers:
        # 새로 검색한 경우 — 후보 목록을 패널(버튼)과 대화 기록(텍스트) 양쪽에 남긴다.
        shown, cand_id = _save_candidates(pipeline, papers, user_input, lang, selected_paper.get("id"))
        message["candidates_md"] = _candidates_md(shown, selected_paper.get("id"))
        message["candidates_id"] = cand_id
    st.session_state.messages.append(message)

# 가장 최근 검색의 후보 목록 — 항상 대화 맨 아래에 둔다
_render_candidate_panel()
