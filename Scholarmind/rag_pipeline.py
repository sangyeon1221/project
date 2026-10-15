"""
rag_pipeline.py
---------------
Orchestrator for the ScholarMind 2-step loop (search & select → read & answer).

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

Each public method is intentionally small so app.py can interleave
Streamlit status updates between the steps.
"""

import hashlib
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from semantic_scholar_tool import SemanticScholarTool, S2RateLimitError
from embedder import Embedder
from llm_client import LLMClient, fix_decimal_colons
from pdf_extractor import extract_text as extract_pdf_text
from vector_store import VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking parameters
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = 1000       # characters
CHUNK_OVERLAP: int = 150     # characters
TOP_K: int = 5         # 내장 그래픽 가속으로 읽기가 5배 빨라져 3 → 5로 다시 늘림 (근거 확대)
CANDIDATE_K: int = 20  # 밀집·BM25 각 순위에서 융합에 넣을 후보 수

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


# 참고문헌 청크 판별 — 참고문헌은 다른 논문 제목이 빽빽하게 나열된 부분이라
# 키워드가 잘 걸린다. 그대로 인덱싱하면 인용된 다른 논문을 이 논문의 주제인
# 것처럼 답하게 된다(관찰: 참고문헌에 한 번 나온 벤치마크 이름을 논문의 초점이라고
# 답함). 제목("References")으로 자르는 방식은 장마다 참고문헌이 있거나 부록
# 제목이 특이한 논문에서 본문을 날렸기 때문에, 청크 단위로 형식을 보고 판별한다.
# 본문의 "(Smith et al., 2020)" 같은 인용에는 걸리지 않는 형식만 센다.
_BIB_AUTHOR = re.compile(
    r"[A-Z][a-zA-Z'\-]+,\s(?:[A-Z]\.\s?){1,3}"          # Sun, F.
    r"|(?<![A-Za-z])(?:[A-Z]\.\s?){1,3}[A-Z][a-z]+,"    # J. Smith,
)
_BIB_YEAR = re.compile(r"\((?:19|20)\d{2}[a-z]?\)\.|,\s(?:19|20)\d{2}[a-z]?\.(?!\d)")
_BIB_INDEX = re.compile(r"(?:^|\s)\[\d{1,3}\]\s+[A-Z][a-z]+")
_BIB_VENUE = re.compile(r"\barXiv\b|\bProceedings\b|\bpp\.\s*\d|\bdoi\b", re.IGNORECASE)
BIB_THRESHOLD: int = 8   # 보정 결과: 참고문헌 22/23 검출, 본문·부록 137개 중 오탐 1


def _looks_like_bibliography(chunk: str) -> bool:
    score = (len(_BIB_AUTHOR.findall(chunk)) + len(_BIB_YEAR.findall(chunk))
             + len(_BIB_INDEX.findall(chunk)) + len(_BIB_VENUE.findall(chunk)))
    return score >= BIB_THRESHOLD


# 약어 보존 — LLM이 모르는 신조어 약어는 키워드 단계에서 엉뚱하게 바뀐다
# (관찰: "mcp"가 "Multi-Agent Systems"로 바뀌어 검색 전체가 빗나감).
# 질문의 약어가 키워드에서 사라졌으면 "약어 + AI"를 첫 구문으로 넣는다.
# "MCP"만으로는 생물학 등 다른 분야 약어와 섞이고, "MCP AI"로 검색하면
# 5편 모두 Model Context Protocol 논문이 나왔다.
# 한국어 질문에서는 소문자로 쓴 약어(mcp, llm)도 잡고, 영어 질문에서는
# 일반 단어와 구별하기 위해 대문자로 쓴 것만 잡는다.
_ACRONYM_KO = re.compile(r"(?<![A-Za-z])[A-Za-z][A-Za-z0-9]{1,3}(?![A-Za-z])")
_ACRONYM_EN = re.compile(r"(?<![A-Za-z])[A-Z][A-Z0-9]{1,5}(?![A-Za-z])")
_ACRONYM_SKIP = frozenset({"ai", "and", "or", "the", "of", "vs", "in", "on", "to"})


def _keep_acronyms(query: str, keywords: str) -> str:
    pattern = _ACRONYM_KO if LLMClient.detect_language(query) == "ko" else _ACRONYM_EN
    kw_lower = keywords.lower()
    missing = [a for a in dict.fromkeys(pattern.findall(query))
               if a.lower() not in _ACRONYM_SKIP
               and not re.search(rf"\b{re.escape(a.lower())}\b", kw_lower)]
    if not missing:
        return keywords
    phrases = [p.strip() for p in keywords.split("|") if p.strip()]
    added = [f"{a.upper()} AI" for a in missing]
    logger.info("키워드에서 사라진 약어 보존: %s", added)
    return " | ".join((added + phrases)[:3])


# 논문 선택 이유가 "직접 다루지 않는다"고 밝혔는지 — 선택 프롬프트의 지시 문구와
# 실제로 관찰된 변형("직접적인 정의나 설명은 없지만")을 함께 잡는다.
_WEAK_MATCH = re.compile(
    r"직접\s*다루지\s*(?:는\s*)?않|직접적인?\s*.{0,15}(?:없|않)|does not directly", re.IGNORECASE
)


# 정의 질문 판별 — "MCP가 뭐야?" 같은 질문은 개념 자체를 설명하는 개요 논문이
# 맞는데, 개념을 응용한 사례 논문이 뽑혀 정작 정의를 설명하지 못했다(관찰).
_DEFINITION_INTENT = re.compile(
    r"뭐야|뭔가요|뭐예요|뭐에요|뭐지|무엇인가|무엇이야|무엇이에요|무엇입니까"
    r"|란\s*(?:뭐|무엇|\?|$)|(?:정의|개념)\s*(?:이|은|는|가)?\s*(?:뭐|무엇|\?|$)"
    r"|^\s*what\s+(?:is|are)\b|\bdefin(?:e|ition)\b|\bexplain\s+what\b",
    re.IGNORECASE,
)


def _is_definition_intent(query: str) -> bool:
    """
    "무엇인가"는 "AI가 직업에 미치는 영향은 무엇인가"처럼 정의가 아닌 질문에도
    쓰이므로, 정의 질문이 대개 짧다는 점을 함께 본다.

    한국어는 한글 글자 수만 센다(12자 이하). 예전에는 전체 길이(16자)를 봐서
    "end to end automation이 뭐야?"처럼 영어 용어가 긴 정의 질문을 놓쳤다(관찰).
    용어가 길어도 정의 질문이고, 늘어나는 건 한국어 서술부일 때 정의가 아니다.
    """
    q = query.strip()
    if not _DEFINITION_INTENT.search(q):
        return False
    if LLMClient.detect_language(q) == "ko":
        return len(re.findall(r"[가-힣]", q)) <= 12
    return len(q.split()) <= 6


def _is_single_term(query: str) -> bool:
    """
    정의를 묻는 대상이 한 단어(MCP, RAG, 강화학습)인가 — survey 검색어를 붙일지 정한다.

    "end to end automation이 뭐야?"처럼 영어 단어 여럿으로 된 용어는 그 자체로 검색이
    충분히 구체적이다. 여기에 survey를 붙였더니 "Survey"라는 단어만 맞는 천문 관측·
    설문 연구 논문과 다른 분야 서베이가 후보를 채우고, 원래 검색어로는 2위였던 논문이
    후보에서 사라졌다(관찰). 한 단어 용어(특히 약어)일 때만 서베이가 도움이 됐다.
    """
    words = [w for w in re.findall(r"[A-Za-z]+", query) if w.lower() not in _DEFINITION_WORDS]
    return len(words) <= 1


# 영어 정의 질문의 질문 표현 — 용어 단어 수를 셀 때 뺀다 ("what is MCP" → MCP 한 단어)
_DEFINITION_WORDS = frozenset({
    "what", "is", "are", "a", "an", "the", "define", "definition", "of", "explain",
})


def _with_survey_phrase(keywords: str) -> str:
    """
    첫 구문에 survey를 붙인 검색어를 맨 앞에 넣는다.

    맨 앞이어야 하는 이유: 검색은 구문 순서대로 진행하고 후보가 다 차면 멈추는데,
    실측상 대부분 첫 구문 하나로 후보 10편이 채워진다. 뒤에 넣으면 검색되지 않는다.
    """
    phrases = [p.strip() for p in keywords.split("|") if p.strip()]
    if not phrases or "survey" in phrases[0].lower():
        return keywords
    return " | ".join([f"{phrases[0]} survey"] + phrases[:2])


# 새 검색 의도 — "최신 논문을 가져와줘"가 직전 논문의 후속 질문으로 분류되어
# 이미 읽은 논문으로 답한 사례가 있었다(관찰). "논문"이 들어 있어 LLM 분류기가
# 후속 질문으로 판단한다. 새 논문을 달라는 표현은 후속 질문 판정보다 먼저 본다.
_NEW_SEARCH_INTENT = re.compile(
    r"가져와|찾아\s*(?:줘|주|봐|와)|검색해|추천해"
    r"|(?:다른|새|새로운|최신|최근)\s*논문"
    r"|\b(?:find|search\s+for|fetch|recommend)\b.{0,20}\bpapers?\b"
    r"|\b(?:another|other|different|new|newer|latest|recent)\s+papers?\b",
    re.IGNORECASE,
)


# "이 논문에서 실험 결과 찾아줘"처럼 지금 논문을 가리키면서 "찾아줘"를 쓰는 경우는
# 후속 질문이다. 단, "이 논문 말고 다른 거"처럼 지금 논문을 밀어내는 표현이면 새 검색.
_CURRENT_PAPER_REF = ("이 논문", "이논문", "방금", "앞서", "this paper", "the paper")
_EXCLUDE_CURRENT = re.compile(r"말고|대신|외에|빼고|\binstead\b|\bother\s+than\b", re.IGNORECASE)


def _is_new_search_intent(query: str) -> bool:
    if not _NEW_SEARCH_INTENT.search(query):
        return False
    refers_current = any(k in query.lower() for k in _CURRENT_PAPER_REF)
    return not refers_current or bool(_EXCLUDE_CURRENT.search(query))


# 최신 의도 — S2의 날짜 정렬(sort=publicationDate)은 검색 API에서 무시되어
# "최신 논문"을 물어도 오래된 고인용 논문이 나왔다. 연도 필터는 실제로 동작하므로
# 최신 의도가 보이면 작년 이후 논문만 검색한다.
_RECENCY_INTENT = re.compile(
    r"최신|최근|요즘|근래|올해|요새"
    r"|\b(?:latest|recent|recently|newest|current\s+trends?|this\s+year)\b",
    re.IGNORECASE,
)


# "제일 최신 논문"처럼 가장 최근 것을 콕 집어 원하는 경우. 작년 이후로 거르고
# 관련도 순으로 고르면 5월 논문을 두고 10월 논문을 놓친다(관찰). 이때는 올해
# 논문부터 찾고, 선택 단계에서 날짜를 보여주며 관련 있는 것 중 가장 최근 것을 고르게 한다.
_NEWEST_INTENT = re.compile(
    r"(?:제일|가장|젤)\s*(?:최신|최근)|\b(?:newest|most\s+recent)\b|\blatest\s+papers?\b",
    re.IGNORECASE,
)


def _wants_newest(query: str) -> bool:
    return bool(_NEWEST_INTENT.search(query))


def _recency_year(query: str) -> Optional[str]:
    """최신 의도면 S2 연도 필터 문자열("2025-")을, 아니면 None을 돌려준다."""
    if not _RECENCY_INTENT.search(query):
        return None
    this_year = date.today().year
    return f"{this_year}-" if _wants_newest(query) else f"{this_year - 1}-"


def _relax_year(year: str) -> Optional[str]:
    """연도 필터 결과가 0건일 때 한 단계 넓힌다: 올해 → 작년 이후 → 제한 없음."""
    this_year = date.today().year
    return f"{this_year - 1}-" if year == f"{this_year}-" else None


# 최신 의도일 때 검색어에서 뺄 시기 표현. 연도 필터가 이미 "최신"을 맡는데 LLM이
# 같은 뜻을 단어로도 넣어서, 제목에 "Advancements", "Cutting-edge"가 들어간
# 논문(분야 무관한 화학·마케팅 논문 포함)이 후보를 채웠다(관찰).
# modern·novel·current는 넣지 않는다 — "modern Hopfield networks", "novel view
# synthesis"처럼 기술 이름의 일부인 경우가 있다. 단수형 development/innovation도
# "drug development"처럼 주제어일 수 있어 복수형만 뺀다.
_RECENCY_WORDS = re.compile(
    r"\b(?:latest|recent|newest|cutting[- ]edge|state[- ]of[- ]the[- ]art|emerging|contemporary"
    r"|advancements?|advances|breakthroughs?|developments|innovations)\b",
    re.IGNORECASE,
)


def _strip_recency_words(keywords: str) -> str:
    phrases = []
    for p in keywords.split("|"):
        cleaned = " ".join(_RECENCY_WORDS.sub(" ", p).split())
        cleaned = re.sub(r"^(?:in|of|and|on)\s+|\s+(?:in|of|and|on)$", "", cleaned, flags=re.IGNORECASE)
        if cleaned and cleaned.lower() not in (x.lower() for x in phrases):
            phrases.append(cleaned)
    if not phrases:
        return keywords    # 전부 시기 표현뿐이면 원래 검색어를 쓴다
    return " | ".join(phrases)


# 중복 불릿 제거 — 소형 모델이 같은 내용을 어미만 바꿔 두 번 쓰는 경우가 있다
# (관찰: "~를 분석합니다"와 "~를 분석하고 있습니다"가 연달아 나옴). 프롬프트로
# 금지해도 완전히 막히지 않아, 이미 로드된 e5 임베딩으로 거의 같은 불릿을 지운다.
#
# 임베딩만으로는 안 된다: 보정해 보니 "계획 능력을 평가" / "도구 사용 능력을 평가"가
# 0.988, "일자리를 줄입니다" / "늘립니다"가 0.988로, 실제 중복(0.99~0.999)과 겹쳤다.
# 그래서 짧은 쪽 불릿의 단어(한국어는 앞 2글자 어간)가 긴 쪽에 모두 들어 있어야
# 중복으로 본다. 핵심 단어 하나만 달라도 남기므로, 놓치는 중복은 있어도 서로 다른
# 내용을 지우는 일은 막는다. 보정: 어미만 다른 반복 5/5 검출, 서로 다른 내용 15쌍
# (단어 하나만 다른 쌍 포함) 오탐 0. 단어를 바꿔 쓴 의역 중복(0/6)은 잡지 못한다.
DEDUP_THRESHOLD: float = 0.97
_BULLET_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*\S)\s*$")
_WORD = re.compile(r"[가-힣]+|[A-Za-z]+|\d+(?:\.\d+)?")


def _stems(text: str) -> set:
    out = set()
    for w in _WORD.findall(text):
        if re.match(r"[가-힣]", w):
            out.add(w[:2])                       # 분석합니다 / 분석하고 → 분석
        else:
            out.add(w.lower().rstrip("s") or w)  # agents / agent → agent
    return out


def _same_words(a: str, b: str) -> bool:
    sa, sb = _stems(a), _stems(b)
    small, big = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    return small <= big


# 너무 짧아 무엇을 찾을지 정할 수 없는 질문 — "컴퓨터"를 물었더니 LLM이 검색어를
# "computer architecture"로 정했고, 제목 단어만 겹치는 이미지 인식 논문(Inception)이
# 뽑혔다(관찰). 한 단어짜리 질문이면 검색 전에 무엇이 궁금한지 되묻는다.
# 대문자 약어(QMIX, RLHF)는 그 자체로 구체적인 기법명이라 그대로 검색한다.
# "트랜스포머"처럼 구체적인 단어도 되묻게 되지만, 화면의 "그대로 검색" 버튼으로
# 한 번에 넘어갈 수 있게 해서 비용을 클릭 한 번으로 줄였다.
def _is_vague_query(query: str) -> bool:
    q = query.strip().strip("?!.~ ").strip()
    if not q or re.search(r"\s", q):
        return False
    if (_DEFINITION_INTENT.search(q) or _NEW_SEARCH_INTENT.search(q)
            or _RECENCY_INTENT.search(q) or _is_summary_intent(q)):
        return False
    if re.fullmatch(r"[가-힣]{1,6}", q):
        return True
    if re.fullmatch(r"[A-Za-z]+", q):
        return not re.fullmatch(r"[A-Z][A-Z0-9]{2,}", q)
    return False


# 수치 의미 검사 — 답변이 "기존 대비 21.2% 향상"이라고 썼는데 원문의 21.2%는
# "top-1 error 21.2%"처럼 지표 값 자체인 경우(관찰: Inception 논문, 같은 질문 4번 모두).
# 프롬프트 규칙과 숫자 옆 표시를 모두 시험했지만 2.4B 모델은 "gains over the state of
# the art: 21.2% top-1 ... error"라는 원문 구조에 끌려 4/4 틀렸다. 막을 수 없어서
# 답변 뒤에 원문과 대조해 경고를 띄운다.
_KO_CHANGE = r"향상|감소|개선|줄어|줄였|낮아|낮췄|하락|증가"
_EN_CHANGE = r"improvement|improved|reduction|reduced|decrease|lower|better|gain|drop|fewer|less"
_METRIC_WORD = re.compile(
    r"\b(?:error|errors|error rate|loss|perplexity|latency|WER|CER)\b", re.IGNORECASE
)


def _metric_misreads(answer: str, context: str) -> List[Tuple[str, str]]:
    """
    답변에서 변화량(향상/감소 X%)으로 쓴 수치 중, 원문에서는 변화량 표현 없이
    오류율·손실 같은 지표와 같은 문장에만 나오는 것을 (수치, 원문 문장)으로 돌려준다.
    """
    context = fix_decimal_colons(context)
    found = []
    for n in dict.fromkeys(re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)\s*%", answer)):
        num = rf"(?<![\d.]){re.escape(n)}\s*%"
        as_change = re.search(
            rf"대비[^.\n]{{0,30}}{num}|{num}\s*(?:의\s*)?(?:{_KO_CHANGE})"
            rf"|(?:{_KO_CHANGE})된\s*[^.\n]{{0,15}}{num}"
            rf"|\bby\s+{num}|{num}\s*(?:\w+\s+){{0,2}}(?:{_EN_CHANGE})\b",
            answer, re.IGNORECASE,
        )
        if not as_change:
            continue
        # 원문에도 변화량으로 나오면("by 21.2%", "21.2% relative reduction") 답변이 맞을 수 있다
        if re.search(rf"\bby\s+{num}|{num}\s*(?:\w+\s+){{0,2}}(?:{_EN_CHANGE}|relative)\b",
                     context, re.IGNORECASE):
            continue
        for sent in re.split(r"(?<=[.!?])\s+", context):
            if re.search(num, sent) and _METRIC_WORD.search(sent):
                found.append((n, " ".join(sent.split())[:220]))
                break
    return found


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

# 지금 논문을 분명히 가리키는 표현만 넣는다. "더 알려", "계속", "what does",
# "tell me more" 같은 말은 "강화학습에 대해 더 알려줘", "what does RLHF stand for"처럼
# 새 주제에도 쓰여서, 키워드로 곧장 후속 질문 처리하면 엉뚱한 논문으로 답했다.
# 이런 애매한 경우는 아래 LLM 분류기가 판단한다.
_FOLLOWUP_KW_KO: frozenset = frozenset([
    "방금", "이 논문", "이논문", "앞서", "이것", "이 내용", "이내용",
    "요약해", "요약하", "해석해", "번역해", "이 부분", "이게 뭔",
])

_FOLLOWUP_KW_EN: frozenset = frozenset([
    "this paper", "the paper", "summarize", "summarise",
    "translate", "explain this", "this section",
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
        self._last_keywords: Optional[str] = None   # 직전 refine_query 결과 (index_paper가 넘겨받음)
        self._paper_keywords: Optional[str] = None  # 현재 논문을 찾은 키워드 (BM25용)
        self.search_year: Optional[str] = None      # 최신 의도일 때 S2 연도 필터 ("2025-")
        self.wants_newest: bool = False             # "제일 최신" — 선택 시 날짜 우선

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
          2. New-search intent ("다른 논문 찾아줘") → always False.
          3. Keyword heuristic — deterministic, zero LLM cost.
          4. LLM classification — fallback for ambiguous queries.
        """
        if self._current_paper is None or self._store.size == 0:
            return False

        if _is_new_search_intent(user_query):
            logger.info("New-search intent detected: '%s'", user_query[:50])
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
        질문을 Semantic Scholar 검색어(영어 구문 3개, " | " 구분)로 바꾼다.

        LLM이 만든 검색어에 규칙 기반 보정을 차례로 적용한다: 최신 의도면 시기 표현
        제거, 사라진 약어 복원, 정의 질문이면 survey 검색어 추가. 결과는
        _last_keywords에 남겨 두었다가 index_paper가 BM25용 키워드로 넘겨받는다.
        """
        keywords = self._llm.refine_query(user_query)
        self.search_year = _recency_year(user_query)
        self.wants_newest = _wants_newest(user_query) and self.search_year is not None
        if self.search_year:
            keywords = _strip_recency_words(keywords)
        keywords = _keep_acronyms(user_query, keywords)
        if _is_definition_intent(user_query) and _is_single_term(user_query):
            keywords = _with_survey_phrase(keywords)
        self._last_keywords = keywords
        return keywords

    @staticmethod
    def is_weak_match(reasoning: str) -> bool:
        """
        논문 선택 이유가 "질문을 직접 다루지 않는다"고 스스로 밝혔는가.

        선택 프롬프트는 맞는 논문이 없으면 이유에 그 사실을 적도록 되어 있다.
        그 신호를 화면 경고로 이어주기 위한 판정이다.
        """
        return bool(_WEAK_MATCH.search(reasoning or ""))

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
        original_keywords = keywords
        year = getattr(self, "search_year", None)   # 최신 의도면 "2025-" 같은 연도 필터
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
                batch = self._s2.search(
                    clean_sub or sub, max_results=max_results, backfill=False, year=year
                )
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
            extra = self._s2.search(
                broad_query, max_results=max_results, backfill=False, year=year
            )
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
                results = self._s2.search(
                    shorter, max_results=max_results, backfill=False, year=year
                )
                if results:
                    merged = results
                    keywords = shorter
                    break

        if not merged and year:
            # 연도 필터 때문에 아무것도 없으면 한 단계 넓혀 다시 — 올해 → 작년 이후 →
            # 제한 없음. "최신"이 없는 것보다 조금 오래된 논문이라도 답하는 편이 낫다.
            # app.py는 search_year가 바뀐 것을 보고 화면 안내를 바꾼다.
            self.search_year = _relax_year(year)
            logger.info("연도 필터(%s) 결과 없음 — %s로 다시 검색", year, self.search_year or "제한 없이")
            return self.search_papers(original_keywords, max_results)

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
        prefer_recent = getattr(self, "wants_newest", False)
        if prefer_recent:
            # 날짜순으로 보여줘야 작은 모델이 날짜를 비교하기 쉽다. 원래 목록은 건드리지 않는다.
            papers = sorted(papers, key=lambda p: p.get("published") or "", reverse=True)
        return self._llm.select_best_paper(
            query, papers,
            prefer_overview=_is_definition_intent(query),
            prefer_recent=prefer_recent,
        )

    def candidates_for_display(self, papers: List[Dict]) -> List[Dict]:
        """
        답변 아래 후보 목록의 순서. 최신 의도면 최신순, 아니면 검색 결과(관련도) 순.
        """
        if getattr(self, "search_year", None):
            return sorted(papers, key=lambda p: p.get("published") or "", reverse=True)
        return list(papers)

    def switch_paper(self, paper: Dict) -> int:
        """
        후보 목록에서 사용자가 고른 논문으로 바꿔 인덱싱한다.

        PDF를 못 받으면 원래 읽고 있던 논문을 다시 올려서(임베딩 캐시라 빠름)
        후속 질문이 계속 원래 논문으로 이어지게 한다. 이 경우 0을 돌려준다.
        """
        prev = self._current_paper
        n = self.index_paper(paper)
        if n == 0 and prev is not None:
            logger.warning("선택한 논문의 PDF 확보 실패 — 원래 논문으로 복구: %s", prev["title"][:60])
            self.index_paper(prev)
        return n

    @staticmethod
    def is_vague_query(query: str) -> bool:
        """한 단어뿐이라 검색 전에 무엇이 궁금한지 되물어야 하는 질문인가."""
        return _is_vague_query(query)

    @staticmethod
    def clarify_options(query: str, lang: str) -> List[Tuple[str, str]]:
        """되물을 때 보여줄 선택지 (버튼 문구, 실제로 검색할 질문)."""
        w = query.strip().strip("?!.~ ").strip()
        if lang == "ko":
            return [
                (f"{w}의 개념 설명", f"{w}란 무엇인가?"),
                (f"{w} 최신 연구", f"{w} 관련 최신 연구 논문을 찾아줘"),
                ("그대로 검색", w),
            ]
        return [
            (f"What is {w}?", f"What is {w}?"),
            (f"Recent research on {w}", f"Find recent papers on {w}"),
            ("Search as is", w),
        ]

    @staticmethod
    def is_paper_request(query: str) -> bool:
        """"논문 찾아줘"처럼 논문 자체를 달라는 요청인가 (후보 목록을 함께 보여줄지)."""
        return bool(_NEW_SEARCH_INTENT.search(query))

    def index_paper(self, paper: Dict) -> int:
        """
        Download the PDF, extract text, chunk it, embed, and build FAISS.

        refine_query가 남긴 검색어를 이 논문의 BM25용 키워드로 넘겨받는다.

        Args:
            paper: Metadata dict from search_papers().

        Returns:
            Number of chunks indexed (0 indicates a failure).
        """
        self._store.reset()
        self._current_paper = paper
        # 이 논문을 찾은 키워드를 BM25 검색용으로 넘겨받는다. PDF 실패로 다음 후보를
        # 다시 인덱싱할 때는 _last_keywords가 이미 비어 있으므로 덮어쓰지 않는다.
        if self._last_keywords:
            self._paper_keywords = self._last_keywords
        self._last_keywords = None

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

        # 참고문헌 청크 제외 — 전부 걸러지는 비정상 상황이면 원본을 쓴다
        body = [c for c in chunks if not _looks_like_bibliography(c)]
        if body:
            if len(body) < len(chunks):
                logger.info("참고문헌으로 보이는 청크 %d개 제외", len(chunks) - len(body))
            chunks = body

        # 임베딩 캐시 — 같은 논문 재질문 시 재계산을 건너뛴다.
        # 파일명에 청크 내용의 해시를 넣는다. 예전에는 논문 ID만 썼는데, PDF 정제나
        # 참고문헌 필터를 고쳐서 청크 내용이 바뀌어도 개수가 같으면 옛 벡터를 그대로
        # 썼다(내용과 벡터가 어긋난 채로 검색됨).
        safe = paper["id"].replace("/", "_").replace(".", "_")
        digest = hashlib.sha1("\x00".join(chunks).encode("utf-8")).hexdigest()[:12]
        emb_path = os.path.join(".", "cache", "emb", f"{safe}_{digest}.npy")
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)

        if os.path.exists(emb_path):
            embeddings = np.load(emb_path)
            logger.info("Embedding cache hit: %s (%d chunks)", emb_path, len(chunks))
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
        query: str = "",
    ) -> Tuple[Optional[Dict], int, bool]:
        """
        선택된 논문을 인덱싱하되, PDF 확보 실패 시 다음 후보로 자동 전환.

        query가 주어지면 대체 후보를 질문과 가까운 순(제목+초록 임베딩 유사도)으로
        시도한다. 목록 순서대로 넘기면 질문과 덜 맞는 논문으로 바뀌는 경우가 있었다
        (관찰: 검색 결과 맨 앞 논문으로 바뀜). LLM에 다시 묻는 것보다 훨씬 싸다(0.1초 수준).

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
        rest = [p for p in papers if p.get("id") != selected.get("id")]
        ordered = [selected] + rest
        for i in range(min(max_attempts, len(ordered))):
            if i == 1 and query and len(rest) > 1:
                # 첫 시도가 실패했을 때만 계산한다 — 대부분은 첫 시도에서 끝난다.
                ordered = [selected] + self._rank_by_query(query, rest)
            paper = ordered[i]
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

    def _rank_by_query(self, query: str, papers: List[Dict]) -> List[Dict]:
        """후보를 질문과 가까운 순으로 정렬한다(제목+초록, e5 임베딩)."""
        texts = [f"{p.get('title', '')}. {(p.get('abstract') or '')[:500]}" for p in papers]
        sims = self._embedder.embed_documents(texts) @ self._embedder.embed_query(query)
        order = np.argsort(-sims)
        return [papers[int(i)] for i in order]

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

        self._remember_context(context_chunks, paper)
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

        return thinking, self.dedup_bullets(answer)

    def dedup_bullets(self, text: str, threshold: float = DEDUP_THRESHOLD) -> str:
        """
        어미만 다르고 내용이 같은 불릿을 하나로 합친다.

        두 불릿 중 더 긴 쪽(대개 정보가 더 많은 쪽)을 앞 불릿 자리에 남긴다.
        불릿이 아닌 줄(도입 문장 등)은 그대로 둔다. 임베딩은 불릿 문장만 계산하므로
        불릿 3개 기준 수십 ms 수준이다.
        """
        if not text:
            return text
        lines = text.split("\n")
        bullets = [(i, m.group(1)) for i, line in enumerate(lines)
                   if (m := _BULLET_LINE.match(line))]
        if len(bullets) < 2:
            return text
        vecs = self._embedder.embed_documents([b for _, b in bullets])
        kept: List[int] = []          # bullets 인덱스
        drop: set = set()             # lines 인덱스
        for j, (line_idx, body) in enumerate(bullets):
            twin = next((k for k in kept
                         if float(vecs[k] @ vecs[j]) >= threshold
                         and _same_words(bullets[k][1], body)), None)
            if twin is None:
                kept.append(j)
                continue
            drop.add(line_idx)
            if len(body) > len(bullets[twin][1]):
                lines[bullets[twin][0]] = lines[line_idx]
        if not drop:
            return text
        logger.info("중복 불릿 %d개 제거", len(drop))
        return "\n".join(line for i, line in enumerate(lines) if i not in drop)

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
        self._remember_context(context_chunks, paper)
        yield from self._llm.stream_rag_response(query, context_chunks, paper, lang)

    def _remember_context(self, context_chunks: List[Dict], paper: Dict) -> None:
        """답변에 쓴 근거(초록 + 문단)를 남겨 둔다 — 답변 뒤 수치 검사용."""
        self._last_context = "\n".join(
            [paper.get("abstract") or ""] + [c["chunk"] for c in context_chunks]
        )

    def metric_misreads(self, answer: str) -> List[Tuple[str, str]]:
        """
        직전 답변에서 원문의 지표 값(오류율 21.2% 등)을 변화량(21.2% 향상)으로 잘못 쓴
        수치와 그 원문 문장. 앱은 이것을 답변 아래 경고로 보여준다.
        """
        return _metric_misreads(answer, getattr(self, "_last_context", ""))

    # ------------------------------------------------------------------
    # Unified retrieval path (Parts 3 + 4)
    # ------------------------------------------------------------------

    def _retrieve(self, query: str, query_vec: np.ndarray) -> List[Dict]:
        """
        Hybrid retrieve → sentence-level compress.

        1. BM25 + dense fusion via RRF (각 순위 상위 CANDIDATE_K개를 합쳐 TOP_K개).
        2. Sentence-level compression using the e5 embedder (no LLM call).

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
            query_text=self._bm25_text(query),
            top_k=TOP_K,
            candidate_k=CANDIDATE_K,
        )
        return self._compress_context(query_vec, chunks)

    def _bm25_text(self, query: str) -> str:
        """
        BM25에 넘길 검색어 — 사용자 질문 + 이 논문을 찾을 때 쓴 영어 키워드.

        BM25는 단어가 그대로 일치해야 점수를 주는데, 질문은 한국어이고 논문 본문은
        영어라서 한국어 질문만 넘기면 일치하는 단어가 하나도 없다. 즉 한국어
        질문에서는 하이브리드 검색이 사실상 밀집 검색 하나로만 돌고 있었다.
        영어 키워드를 함께 넘기면 약어·고유명사 정확 매치가 한국어 질문에서도 동작한다.
        """
        kw = getattr(self, "_paper_keywords", None)
        if not kw:
            return query
        return f"{query} {kw.replace('|', ' ')}"

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
        # UNPAYWALL_EMAIL이 있으면 arXiv·openAccessPdf가 없는 논문마다 Unpaywall을
        # 조회한다(건당 최대 10초). 차례로 하면 느린 와이파이에서 수십 초가 걸릴 수
        # 있어 동시에 조회한다. 결과 순서는 그대로 유지된다.
        with ThreadPoolExecutor(max_workers=5) as pool:
            flags = list(pool.map(self._s2.is_pdf_likely_available, papers))
        available = [p for p, ok in zip(papers, flags) if ok]
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
        self._paper_keywords = None
        self.search_year = None
        self.wants_newest = False
