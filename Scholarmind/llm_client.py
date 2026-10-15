"""
llm_client.py
-------------
Ollama-backed LLM client for ScholarMind.
LLM backend: exaone3.5:2.4b (LG AI Research, Korean/English bilingual).
Pull command: ollama pull exaone3.5:2.4b

API endpoint
    All calls use /api/chat (role-message format). Ollama applies EXAONE's
    chat template: [|user|]{prompt}[|endofturn|]\n[|assistant|]
    Each call is a single message — no conversation history is carried over.

Stop sequences
    EXAONE's turn-end token is [|endofturn|].

repeat_penalty
    LG officially recommends repeat_penalty ≤ 1.0 for EXAONE.  Any value
    above 1.0 can starve the Korean token pool and degrade output quality.
    All call sites pass 1.0 (penalty disabled).

num_ctx
    3072 by default (OLLAMA_NUM_CTX overrides). The answer prompt is
    ~500 tokens of rules + abstract + 5 compressed passages (~700 chars),
    well under the limit; the selection prompt (10 abstracts × 300 chars) is
    the largest and also fits.

License
    EXAONE AI Model License Agreement 1.0 – NC.  Non-commercial academic
    use (graduation project) is explicitly permitted.  A separate commercial
    license from LG AI Research is required before any commercial deployment.
"""

import json
import logging
import os
import re
from typing import Dict, Iterator, List, Optional, Tuple

import requests

from pdf_extractor import clean_llm_output

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = "http://localhost:11434"
MODEL_NAME: str = "exaone3.5:2.4b"
TEMPERATURE: float = 0.05   # default for utility calls (refine / select / classify)
OLLAMA_TIMEOUT: int = 300   # seconds — conservative for 2.4B on CPU (~5 tok/s)

# ---------------------------------------------------------------------------
# Stop sequences — EXAONE turn-end token only
# ---------------------------------------------------------------------------
# [|endofturn|] is EXAONE's official turn-end marker (LG docs).
# The Qwen-era tokens (<|endoftext|>, <|im_end|>, <|end|>) are removed.

_STOP_EOS: List[str] = [
    "[|endofturn|]",
]

# ---------------------------------------------------------------------------
# 출력 형식 정규화
# ---------------------------------------------------------------------------
# EXAONE 2.4B는 한국어 응답에서 "'- '로 시작하라"는 프롬프트 규칙을 일관되게
# 무시하고 "1. 2. 3." 번호 목록을 출력한다(영어 응답은 규칙을 지킴 — 실측 확인).
# 프롬프트에 출력 예시를 추가해도 소형 모델에서는 100% 보장되지 않으므로,
# 줄 시작의 번호 매김만 하이픈 불릿으로 후처리 변환한다.
#
# 줄 맨 앞에만 적용한다 — 본문 중간의 "단계 1. ..." 같은 정당한 번호 표기나
# 소수점(예: "3.5배"), 연도 등은 건드리지 않는다.
_LEADING_NUMBER = re.compile(r"^\s*\d+\s*[\.\)]\s+", re.MULTILINE)


def _gpu_option() -> Dict:
    """
    OLLAMA_NUM_GPU가 설정돼 있을 때만 num_gpu(GPU에 올릴 레이어 수)를 넘긴다.
    0이면 CPU로만 돈다 — 내장 그래픽 드라이버 문제가 생겼을 때의 우회로이자,
    eval/benchmark_speed.py가 CPU와 GPU를 비교할 때 쓴다. 미설정이면 Ollama 기본값.
    """
    value = os.getenv("OLLAMA_NUM_GPU", "")
    return {"num_gpu": int(value)} if value.strip() else {}


# 프롬프트 끝의 "[최종 답변]:" / "[Final Answer]:"를 모델이 답변 첫머리에 그대로
# 따라 쓰는 경우가 있어 맨 앞에서만 지운다. 예전 파서는 답변 어디든 "답변:"이
# 나오면 그 앞을 "생각"으로 보고 버렸는데(이전 모델 시절의 Thinking/Answer 형식),
# 본문에 그 단어가 들어간 정상 답변까지 잘릴 수 있어 없앴다.
_ANSWER_ECHO = re.compile(
    r"^\s*\**\[?\s*(?:최종\s*답변|final\s+answer|답변|answer)\s*\]?\s*[:：]\**\s*", re.IGNORECASE
)


def strip_answer_echo(text: str) -> str:
    return _ANSWER_ECHO.sub("", text or "", count=1)


def normalize_bullets(text: str) -> str:
    """줄 머리의 '1. ' / '2) ' 형태를 '- '로 통일한다."""
    if not text:
        return text
    return _LEADING_NUMBER.sub("- ", text)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# ── Zero-shot keyword extractor ───────────────────────────────────────────────

_REFINE_PROMPT = """\
컴퓨터 공학 학술 논문 검색을 위한 핵심 영단어 추출기입니다.
임무: 입력된 사용자의 질문(한국어 또는 영어)을 분석하여, Semantic Scholar 학술 논문 검색에 가장 적합한 3개의 전문 영어 학술 키워드 구문만 도출하십시오.

규칙:
- 오직 단 한 줄에 " | " 기호로 구분된 3개의 영어 구문만 출력하십시오.
- 각 구문은 2~4 단어의 간결한 표준 학술 용어로만 작성하십시오. 불용어/일반어("study", "and", "research", "analysis", "about", "tell me" 등)는 절대 포함하지 마십시오.
- 사용자가 입력한 표현을 그대로 베끼지 마십시오. 모호한 약어나 구어체는 표준 학술 용어로 정규화하십시오 (예: "llm" -> "large language models").
- 첫 번째 구문은 해당 주제의 단일 가장 정확한 표준 용어여야 합니다. 논문 제목/초록에 그대로 나타날 가능성이 가장 높은 구문을 선택하십시오.
- 하이픈으로 연결된 복합어(예: neuro-symbolic, self-supervised, cross-modal)는 절대 분리하지 마십시오. 원형 그대로 유지하십시오.
- Semantic Scholar는 불리언 검색을 지원하지 않으므로 "OR" 같은 연산자를 절대 사용하지 마십시오.
- 질문이 광범위하거나 모호할 경우(구체적인 기법명이 없는 경우), 학계에서 실제로 통용되는 확립된 하위분야명이나 표준 용어를 사용하십시오. 존재 여부가 불확실한 기업식 합성어를 새로 만들어내지 마십시오.
- 어떠한 한국어 설명이나 수식도 최종 출력에 절대 포함하지 마십시오.

[예시]
질문: llm이 알아서 진화를 할수있어?
Output: LLM self-improvement | autonomous model adaptation | self-play reinforcement learning

질문: slm의 전망에 대해서 알려줘
Output: small language models | parameter-efficient fine-tuning | on-device NLP

질문: study and llm
Output: large language models | LLM evaluation | natural language processing

질문: artificial intelligence workflow
Output: MLOps | workflow automation | AI pipeline orchestration

현재 분석해야 할 사용자 질문: {query}
Output:"""


# ── Follow-up / new-topic classifier ─────────────────────────────────────────

_CLASSIFY_PROMPT = """\
TASK: Decide if the user's question is a follow-up about the CURRENT paper \
or a request for a completely NEW topic.

Current paper title: "{title}"
User question: "{query}"

FOLLOW-UP signs → output "followup":
  이 논문, 방금, 이것, 이 내용, 요약, 번역, 해석, 더 설명, this paper,
  summarize, translate, explain this, what does that mean, elaborate, continue.

NEW TOPIC signs → output "new":
  The user introduces a different research area or the question is clearly
  unrelated to the current paper.

Output ONLY one word — no punctuation, no explanation:
followup
OR
new

Answer:"""


# ── Paper selection — language-aware templates ────────────────────────────────

_SELECT_PROMPT_KO = """\
지시: 아래의 후보 논문 목록을 읽고, 사용자의 질문에 가장 관련성 높은 단 한 편의 논문을 고르십시오.

중요 규칙: 논문의 제목과 초록에 실제로 등장하는 내용만 이유로 쓰십시오.
후보 논문 중 어떤 것도 질문의 핵심 개념을 직접 다루지 않는다면, 가장 유사한 논문을 고르되 이유에 "이 논문은 [핵심 개념]을 직접 다루지는 않지만"이라고 반드시 명시하십시오. 관련 없는 논문을 관련 있는 것처럼 설명하지 마십시오.
사용자의 질문을 다른 말로 바꿔 반복하는 것을 이유로 쓰지 마십시오. 논문의 제목/초록에 나온 구체적인 방법론, 기법, 데이터셋, 또는 발견을 근거로 드십시오.
{overview_hint}
사용자 질문: {query}

후보 논문 목록:
{papers_block}

아래 두 줄 형식으로만 답하고, 다른 말은 절대 하지 마십시오.
번호: <선택한 논문의 숫자 하나만>
이유: <이 논문이 질문에 적합한 이유를 한국어 한 문장으로>"""

# 정의 질문("MCP가 뭐야?")에서 쓰는 안내. 관찰: 이런 질문에 개념을 응용한 사례
# 논문이 뽑혀 개념 자체를 설명하지 못했다. 개요 논문은 후보 목록에 [개요 논문]
# 표시를 붙여서, 작은 모델도 바로 알아보게 한다.
_OVERVIEW_HINT_KO = (
    "\n이 질문은 개념의 정의를 묻고 있습니다. 후보 중 [개요 논문] 표시가 붙은 논문이 "
    "질문의 개념을 다룬다면 그 논문을 우선 고르십시오. 개념을 특정 분야에 응용한 "
    "사례 논문보다 개념 자체를 설명하는 논문이 더 적합합니다.\n"
)
_OVERVIEW_HINT_EN = (
    "\nThis question asks what a concept is. If a candidate marked [Overview] covers the "
    "concept, prefer it. A paper that explains the concept itself fits better than one "
    "that applies it to a specific domain.\n"
)
# "제일 최신 논문" 질문용. 후보에 날짜를 보여주고 날짜순으로 정렬해서 넘긴다.
# 날짜만 보고 주제와 무관한 논문을 고르지 않도록 "관련 있는 것 중"을 명시한다.
_RECENT_HINT_KO = (
    "\n이 질문은 가장 최근에 나온 논문을 원합니다. 후보는 [논문_날짜] 기준 최신순으로 "
    "정렬되어 있습니다. 질문의 주제와 관련 있는 후보 중 날짜가 가장 늦은 논문을 고르십시오. "
    "주제와 무관한 논문은 날짜가 늦어도 고르지 마십시오.\n"
)
_RECENT_HINT_EN = (
    "\nThis question wants the most recent paper. Candidates are sorted newest first by [논문_날짜] (publication date). "
    "Among candidates relevant to the question's topic, pick the one with the latest date. "
    "Never pick an off-topic paper just because it is newer.\n"
)

# 서베이 논문 제목의 형태만 잡는다. 예전에는 "survey"라는 단어만 있으면 [개요 논문]을
# 붙여서 "Kilo Degree Survey"(천문 관측 프로젝트), "Survey Evidence on ..."(설문 연구)
# 같은 논문까지 개요 논문으로 표시됐다(관찰).
_OVERVIEW_TITLE = re.compile(
    r"\b(?:a|an|comprehensive|systematic|brief|short|literature|critical)\s+"
    r"(?:survey|review|overview)\b"
    r"|\b(?:survey|review|overview)\s+(?:of|on)\b"
    r"|:\s*(?:a\s+|an\s+)?(?:survey|review|overview|tutorial|primer)\b"
    r"|\btutorial\b|\bintroduction to\b|\bprimer\b|\btaxonomy\b",
    re.IGNORECASE,
)

_SELECT_PROMPT_EN = """\
Instruction: Read the candidate papers below and pick the SINGLE most relevant paper for answering
the user's question.

Important rule: base your reason ONLY on what actually appears in the paper's title and abstract.
If none of the candidates directly address the core concept in the question, still pick the closest
one, but the reason MUST begin with "This paper does not directly cover [core concept], but" —
do NOT claim a paper covers a topic it does not.
Do not restate the user's question back in different words as your justification. Name a specific
method, technique, dataset, or finding from the paper's title/abstract that makes it the best match.
{overview_hint}
User question: {query}

Candidate papers:
{papers_block}

Answer in EXACTLY these two lines and nothing else:
Number: <one paper number only>
Reason: <one sentence in English on why this paper fits the question>"""


# ── English RAG prompt — XML-tag structural isolation ─────────────────────────
# Constraints live inside <system_instruction> (policy, not text to echo).
# Generation trigger "[Final Answer]:" is outside all tags.

_RAG_PROMPT_EN = """\
<system_instruction>
Role: CS research assistant.
Task: Answer using ONLY the provided <retrieved_context>. Never invent facts not in the context.
Rules:
- Write at most 3 bullet points, each 1–2 sentences, concise. Synthesize; do not copy context sentences verbatim (numbers and metric names are the exception — keep them exactly as in the context), and do not repeat a key phrase across bullets.
- If the paper DOES address the term or concept the user asked about, answer confidently with no hedge or disclaimer. ONLY when the paper genuinely does not cover that term, say so plainly ("This paper does not directly address X, but ...") and then explain the most relevant thing it actually covers. Never add a "does not directly address" disclaimer when the paper clearly is about the topic.
- Every bullet must state one concrete fact from the context (a method, result, number, or definition). Never fill a bullet with an evaluation of the paper such as "it covers this only to a limited extent".
- The [Abstract] is the authors' own summary of what the paper does. Use it as the reference point when reading the passages, and do not mistake a case study or background explanation for the paper's contribution.
- Never guess what an acronym stands for. Expand it only if the expansion appears in <retrieved_context>; otherwise keep the acronym as written.
- If the question asks about a whole field (trends, state of the field, outlook, latest research), remember this answer rests on ONE paper. Open the first bullet by stating its scope ("From the perspective of this paper, ..."), and never present this paper's content as a fact about the entire field or as its only trend. Stating scope is not a hedge: after stating it, report the paper's content confidently.
- If the question asks what a concept is (e.g. "What is X?"), the first bullet must define the concept in one sentence based on <retrieved_context>. If the context has no definition and the paper only applies the concept, explain what this paper takes the concept to mean and how it implements it. If the scope rule above also applies, combine both in one sentence ("This paper defines X as ...").
- Report numbers with their original meaning. A value of a lower-is-better metric (error rate, loss, latency), e.g. "21.2% top-1 error", is the metric itself, not an improvement. Never rewrite it as "improved by X%"; say "improved" only when the context explicitly states what it is compared against and by how much.
- Never echo anything inside this <system_instruction> block.
</system_instruction>
<user_query>{query}</user_query>
<retrieved_context source="{title}" year="{year}">
{context}
</retrieved_context>
[Final Answer]:"""


# ── Korean RAG prompt — full Korean instruction, XML-tag structural isolation ─

_RAG_PROMPT_KO = """\
당신은 공학 논문을 분석하는 전문 연구원입니다. 오직 아래 <연구_문맥>에 실제로 담긴 내용만을 근거로 사용자의 질문에 한국어로 답하십시오.

작성 규칙:
- 각 항목은 '- '(하이픈 공백)으로 시작하십시오. 최대 3개 항목, 각 1–2문장, 간결하게 작성하십시오. 번호(1., 2.), 대괄호 마커, 이 규칙 자체는 출력하지 마십시오.
- <연구_문맥>에 등장하지 않는 사실을 지어내지 말고, 문맥에 실제로 있는 근거만 사용하십시오.
- 이 논문이 사용자가 물은 핵심 용어·개념을 실제로 다루는 경우에는, 어떠한 단서나 헤지도 붙이지 말고 단정적으로 답하십시오. 오직 논문이 그 용어·개념을 전혀 다루지 않을 때에만 "이 논문은 해당 용어를 직접 다루지는 않지만"이라고 먼저 밝힌 뒤, 논문이 실제로 다루는 가장 관련 있는 내용을 설명하십시오. 논문이 명백히 그 주제를 다루는데도 "직접 다루지 않는다"는 식의 불필요한 단서를 절대 붙이지 마십시오.
- 앞 문장에서 쓴 핵심 표현을 다음 문장에서 그대로 반복하지 마십시오.
- 모든 항목은 <연구_문맥>에 실제로 있는 구체적인 사실(방법, 결과, 수치, 정의) 하나를 담아야 합니다. "구체적인 내용은 제한적으로 다루고 있습니다"처럼 논문을 평가하는 문장으로 항목을 채우지 마십시오.
- [논문 초록]은 이 논문이 무엇을 하는지에 대한 저자의 요약입니다. 본문 문단을 해석할 때 기준으로 삼고, 본문에 나오는 사례나 배경 설명을 논문의 기능이나 기여로 착각하지 마십시오.
- 약어의 뜻을 추측해서 풀어 쓰지 마십시오. <연구_문맥>에 풀이가 나와 있을 때만 풀어 쓰고, 없으면 약어를 그대로 쓰십시오.
- 질문이 분야 전체(트렌드, 동향, 전망, 최신 연구 등)를 묻는 경우, 이 답변의 근거는 논문 한 편뿐입니다. 첫 항목을 "이 논문은 ~ 관점에서"처럼 논문의 범위를 밝히며 시작하고, 이 논문의 내용을 분야 전체의 사실이나 유일한 트렌드인 것처럼 일반화하지 마십시오. 범위를 밝히는 것은 단서(헤지)가 아니므로, 범위를 밝힌 뒤에는 논문의 내용을 단정적으로 쓰십시오.
- 질문이 개념의 정의를 묻는 경우(예: "X가 뭐야?"), 첫 항목에서 <연구_문맥>에 근거해 그 개념이 무엇인지 한 문장으로 정의하십시오. 문맥에 정의가 없고 논문이 그 개념을 응용만 한다면, 이 논문이 그 개념을 무엇으로 보고 어떻게 구현하는지로 설명하십시오. 바로 위의 범위 규칙(분야 전체를 묻는 경우)도 해당되면 "이 논문은 X를 ~로 정의합니다"처럼 한 문장에 함께 담으십시오.
- 수치는 <연구_문맥>에 적힌 의미 그대로 옮기십시오. 오류율·손실·지연 시간처럼 낮을수록 좋은 지표의 값(예: "top-1 오류율 21.2%")은 그 지표의 값 자체이지 개선 폭이 아닙니다. "기존 대비 X% 향상"처럼 비교로 바꾸어 쓰지 말고, 문맥에 비교 대상과 차이가 명시된 경우에만 "향상", "개선"이라고 쓰십시오.

출력 형식 예시 (이 형식을 정확히 따르십시오):
- 이 논문은 A 기법을 제안하여 B 문제를 해결합니다.
- C 데이터셋에서 D 지표 E를 기록했습니다.

<사용자_질문>{query}</사용자_질문>
<연구_문맥>
{context}
</연구_문맥>
[최종 답변]:"""


# ── Simplified fallback prompts ───────────────────────────────────────────────
# Used when the main RAG call returns an empty string.  Much shorter than the
# full prompt — fewer rules means less chance the small model returns nothing.
# 규칙을 일부러 뺐지만, 중복 제거·수치 검사 같은 후처리는 이 답변에도 적용된다.

_FALLBACK_PROMPT_EN = """\
You are a research assistant. Answer the question below using ONLY the \
provided excerpt. Write 3-5 complete, factual sentences.

Paper: "{title}" ({year})
Excerpt: {context}
Question: {query}

Answer:"""

_FALLBACK_PROMPT_KO = """\
당신은 연구 보조 AI입니다. 아래 발췌문만을 근거로 질문에 답하십시오. \
완전한 한국어 문장으로 3~5개 작성하십시오.

논문: "{title}" ({year})
발췌: {context}
질문: {query}

답변:"""


# 답변 문맥 구성 — 검색된 문단 앞에 논문 초록을 붙인다.
# 관찰: 사례 연구의 배경 문장("FMRP 단백질은 뇌 발달에 중요")을 플랫폼의 기능처럼
# 옮긴 답변이 나왔다. 모델이 문단 하나하나가 논문에서 어떤 위치인지 모르고 읽었기
# 때문이다. 초록은 저자가 직접 쓴 "이 논문이 무엇을 하는가"라서 기준점이 된다.
def _build_context(context_chunks: List[Dict], paper: Dict, lang: str) -> str:
    body = "\n\n---\n\n".join(c["chunk"] for c in context_chunks)
    abstract = (paper.get("abstract") or "").strip()
    if not abstract:
        return fix_decimal_colons(body)
    label = "[논문 초록]" if lang == "ko" else "[Abstract]"
    return fix_decimal_colons(f"{label}\n{abstract}\n\n---\n\n{body}")


# PDF·초록 추출에서 소수점이 콜론으로 깨지는 경우가 있다(관찰: Inception 초록의
# "21:2% top-1"). 뒤에 %가 붙은 숫자만 고친다 — "3:1" 같은 비율 표기는 건드리지 않는다.
_DECIMAL_COLON = re.compile(r"(?<=\d):(?=\d+\s*%)")


def fix_decimal_colons(text: str) -> str:
    return _DECIMAL_COLON.sub(".", text)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class LLMClient:
    """Ollama LLM client with XML-tag isolated prompts and language-aware output."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = MODEL_NAME,
        temperature: float = TEMPERATURE,
        timeout: int = OLLAMA_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Connection check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Return the names of locally available Ollama models."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Language detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_language(text: str) -> str:
        """Return 'ko' if text contains Hangul, 'en' otherwise."""
        if re.search(r"[가-힣ᄀ-ᇿ㄰-㆏]", text):
            return "ko"
        return "en"

    # ------------------------------------------------------------------
    # Step 1: Query refinement — zero-shot with token-avoid injection
    # ------------------------------------------------------------------

    def refine_query(self, user_query: str) -> str:
        """
        Convert *user_query* into 3 English search phrases joined by " | ".

        예전에는 직전 키워드를 "제외할 검색어"로 함께 넘겼다. 그런데 Ollama 호출은
        매번 대화 기록 없이 새로 시작하므로 모델이 이전 키워드를 "재활용"할 길이
        없었고, 이 제외 목록은 검색이 실패한 직후에만 붙어서 사용자가 같은 핵심어로
        다시 물을 때 그 핵심어를 피하게 만들었다. 그래서 없앴다.
        """
        prompt = _REFINE_PROMPT.format(query=user_query)
        raw = self._generate_sync(
            prompt,
            temperature=0.0,    # frozen — eliminates random drift in keyword output
            repeat_penalty=1.0, # no penalty — technical multi-word terms must flow freely
            top_p=1.0,
        )
        raw_clean = clean_llm_output(raw)
        keywords = self._parse_keywords(raw_clean, user_query)
        logger.info("Refined query: '%s' → '%s'", user_query[:50], keywords)
        return keywords

    # ------------------------------------------------------------------
    # Step 2a: Paper selection
    # ------------------------------------------------------------------

    def select_best_paper(
        self, query: str, papers: List[Dict], prefer_overview: bool = False,
        prefer_recent: bool = False,
    ) -> Tuple[Dict, str]:
        """
        Autonomously choose the most relevant paper from *papers*.

        Detects query language and picks the matching prompt template so the
        model's output markers align with _parse_selection's regex, preventing
        the canned-fallback reasoning bug on English queries.

        prefer_overview: 정의 질문일 때 True — 개요 논문에 표시를 붙이고
                         우선 고르도록 안내한다.

        Returns:
            (selected_paper_dict, reasoning_string).
            Falls back to index 0 if structured parsing fails.
        """
        lang = self.detect_language(query)
        template = _SELECT_PROMPT_KO if lang == "ko" else _SELECT_PROMPT_EN
        label, hint = "", ""
        # 두 안내가 동시에 켜지면 "개요 논문 우선"과 "가장 최근 논문 우선"이 서로
        # 다른 논문을 가리킬 수 있다. 사용자가 "최신"을 명시했으므로 그쪽을 따른다.
        if prefer_recent:
            prefer_overview = False
        if prefer_overview:
            label = "[개요 논문]" if lang == "ko" else "[Overview]"
            hint = _OVERVIEW_HINT_KO if lang == "ko" else _OVERVIEW_HINT_EN
        if prefer_recent:
            hint += _RECENT_HINT_KO if lang == "ko" else _RECENT_HINT_EN
        papers_block = self._format_papers_for_selection(
            papers, overview_label=label, show_date=prefer_recent
        )
        prompt = template.format(query=query, papers_block=papers_block, overview_hint=hint)
        # repeat_penalty=1.0: LG recommends not exceeding 1.0 for EXAONE.
        raw = self._generate_sync(
            prompt,
            temperature=0.3,
            repeat_penalty=1.0,
            top_p=0.9,
        )
        idx, reasoning = self._parse_selection(raw, len(papers))
        if not reasoning:
            reasoning = (
                "질문과 가장 관련성이 높은 논문입니다."
                if lang == "ko"
                else "Most relevant to the query."
            )
        selected = papers[idx]
        logger.info(
            "Selected paper [%d]: %s | %s",
            idx,
            selected["title"][:60],
            reasoning,
        )
        return selected, reasoning

    # ------------------------------------------------------------------
    # Follow-up / new-topic classification
    # ------------------------------------------------------------------

    def classify_query(self, user_query: str, paper_title: str) -> str:
        """
        Ask the LLM whether *user_query* is a follow-up about *paper_title*.

        Returns 'followup' or 'new'.
        """
        prompt = _CLASSIFY_PROMPT.format(query=user_query, title=paper_title)
        raw = self._generate_sync(prompt).lower().strip()
        if "followup" in raw or "follow-up" in raw or "follow up" in raw:
            logger.info("LLM classified query as follow-up")
            return "followup"
        logger.info("LLM classified query as new topic")
        return "new"

    # ------------------------------------------------------------------
    # Step 2b: RAG answer generation
    # ------------------------------------------------------------------

    def generate_rag_response(
        self,
        query: str,
        context_chunks: List[Dict],
        paper: Dict,
        lang: str,
    ) -> Tuple[str, str]:
        """
        Generate the final answer using the language-appropriate prompt
        (_RAG_PROMPT_KO for lang == "ko", _RAG_PROMPT_EN otherwise) and
        clean it (noise characters, numbered lists, echoed answer label).

        Hyperparameters:
            temperature=0.1  — deterministic stability for 2–4 bullet layout.
            repeat_penalty=1.0 — LG recommends ≤ 1.0 for EXAONE; any higher
                                  value degrades Korean output quality.
            top_p=0.9

        Returns:
            ("", answer) — 첫 값은 예전 Thinking/Answer 형식의 흔적으로 항상 빈 문자열.
        """
        prompt = self._rag_prompt(query, context_chunks, paper, lang)
        raw = self._generate_sync(
            prompt,
            temperature=0.1,
            repeat_penalty=1.0,
            top_p=0.9,
        )
        answer = strip_answer_echo(normalize_bullets(clean_llm_output(raw))).strip()
        return "", answer

    @staticmethod
    def _rag_prompt(query: str, context_chunks: List[Dict], paper: Dict, lang: str) -> str:
        """일반·스트리밍 답변이 같은 프롬프트를 쓰도록 한 곳에서 만든다."""
        template = _RAG_PROMPT_KO if lang == "ko" else _RAG_PROMPT_EN
        return template.format(
            query=query,
            title=paper["title"],
            year=paper["published"][:4],
            context=_build_context(context_chunks, paper, lang),
        )

    def stream_rag_response(
        self,
        query: str,
        context_chunks: List[Dict],
        paper: Dict,
        lang: str,
    ) -> Iterator[str]:
        """
        Streaming counterpart to generate_rag_response. Builds the identical
        prompt and yields answer text as it's generated. clean_llm_output is
        NOT applied per piece (it can corrupt multi-byte boundaries mid-stream);
        the caller cleans the assembled text once after streaming completes,
        matching generate_rag_response's existing post-processing.
        """
        prompt = self._rag_prompt(query, context_chunks, paper, lang)
        yield from self._generate_stream(prompt, temperature=0.1, repeat_penalty=1.0, top_p=0.9)

    # ------------------------------------------------------------------
    # Simplified fallback when main RAG call returns empty
    # ------------------------------------------------------------------

    def generate_fallback_summary(
        self,
        query: str,
        context_chunks: List[Dict],
        paper: Dict,
        lang: str,
    ) -> Tuple[str, str]:
        """
        Retry with a stripped-down prompt when generate_rag_response
        returns an empty answer.

        Uses only the first 3 chunks and conservative parameters so
        nothing interferes with generation completing.

        Returns:
            ("", answer) — thinking always empty on the fallback path.
        """
        context_text = _build_context(context_chunks[:3], paper, lang)
        year = paper["published"][:4]
        template = _FALLBACK_PROMPT_KO if lang == "ko" else _FALLBACK_PROMPT_EN
        prompt = template.format(
            query=query,
            title=paper["title"],
            year=year,
            context=context_text,
        )
        raw = self._generate_sync(prompt, temperature=0.1, repeat_penalty=1.0, top_p=1.0)
        answer = strip_answer_echo(normalize_bullets(clean_llm_output(raw))).strip()
        logger.info("Fallback summary — %d chars returned", len(answer))
        return "", answer

    # ------------------------------------------------------------------
    # Internal Ollama call
    # ------------------------------------------------------------------

    def _generate_sync(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        repeat_penalty: float = 1.0,
        top_p: float = 1.0,
    ) -> str:
        """
        Non-streaming Ollama /api/chat call; returns the full response string.

        Uses /api/chat (role-message format) so Ollama automatically applies
        EXAONE's chat template: [|user|]{prompt}[|endofturn|]\n[|assistant|]
        num_ctx defaults to 3072 (override via OLLAMA_NUM_CTX) — fits
        TOP_K=5 compressed chunks with room to spare while keeping KV-cache
        memory safe on an 8 GB laptop.

        Args:
            prompt:         The complete prompt string (sent as "user" role).
            temperature:    Per-call override; defaults to self.temperature.
            repeat_penalty: Token repetition penalty (keep at 1.0 for EXAONE).
            top_p:          Nucleus sampling threshold.
        """
        options: Dict = {
            "temperature": temperature if temperature is not None else self.temperature,
            "repeat_penalty": repeat_penalty,
            "top_p": top_p,
            # num_ctx: env override available; 3072 fits TOP_K=5 compressed chunks
            # (~750 tok context) + ~500 tok instructions with room to spare.
            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "3072")),
            # num_thread: 0 = Ollama default. Set to physical core count for a
            # free speedup (e.g. OLLAMA_NUM_THREAD=8).
            "num_thread": int(os.getenv("OLLAMA_NUM_THREAD", "0")),
            "stop": _STOP_EOS,
            **_gpu_option(),
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "keep_alive": "30m",   # keep model in VRAM/RAM between turns
            "options": options,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "").strip()
        except requests.exceptions.ReadTimeout:
            logger.error("Ollama timeout after %d s", self.timeout)
            return ""
        except Exception as exc:
            logger.error("Ollama error: %s", exc)
            return ""

    def _generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        repeat_penalty: float = 1.0,
        top_p: float = 1.0,
    ) -> Iterator[str]:
        """
        Streaming counterpart to _generate_sync — yields content pieces as Ollama
        generates them instead of returning the full string at once. Same options
        dict as _generate_sync (num_ctx/num_thread env overrides, keep_alive) so
        behavior is identical except for the stream flag. On any error this stops
        yielding silently; the caller falls back to generate_answer (non-streaming).
        """
        options: Dict = {
            "temperature": temperature if temperature is not None else self.temperature,
            "repeat_penalty": repeat_penalty,
            "top_p": top_p,
            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "3072")),
            "num_thread": int(os.getenv("OLLAMA_NUM_THREAD", "0")),
            "stop": _STOP_EOS,
            **_gpu_option(),
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "keep_alive": "30m",
            "options": options,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout, stream=True,
            )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                piece = data.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if data.get("done"):
                    break
        except requests.exceptions.ReadTimeout:
            logger.error("Ollama stream timeout after %d s", self.timeout)
            return
        except Exception as exc:
            logger.error("Ollama stream error: %s", exc)
            return

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_keywords(raw: str, fallback: str) -> str:
        """
        Extract 3 CS keyword phrases from the model output safely without returning None.

        Priority order:
          1. Pipe-separated phrases on a single line.
          2. Numbered-list items.
          3. First valid non-marker line.
          4. Original query (last resort).
        """
        # Strip leading "Output:" / "출력:" artifacts before any token splitting
        clean_raw = raw.replace("Output:", "").replace("출력:", "").strip()

        if "|" in clean_raw:
            parts = [p.strip() for p in clean_raw.split("|") if p.strip()]
            parts = [
                p for p in parts
                if not p.lower().startswith(
                    ("bad", "good", "output", "cs ", "question", "출력")
                )
            ]
            if parts:
                return " | ".join(parts[:3])

        numbered = re.findall(r"^\d+[\.\)]\s*(.+)$", clean_raw, re.MULTILINE)
        if numbered:
            return " | ".join(numbered[:3])

        skip_prefixes = (
            "bad", "good", "output", "cs ", "question", "you are",
            "task", "rules", "now", "few", "example", "출력",
        )
        for line in clean_raw.splitlines():
            line = line.strip().lstrip("0123456789.-) ")
            if line and not line.lower().startswith(skip_prefixes):
                return line

        return fallback

    @staticmethod
    def _format_papers_for_selection(
        papers: List[Dict], overview_label: str = "", show_date: bool = False
    ) -> str:
        """
        Format candidate papers using strictly standardized Korean index tokens.

        overview_label이 주어지면 제목에 survey/review/overview 등이 들어간 논문 앞에
        그 표시를 붙인다(정의 질문용). show_date면 게재일을 넣는다("제일 최신" 질문용) —
        평소에는 넣지 않아 선택이 날짜에 끌리지 않게 한다.
        """
        lines = []
        for i, p in enumerate(papers):
            abstract_preview = p["abstract"][:300].replace("\n", " ")
            mark = f"{overview_label} " if overview_label and _OVERVIEW_TITLE.search(p["title"]) else ""
            date_line = f"    [논문_날짜]: {p.get('published', '')[:10]}\n" if show_date else ""
            lines.append(
                f"제 {i} 번 논문\n"
                f"    [논문_제목]: {mark}{p['title']}\n"
                f"{date_line}"
                f"    [논문_저자]: {', '.join(p['authors'][:3])}\n"
                f"    [논문_초록]: {abstract_preview}..."
            )
        return "\n\n".join(lines)

    @staticmethod
    def _parse_selection(raw: str, n_papers: int) -> Tuple[int, str]:
        """
        Parse selection output tolerantly — works with both loose markers
        ("번호: 0", "이유: ...") and the old bracketed form ("[선택_번호]: 0").

        Returns (idx, reasoning) where reasoning is "" when nothing matched —
        select_best_paper supplies the language-appropriate default string.
        """
        idx = 0
        reasoning = ""

        # Number: loose ("번호: 0") and bracketed ("[선택_번호]: 0") / EN equivalents
        num_match = re.search(
            r"(?:선택\s*)?(?:번호|number)\s*\]?\s*[:：]\s*(\d+)",
            raw,
            re.IGNORECASE,
        )
        if num_match:
            cand = int(num_match.group(1))
            if 0 <= cand < n_papers:
                idx = cand
        else:
            nums = re.findall(r"\d+", raw)
            if nums:
                cand = int(nums[0])
                if 0 <= cand < n_papers:
                    idx = cand

        # Reason: loose ("이유: ...") and bracketed ("[선택_이유]: ...") / EN equivalents
        reason_match = re.search(
            r"(?:선택\s*)?(?:이유|reason(?:ing)?)\s*\]?\s*[:：]\s*(.+)",
            raw,
            re.IGNORECASE,
        )
        if reason_match:
            raw_reason = reason_match.group(1).strip()
            # Strip "N번 논문" prefix EXAONE occasionally prepends to the reason text
            reasoning = re.sub(r"^\d+\s*번\s*논문[\s.:]*", "", raw_reason).strip()
        else:
            for line in raw.splitlines():
                s = line.strip()
                if (s
                        and not re.search(r"(?:번호|number|이유|reason)", s, re.IGNORECASE)
                        and not re.fullmatch(r"[\d\s.:]+", s)):
                    reasoning = s
                    break

        return idx, reasoning


