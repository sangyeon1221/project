"""
llm_client.py
-------------
Ollama-backed LLM client for Ragxiv.
LLM backend: exaone3.5:2.4b (LG AI Research, Korean/English bilingual).
Pull command: ollama pull exaone3.5:2.4b

EXAONE migration notes
----------------------
API endpoint
    All calls use /api/chat (role-message format) instead of /api/generate.
    Ollama automatically applies EXAONE's chat template:
    [|user|]{prompt}[|endofturn|]\n[|assistant|]
    The response is read from json["message"]["content"].

Stop sequences
    EXAONE's turn-end token is [|endofturn|].  All Qwen-era tokens
    (<|endoftext|>, <|im_end|>, <|end|>) have been replaced.

repeat_penalty
    LG officially recommends repeat_penalty ≤ 1.0 for EXAONE.  Any value
    above 1.0 can starve the Korean token pool and degrade output quality.
    All call sites now pass 1.0 (penalty disabled).

num_ctx
    Fixed at 4096 across every call.  Rough memory math for the KV cache
    (EXAONE 2.4B: 32 layers, 32 heads, head_dim 64, fp16):
        2 × 32 × 32 × 64 × 4096 × 2 bytes ≈ 512 MB
    Model weights at Q4 ≈ 1.5 GB → total ≈ 2 GB, safe on 8 GB laptop.
    Fits the RAG context comfortably: TOP_K=5 × 1000-char chunks ≈ 2 500
    Korean tokens + ~500 instruction tokens = ~3 000 tokens < 4096.

Chinese-leakage workarounds removed
    All "중국 알리바바 베이스셋 오염" warnings in prompts were qwen2.5-specific
    and are now dropped — EXAONE is Korean-native and does not exhibit that
    failure mode.

License
    EXAONE AI Model License Agreement 1.0 – NC.  Non-commercial academic
    use (graduation project) is explicitly permitted.  A separate commercial
    license from LG AI Research is required before any commercial deployment.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

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
# Prompt templates
# ---------------------------------------------------------------------------

# ── Zero-shot keyword extractor ───────────────────────────────────────────────
# "Fresh instance" framing signals that no prior context carries over.
# {avoid_block} is injected by refine_query() when the pipeline has tracked
# a prior keyword string.  Empty string when no prior keywords exist.

_REFINE_PROMPT = """\
컴퓨터 공학 학술 논문 검색을 위한 핵심 영단어 추출기입니다.
임무: 입력된 사용자의 질문(한국어 또는 영어)을 분석하여, arXiv 논문 검색에 가장 적합한 3개의 전문 영어 학술 키워드 구문만 도출하십시오.

규칙:
- 오직 단 한 줄에 " | " 기호로 구분된 3개의 영어 구문만 출력하십시오.
- 첫 번째 구문은 해당 주제의 단일 가장 정확한 표준 용어여야 합니다. 논문 제목/초록에 그대로 나타날 가능성이 가장 높은 구문을 선택하십시오. 주요 용어에 흔한 철자 변형이 있다면 "A OR B" 형식으로 두 가지를 모두 포함하십시오.
- 하이픈으로 연결된 복합어(예: neuro-symbolic, self-supervised, cross-modal)는 절대 분리하지 마십시오. 원형 그대로 유지하십시오.
- 사용자 질문의 핵심 명사를 임의로 바꾸지 마십시오. 영어 학술 표준 용어를 그대로 사용하십시오.
- 어떠한 한국어 설명이나 수식도 최종 출력에 절대 포함하지 마십시오.

[예시]
질문: llm이 알아서 진화를 할수있어?
Output: LLM self-improvement OR self-evolution | autonomous model adaptation | self-play reinforcement learning

질문: slm의 전망에 대해서 알려줘
Output: small language models OR SLM | parameter-efficient fine-tuning | on-device NLP optimization

질문: 뉴로 심볼릭 ai
Output: neuro-symbolic OR neurosymbolic | neural-symbolic reasoning | logic neural network integration

{avoid_block}
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

사용자 질문: {query}

후보 논문 목록:
{papers_block}

아래 두 줄 형식으로만 답하고, 다른 말은 절대 하지 마십시오.
번호: <선택한 논문의 숫자 하나만>
이유: <이 논문이 질문에 적합한 이유를 한국어 한 문장으로>"""

_SELECT_PROMPT_EN = """\
Instruction: Read the candidate papers below and pick the SINGLE most relevant paper for answering
the user's question.

Important rule: base your reason ONLY on what actually appears in the paper's title and abstract.
If none of the candidates directly address the core concept in the question, still pick the closest
one, but the reason MUST begin with "This paper does not directly cover [core concept], but" —
do NOT claim a paper covers a topic it does not.

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
- Write 2 to 4 bullet points, each a complete factual sentence. Synthesize; do not copy context sentences verbatim, and do not repeat a key phrase across sentences.
- If the paper DOES address the term or concept the user asked about, answer confidently with no hedge or disclaimer. ONLY when the paper genuinely does not cover that term, say so plainly ("This paper does not directly address X, but ...") and then explain the most relevant thing it actually covers. Never add a "does not directly address" disclaimer when the paper clearly is about the topic.
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
- 각 항목은 '- '(하이픈 공백)으로 시작하는 완결된 한 문장으로 작성하십시오. 2~4개 항목을 쓰고, 번호(1., 2.), 대괄호 마커, 이 규칙 자체는 출력하지 마십시오.
- <연구_문맥>에 등장하지 않는 사실을 지어내지 말고, 문맥에 실제로 있는 근거만 사용하십시오.
- 이 논문이 사용자가 물은 핵심 용어·개념을 실제로 다루는 경우에는, 어떠한 단서나 헤지도 붙이지 말고 단정적으로 답하십시오. 오직 논문이 그 용어·개념을 전혀 다루지 않을 때에만 "이 논문은 해당 용어를 직접 다루지는 않지만"이라고 먼저 밝힌 뒤, 논문이 실제로 다루는 가장 관련 있는 내용을 설명하십시오. 논문이 명백히 그 주제를 다루는데도 "직접 다루지 않는다"는 식의 불필요한 단서를 절대 붙이지 마십시오.
- 앞 문장에서 쓴 핵심 표현을 다음 문장에서 그대로 반복하지 마십시오.

<사용자_질문>{query}</사용자_질문>
<연구_문맥>
{context}
</연구_문맥>
[최종 답변]:"""


# ── Simplified fallback prompts ───────────────────────────────────────────────
# Used when the main RAG call returns an empty string.  Much shorter than the
# full prompt — fewer tokens means less chance the 1B model halts early.

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

    def refine_query(
        self, user_query: str, avoid_tokens: Optional[str] = None
    ) -> str:
        """
        Convert *user_query* into 3 CS-specific arXiv keyword phrases.

        Args:
            user_query:    Raw user question (any language).
            avoid_tokens:  Keyword string from the previous call; injected as
                           a short factual exclusion hint so the 1B model does
                           not recycle stale phrase components across turns.

        Returns:
            Space-joined keyword string suitable for arXiv search.
        """
        if avoid_tokens and avoid_tokens.strip():
            # Short, factual label — avoids negative phrasing that triggers
            # the model's safety refusal ("I can't help with that").
            avoid_block = f"[Excluded Historical Search Terms]: {avoid_tokens}\n"
        else:
            avoid_block = ""

        prompt = _REFINE_PROMPT.format(query=user_query, avoid_block=avoid_block)
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
        self, query: str, papers: List[Dict]
    ) -> Tuple[Dict, str]:
        """
        Autonomously choose the most relevant paper from *papers*.

        Detects query language and picks the matching prompt template so the
        model's output markers align with _parse_selection's regex, preventing
        the canned-fallback reasoning bug on English queries.

        Returns:
            (selected_paper_dict, reasoning_string).
            Falls back to index 0 if structured parsing fails.
        """
        lang = self.detect_language(query)
        template = _SELECT_PROMPT_KO if lang == "ko" else _SELECT_PROMPT_EN
        papers_block = self._format_papers_for_selection(papers)
        prompt = template.format(query=query, papers_block=papers_block)
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
        Generate the final answer using pure Korean-instruction prompts, apply
        Unicode cleaning (pdf_extractor.clean_llm_output), and return
        (thinking, answer).

        Hyperparameters:
            temperature=0.1  — deterministic stability for 2–4 bullet layout.
            repeat_penalty=1.0 — LG recommends ≤ 1.0 for EXAONE; any higher
                                  value degrades Korean output quality.
            top_p=0.9

        Returns:
            (thinking, answer) — thinking is "" with the Korean-prompt format.
        """
        context_text = "\n\n---\n\n".join(c["chunk"] for c in context_chunks)
        year = paper["published"][:4]
        authors = ", ".join(paper["authors"][:3])
        if len(paper["authors"]) > 3:
            authors += " et al."

        template = _RAG_PROMPT_KO if lang == "ko" else _RAG_PROMPT_EN
        prompt = template.format(
            query=query,
            title=paper["title"],
            year=year,
            context=context_text,
            authors=authors,
        )

        raw = self._generate_sync(
            prompt,
            temperature=0.1,
            repeat_penalty=1.0,
            top_p=0.9,
        )
        raw_clean = clean_llm_output(raw)
        thinking, answer = self._parse_thinking_answer(raw_clean)
        logger.debug(
            "RAG response — thinking=%d chars, answer=%d chars",
            len(thinking),
            len(answer),
        )
        return thinking, answer

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
        context_text = "\n\n".join(c["chunk"] for c in context_chunks[:3])
        year = paper["published"][:4]
        template = _FALLBACK_PROMPT_KO if lang == "ko" else _FALLBACK_PROMPT_EN
        prompt = template.format(
            query=query,
            title=paper["title"],
            year=year,
            context=context_text,
        )
        raw = self._generate_sync(prompt, temperature=0.1, repeat_penalty=1.0, top_p=1.0)
        raw_clean = clean_llm_output(raw)
        logger.info("Fallback summary — %d chars returned", len(raw_clean))
        return "", raw_clean.strip()

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
        num_ctx is fixed at 4096 — fits TOP_K=5 × 1000-char chunks with room
        to spare while keeping KV-cache memory safe on an 8 GB laptop.

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
            "num_ctx": 4096,
            "stop": _STOP_EOS,
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
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
                return " ".join(parts[:3])

        numbered = re.findall(r"^\d+[\.\)]\s*(.+)$", clean_raw, re.MULTILINE)
        if numbered:
            return " ".join(numbered[:3])

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
    def _parse_thinking_answer(raw: str) -> Tuple[str, str]:
        """
        Split a 'Thinking: … Answer: …' response into (thinking, answer).

        Falls back to ("", raw) when no Answer: marker is found — which
        is the expected path for the XML-prompt format where the entire
        generated text is the answer.
        """
        answer_markers = [
            "Answer:\n", "Answer:",
            "답변:\n", "답변:",
            "ANSWER:\n", "ANSWER:",
            "**Answer:**", "**답변:**",
        ]
        for marker in answer_markers:
            if marker in raw:
                pre, _, post = raw.partition(marker)
                thinking = re.sub(
                    r"^(Thinking:|생각:)\s*", "", pre.strip(), flags=re.IGNORECASE
                ).strip()
                return thinking, post.strip()
        return "", raw.strip()

    @staticmethod
    def _format_papers_for_selection(papers: List[Dict]) -> str:
        """Format candidate papers using strictly standardized Korean index tokens."""
        lines = []
        for i, p in enumerate(papers):
            abstract_preview = p["abstract"][:300].replace("\n", " ")
            # Purged English brackets [i] to lock down Qwen's multi-lingual decoding space.
            lines.append(
                f"제 {i} 번 논문\n"
                f"    [논문_제목]: {p['title']}\n"
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


