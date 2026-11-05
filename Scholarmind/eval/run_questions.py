"""
eval/run_questions.py
---------------------
질문 10개를 자동으로 돌려 결과를 파일로 남긴다. 실행 중 사람 개입은 없다.

healthcheck.py 와 목적이 다르다:
  healthcheck.py  — 시스템이 작동하는가 (자동 판정, 40초)
  run_questions.py — 실제 질문에서 어떤 답이 나오는가 (기계적 실행만 자동화,
                     답변 품질 판단은 사람이 리포트를 훑어보며 한다)

설계 (이전 버전의 실패에서 도출):
  - 429는 "검색 실패"가 아니라 "API 문제"로 따로 표시한다. 프로젝트 결함과
    외부 장애를 섞으면 결과가 무의미해진다.
  - 질문끼리 상태를 공유하지 않는다. 후속질문은 선행질문이 성공했을 때만
    의미가 있으므로, 선행이 실패하면 FAIL이 아니라 SKIP으로 남긴다.
  - 자동 판정은 객관적으로 확실한 것만 한다. 불용어 목록이나 단어겹침 비율
    같은 휴리스틱은 오탐이 많아 전부 뺐다.
  - 채점 빈칸을 만들지 않는다. 대신 판단에 필요한 정보를 압축해서 보여준다.

실행:
    python eval/run_questions.py            # 전체 10문항
    python eval/run_questions.py 1 3 5      # 특정 번호만
결과:
    eval/results/questions_<timestamp>.md
"""

import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from embedder import Embedder
from llm_client import LLMClient
from rag_pipeline import MIN_CANDIDATE_POOL, RAGPipeline
from semantic_scholar_tool import S2RateLimitError


@dataclass
class Question:
    qid: int
    category: str
    query: str
    # 이 질문은 지정한 번호의 질문이 성공한 직후에 이어서 물어야 의미가 있다
    follows: Optional[int] = None
    expect_followup: Optional[bool] = None


QUESTIONS: List[Question] = [
    Question(1, "한국어 복합어", "뉴로 심볼릭 ai"),
    Question(2, "영어 표준 용어", "multi-agent reinforcement learning"),
    Question(3, "모호한 구어체", "study and llm"),
    Question(4, "광범위한 질문", "artificial intelligence workflow"),
    Question(5, "개념+분야 교차", "환각과 의학"),
    Question(6, "후속 질문", "방금 논문 요약해줘", follows=5, expect_followup=True),
    Question(7, "새 주제 전환", "양자컴퓨팅 최신 동향", follows=6, expect_followup=False),
    Question(8, "구어체+약어", "llm이 알아서 진화를 할수있어?"),
    Question(9, "희귀 키워드", "rare disease diagnosis with few-shot learning"),
    Question(10, "최신 트렌드", "MLOps pipeline orchestration"),
]


@dataclass
class Run:
    q: Question
    status: str = "OK"          # OK | API | SKIP | ERROR
    note: str = ""
    lang: str = ""
    followup: Optional[bool] = None
    keywords: str = ""
    n_candidates: int = 0
    paper_title: str = ""
    citation: str = ""
    reasoning: str = ""
    n_chunks: int = 0
    substituted: bool = False
    answer: str = ""
    elapsed: float = 0.0
    checks: Dict[str, bool] = field(default_factory=dict)


def citation_label(paper: Dict) -> str:
    if paper.get("arxiv_id"):
        return f"arXiv:{paper['arxiv_id']}"
    if paper.get("doi"):
        return f"DOI:{paper['doi']}"
    return "Semantic Scholar"


def run_one(pipe: RAGPipeline, q: Question, prior_ok: bool) -> Run:
    r = Run(q=q)
    t0 = time.time()

    if q.follows is not None and not prior_ok:
        r.status = "SKIP"
        r.note = f"Q{q.follows}가 성공하지 않아 이 질문은 의미가 없음 (실패 아님)"
        return r

    try:
        if q.follows is None:
            pipe.reset()

        r.lang = pipe.detect_language(q.query)
        r.followup = pipe.is_followup(q.query)
        if q.expect_followup is not None:
            r.checks["후속질문 판정"] = r.followup == q.expect_followup

        if r.followup:
            paper = pipe.current_paper
            r.paper_title = paper["title"]
            r.citation = citation_label(paper)
            _, r.answer = pipe.generate_answer(q.query, paper, r.lang)
        else:
            r.keywords = pipe.refine_query(q.query)
            r.checks["키워드 형식"] = (
                " | " in r.keywords and len(r.keywords.split(" | ")) >= 2
            )

            papers, _ = pipe.search_papers(r.keywords, max_results=10)
            r.n_candidates = len(papers)
            r.checks["후보 확보"] = len(papers) > 0
            if not papers:
                r.status = "ERROR"
                r.note = "검색 결과 0건 (API는 정상 응답)"
                return r

            paper, r.reasoning = pipe.select_paper(q.query, papers)
            r.checks["선택 이유 생성"] = bool(r.reasoning.strip())

            indexed, r.n_chunks, r.substituted = pipe.index_with_fallback(
                papers, paper, query=q.query
            )
            r.checks["PDF 확보"] = r.n_chunks > 0
            if r.n_chunks == 0:
                r.paper_title = paper["title"]
                r.citation = citation_label(paper)
                r.status = "ERROR"
                r.note = "상위 후보 3편 모두 PDF 확보 실패"
                return r

            r.paper_title = indexed["title"]
            r.citation = citation_label(indexed)
            _, r.answer = pipe.generate_answer(q.query, indexed, r.lang)

        r.checks["답변 생성"] = bool(r.answer.strip())
        if r.answer.strip():
            numbered = any(
                re.match(r"^\s*\d+[\.\)]\s", l) for l in r.answer.splitlines()
            )
            r.checks["불릿 형식"] = not numbered
            r.checks["답변 언어 일치"] = LLMClient.detect_language(r.answer) == r.lang

    except S2RateLimitError:
        r.status = "API"
        r.note = "Semantic Scholar 요청 한도(429) — 프로젝트 결함 아님"
    except Exception as exc:
        r.status = "ERROR"
        r.note = f"{type(exc).__name__}: {exc}"
    finally:
        r.elapsed = time.time() - t0
    return r


def render(runs: List[Run]) -> str:
    L = [
        "# ScholarMind 질문 실행 결과",
        "",
        f"실행 시각: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 한눈에 보기",
        "",
        "| # | 질문 | 상태 | 자동체크 | 후보 | 선택된 논문 |",
        "|---|---|---|---|---|---|",
    ]
    for r in runs:
        ok = sum(1 for v in r.checks.values() if v)
        tot = len(r.checks)
        chk = f"{ok}/{tot}" if tot else "-"
        if tot and ok < tot:
            chk = f"**{chk}**"
        title = (r.paper_title[:40] + "...") if len(r.paper_title) > 40 else (r.paper_title or "-")
        L.append(
            f"| {r.q.qid} | {r.q.query} | {r.status} | {chk} | "
            f"{r.n_candidates or '-'} | {title} |"
        )

    failed = [r for r in runs if r.status == "ERROR"]
    api = [r for r in runs if r.status == "API"]
    bad_checks = [r for r in runs if any(not v for v in r.checks.values())]

    L += ["", "## 확인이 필요한 항목", ""]
    if not failed and not api and not bad_checks:
        L.append("자동 체크는 전부 통과. 아래 상세에서 답변 내용만 확인하면 된다.")
    if api:
        L.append(f"- **API 한도(429)**: Q{', Q'.join(str(r.q.qid) for r in api)} "
                 "— 잠시 후 재실행. 코드 문제가 아니다.")
    if failed:
        for r in failed:
            L.append(f"- **Q{r.q.qid} 실패**: {r.note}")
    for r in bad_checks:
        names = [k for k, v in r.checks.items() if not v]
        L.append(f"- **Q{r.q.qid} 자동체크 불통과**: {', '.join(names)}")

    L += [
        "",
        "## 눈으로 볼 것",
        "",
        "자동으로 판정할 수 없는 두 가지만 보면 된다:",
        "",
        "1. **선택된 논문이 질문과 관련 있는가** (엉뚱한 논문이 뽑히지 않았는가)",
        "2. **답변에 논문에 없는 내용이 있는가** (환각)",
        "",
        "---",
        "",
        "## 상세",
    ]

    for r in runs:
        L += ["", f"### Q{r.q.qid}. {r.q.query}", ""]
        L.append(f"`{r.q.category}` · {r.status} · {r.elapsed:.0f}초")
        if r.note:
            L.append(f"> {r.note}")
        if r.status == "SKIP":
            continue
        L.append("")
        if r.keywords:
            L.append(f"- 키워드: `{r.keywords}`")
        if r.followup is not None and r.q.expect_followup is not None:
            L.append(f"- 후속질문 판정: `{r.followup}` (기대 `{r.q.expect_followup}`)")
        if r.n_candidates:
            flag = " ⚠️ 후보 부족" if r.n_candidates < MIN_CANDIDATE_POOL else ""
            L.append(f"- 후보: {r.n_candidates}편{flag}")
        if r.paper_title:
            L.append(f"- 논문: **{r.paper_title}** (`{r.citation}`)")
        if r.substituted:
            L.append("- ⚠️ 원래 고른 논문의 PDF 실패로 다른 논문으로 대체됨")
        if r.reasoning:
            L.append(f"- 선택 이유: _{r.reasoning}_")
        if r.n_chunks:
            L.append(f"- 인덱싱: {r.n_chunks}청크")
        if r.checks:
            marks = " ".join(
                f"{'✅' if v else '❌'}{k}" for k, v in r.checks.items()
            )
            L.append(f"- 자동체크: {marks}")
        if r.answer:
            L += ["", "```", r.answer.strip(), "```"]

    return "\n".join(L)


def main() -> int:
    picks = {int(a) for a in sys.argv[1:] if a.isdigit()}
    todo = [q for q in QUESTIONS if not picks or q.qid in picks]

    logging.disable(logging.WARNING)   # 파이프라인 내부 로그는 리포트에 불필요

    llm = LLMClient()
    if not llm.is_available():
        print("Ollama가 응답하지 않는다. `ollama serve` 후 다시 실행.")
        return 1
    if not os.environ.get("S2_API_KEY"):
        print("경고: S2_API_KEY가 없다. 429로 대부분 실패할 수 있다.\n")

    pipe = RAGPipeline(embedder=Embedder(), llm=llm)

    runs: List[Run] = []
    ok_by_id: Dict[int, bool] = {}
    for q in todo:
        prior_ok = ok_by_id.get(q.follows, False) if q.follows else True
        print(f"[{q.qid}/{len(QUESTIONS)}] {q.query} ...", end="", flush=True)
        r = run_one(pipe, q, prior_ok)
        runs.append(r)
        ok_by_id[q.qid] = r.status == "OK"
        bad = [k for k, v in r.checks.items() if not v]
        print(f" {r.status} ({r.elapsed:.0f}s)" + (f" / 미통과: {', '.join(bad)}" if bad else ""))

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"questions_{datetime.now():%Y%m%d_%H%M%S}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render(runs))

    n_ok = sum(1 for r in runs if r.status == "OK")
    n_api = sum(1 for r in runs if r.status == "API")
    print(f"\n완료: {n_ok}/{len(runs)} 정상"
          + (f", {n_api}건 API 한도" if n_api else ""))
    print(f"리포트: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
