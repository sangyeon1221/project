"""
eval/compare_selection.py
-------------------------
논문 선택 방식 비교 — 현재 방식 vs 절충안.

  현재  : LLM이 후보 전체(최대 10편)를 읽고 선택
  절충안: 임베딩 유사도로 상위 3편을 추린 뒤 LLM이 그중 선택

측정하는 것:
  1. 현재 방식의 자기 일치율 — 같은 후보로 두 번 돌렸을 때 같은 논문을 고르는가.
     현재 방식이 원래 들쭉날쭉하다면, 절충안이 다른 논문을 고른 것만으로는
     품질 저하라고 볼 수 없다.
  2. 현재 방식과 절충안의 일치율
  3. 선택 시간
  4. 블라인드 비교용 리포트 — 두 방식이 고른 논문을 A/B로 섞어 보여준다.
     어느 쪽이 어느 방식인지는 별도 정답표 파일에만 적는다.

두 방식이 같은 후보 목록에서 고르도록 검색 결과는 질문당 한 번만 받아
파일에 저장해두고 재사용한다(재실행 시 검색 생략).

실행:
    python eval/compare_selection.py
결과:
    eval/results/selection_blind.md   ← 이것만 보고 판단
    eval/results/selection_key.md     ← 판단 후에 열어볼 정답표
"""

import json
import logging
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from embedder import Embedder
from llm_client import LLMClient
from rag_pipeline import RAGPipeline
from semantic_scholar_tool import S2RateLimitError

QUERIES = [
    "뉴로 심볼릭 ai",
    "multi-agent reinforcement learning",
    "study and llm",
    "artificial intelligence workflow",
    "환각과 의학",
    "llm이 알아서 진화를 할수있어?",
    "rare disease diagnosis with few-shot learning",
    "MLOps pipeline orchestration",
    "AI 등장에 따른 미래 직업의 변화",
    "시계열 예측을 위한 트랜스포머 모델",
    "그래프 신경망을 이용한 추천 시스템",
    "양자컴퓨팅 최신 동향",
    "diffusion models for image generation",
    "연합학습의 개인정보 보호",
    "code generation with large language models",
]

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
POOLS = os.path.join(OUT, "selection_pools.json")


def get_pool(pipe, q, pools):
    """질문별 후보 목록. 저장된 게 있으면 재사용하고, 없으면 검색한다."""
    if q in pools:
        return pools[q]
    for attempt in range(4):
        try:
            pipe.reset()
            papers, _ = pipe.search_papers(pipe.refine_query(q), max_results=10)
            pools[q] = papers
            with open(POOLS, "w", encoding="utf-8") as fh:
                json.dump(pools, fh, ensure_ascii=False)
            return papers
        except S2RateLimitError:
            print("  (API 한도 — 30초 대기 후 재시도)", flush=True)
            time.sleep(30)
    return []


def paper_block(label, p):
    venue = p.get("venue") or "-"
    abstract = (p.get("abstract") or "(초록 없음)")[:400].replace("\n", " ")
    return [
        f"**{label}. {p['title']}**",
        f"{p['published'][:4]} · {venue} · 피인용 {p.get('citation_count', 0)}",
        "",
        f"> {abstract}...",
        "",
    ]


def main():
    logging.disable(logging.WARNING)
    os.makedirs(OUT, exist_ok=True)
    pools = json.load(open(POOLS, encoding="utf-8")) if os.path.exists(POOLS) else {}

    emb = Embedder()
    pipe = RAGPipeline(embedder=emb, llm=LLMClient())
    rng = random.Random(42)

    rows = []
    for n, q in enumerate(QUERIES, 1):
        print(f"[{n}/{len(QUERIES)}] {q}", flush=True)
        papers = get_pool(pipe, q, pools)
        if len(papers) < 4:
            print(f"  후보 {len(papers)}편 — 비교 의미 없어 건너뜀", flush=True)
            continue

        t = time.time(); cur1, _ = pipe.select_paper(q, papers); t_cur = time.time() - t
        cur2, _ = pipe.select_paper(q, papers)

        qv = emb.embed_query(q)
        docs = emb.embed_documents([f"{p['title']}. {p.get('abstract') or ''}" for p in papers])
        order = list(np.argsort(docs @ qv)[::-1])
        top3 = [papers[i] for i in order[:3]]
        t = time.time(); comp, _ = pipe.select_paper(q, top3); t_comp = time.time() - t

        cur_rank = next(r for r, i in enumerate(order) if papers[i]["id"] == cur1["id"]) + 1
        rows.append({
            "q": q, "n": len(papers), "cur": cur1, "cur2": cur2, "comp": comp,
            "self_same": cur1["id"] == cur2["id"], "same": cur1["id"] == comp["id"],
            "cur_rank": cur_rank, "t_cur": t_cur, "t_comp": t_comp,
            "comp_is_a": rng.random() < 0.5,
        })
        print(f"  자기일치 {rows[-1]['self_same']} | 절충안과 일치 {rows[-1]['same']} | "
              f"{t_cur:.0f}s → {t_comp:.0f}s", flush=True)

    k = len(rows)
    self_rate = sum(r["self_same"] for r in rows) / k
    same_rate = sum(r["same"] for r in rows) / k
    in_top3 = sum(r["cur_rank"] <= 3 for r in rows) / k
    t_cur = np.mean([r["t_cur"] for r in rows])
    t_comp = np.mean([r["t_comp"] for r in rows])

    # ---------- 블라인드 리포트 ----------
    B = [
        "# 논문 선택 블라인드 비교",
        "",
        "각 질문마다 두 방식이 고른 논문이 A/B로 섞여 있습니다. 어느 쪽이 어느 방식인지는 "
        "`selection_key.md`에만 있으니 **판단을 끝낸 뒤에 열어보세요.**",
        "",
        "판단 기준: **이 질문에 답하기 위해 PDF를 열어볼 논문으로 어느 쪽이 더 나은가.** "
        "각 질문 아래 `판단:` 줄에 `A`, `B`, `같음` 중 하나를 적어주세요.",
        "",
        "두 방식이 같은 논문을 고른 질문은 판단할 필요가 없어 따로 표시했습니다.",
        "",
        "---",
    ]
    K = ["# 정답표", "", "| # | 질문 | A | B |", "|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        B += ["", f"## Q{i}. {r['q']}", ""]
        if r["same"]:
            B += ["*두 방식이 같은 논문을 선택함 — 판단 불필요*", ""]
            B += paper_block("공통", r["cur"])
            K.append(f"| {i} | {r['q']} | 동일 | 동일 |")
            continue
        a, b = (r["comp"], r["cur"]) if r["comp_is_a"] else (r["cur"], r["comp"])
        B += paper_block("A", a) + paper_block("B", b) + ["판단: ", ""]
        K.append(f"| {i} | {r['q']} | {'절충안' if r['comp_is_a'] else '현재'} | "
                 f"{'현재' if r['comp_is_a'] else '절충안'} |")

    K += [
        "",
        "## 자동 측정 결과",
        "",
        f"- 비교한 질문: {k}개",
        f"- **현재 방식 자기 일치율** (같은 후보로 두 번 돌려 같은 논문): **{self_rate:.0%}**",
        f"- 현재 방식과 절충안 일치율: {same_rate:.0%}",
        f"- 현재 방식이 고른 논문이 임베딩 상위 3편 안에 든 비율: {in_top3:.0%}",
        f"- 평균 선택 시간: 현재 {t_cur:.1f}초 → 절충안 {t_comp:.1f}초",
        "",
        "| # | 질문 | 후보 | 현재 선택의 임베딩 순위 | 자기일치 | 절충안 일치 |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        K.append(f"| {i} | {r['q']} | {r['n']} | {r['cur_rank']} | "
                 f"{'O' if r['self_same'] else 'X'} | {'O' if r['same'] else 'X'} |")

    with open(os.path.join(OUT, "selection_blind.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(B))
    with open(os.path.join(OUT, "selection_key.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(K))

    print(f"\n자기 일치율 {self_rate:.0%} / 절충안 일치율 {same_rate:.0%} / "
          f"시간 {t_cur:.1f}s → {t_comp:.1f}s")
    print("판단용: eval/results/selection_blind.md")


if __name__ == "__main__":
    main()
