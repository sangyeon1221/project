"""
eval/benchmark_speed.py
-----------------------
단계별 실행 시간 측정. 네트워크 없이(캐시된 PDF로) 같은 조건을 반복 측정해서
"무엇이 시간을 얼마나 줄였는가"를 숫자로 남긴다.

측정 항목:
  1. 임베딩 모델 로드
  2. 인덱싱 — 임베딩 캐시 없음(첫 질문) / 있음(같은 논문 재질문)
  3. 검색 + 문장 압축
  4. LLM 단계 — GPU(기본) vs CPU(OLLAMA_NUM_GPU=0)
       논문 선택(후보 10편) / 답변 생성
  5. 답변 설정 비교(GPU) — 청크 5개+압축(현재) / 압축 끔 / 청크 3개

제외한 것: Semantic Scholar 검색과 PDF 다운로드. 네트워크 상태에 따라 수 배씩
달라져 같은 조건 반복 측정이 안 되기 때문이다. 이 둘은 앱 화면에서 따로 확인한다.

실행:
    python eval/benchmark_speed.py              # 질문 3개, CPU 비교 포함 (약 5분)
    python eval/benchmark_speed.py --skip-cpu   # GPU만 (약 2분)
    python eval/benchmark_speed.py --questions 1

결과는 화면에 표로 출력하고 eval/results/speed_benchmark.md에 저장한다.
CPU 측정 뒤에는 기본 설정으로 모델을 다시 올려 앱이 CPU 모드로 남지 않게 한다.
"""

import argparse
import logging
import os
import shutil
import statistics
import sys
import tempfile
import time
from datetime import datetime

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.WARNING)

QUESTIONS = [
    "이 논문의 주요 방법론은 무엇인가?",
    "실험 결과에서 가장 중요한 수치는 무엇인가?",
    "이 연구의 한계는 무엇인가?",
]
SELECT_QUESTIONS = [
    "대규모 언어 모델의 추론 능력을 높이는 방법",
    "검색 증강 생성의 최신 연구",
    "AI 에이전트의 도구 사용",
]


def timed(fn, *a, **k):
    t = time.perf_counter()
    out = fn(*a, **k)
    return time.perf_counter() - t, out


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def gpu_share(llm) -> str:
    """Ollama /api/ps로 모델이 실제로 GPU 메모리에 올라갔는지 확인한다."""
    try:
        models = requests.get(f"{llm.base_url}/api/ps", timeout=5).json().get("models", [])
    except Exception:
        return "?"
    for m in models:
        if m.get("name", "").split(":")[0] in llm.model:
            size, vram = m.get("size") or 0, m.get("size_vram") or 0
            return f"{100 * vram / size:.0f}% GPU" if size else "?"
    return "?"


def unload(llm) -> None:
    """
    모델을 메모리에서 내린다. num_gpu를 빼고 호출만 하면 Ollama는 이미 올라가 있는
    (CPU 모드) 모델을 그대로 재사용한다(실측: 복구 후에도 0% GPU). 설정을 바꿀 때는
    먼저 내려야 새 설정으로 다시 올라간다.
    """
    requests.post(f"{llm.base_url}/api/generate",
                  json={"model": llm.model, "keep_alive": 0}, timeout=30)
    time.sleep(1)


def cached_papers(n=10):
    """캐시된 PDF에서 선택 후보용 논문 dict를 만든다(제목=첫 줄, 초록=앞부분)."""
    from pdf_extractor import extract_text
    pdf_dir = os.path.join(ROOT, "pdfs")
    out = []
    for name in sorted(os.listdir(pdf_dir)):
        if not name.endswith(".pdf"):
            continue
        path = os.path.join(pdf_dir, name)
        text = extract_text(path) or ""
        if len(text) < 3000:
            continue
        first = next((l.strip() for l in text.splitlines() if len(l.strip()) > 20), name)
        out.append({
            "id": f"bench-{name[:-4]}", "title": first[:150], "abstract": text[:800],
            "authors": ["A. Author"], "published": "2025-01-01", "url": "", "path": path,
        })
        if len(out) == n:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=3)
    ap.add_argument("--skip-cpu", action="store_true")
    args = ap.parse_args()
    qs, sqs = QUESTIONS[:args.questions], SELECT_QUESTIONS[:args.questions]

    rows = []           # (구분, 항목, 중앙값 초, 비고)

    # 1. 임베딩 모델 로드
    from embedder import Embedder
    from llm_client import LLMClient
    import rag_pipeline as rp
    t_load, emb = timed(Embedder)
    rows.append(("시작", "임베딩 모델 로드", t_load, "local_files_only"))

    papers = cached_papers()
    if not papers:
        print("캐시된 PDF가 없습니다 — 앱에서 질문을 한 번 해서 pdfs/를 채운 뒤 실행하세요.")
        return 1
    target = papers[0]

    # 2. 인덱싱 — 임시 폴더에서 실행해 프로젝트의 cache/를 건드리지 않는다
    workdir = tempfile.mkdtemp(prefix="sm_bench_")
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        llm = LLMClient()
        pipe = rp.RAGPipeline(embedder=emb, llm=llm, save_dir=workdir)
        pipe._s2.download_pdf = lambda p: p["path"]
        pipe._last_keywords = "large language models | reasoning"
        t_miss, n = timed(pipe.index_paper, dict(target))
        pipe._last_keywords = "large language models | reasoning"
        t_hit, _ = timed(pipe.index_paper, dict(target))
        rows.append(("인덱싱", f"임베딩 캐시 없음 ({n}청크)", t_miss, "첫 질문"))
        rows.append(("인덱싱", "임베딩 캐시 있음", t_hit, "같은 논문 재질문"))

        # 3. 검색 + 압축, 그리고 답변 설정별 문맥 미리 만들기
        contexts = {"5청크+압축": [], "5청크 압축 끔": [], "3청크+압축": []}
        t_ret = []
        for q in qs:
            qv = emb.embed_query(q)
            t, ctx = timed(pipe._retrieve, q, qv)
            t_ret.append(t)
            pool = pipe._store.search(qv, query_text=pipe._bm25_text(q),
                                      top_k=rp.CANDIDATE_K, candidate_k=rp.CANDIDATE_K)
            contexts["5청크+압축"].append(ctx)
            contexts["5청크 압축 끔"].append(pool[:5])
            contexts["3청크+압축"].append(pipe._compress_context(qv, pool[:3]))
        rows.append(("검색", "하이브리드 검색 + 문장 압축", med(t_ret), f"질문 {len(qs)}개"))
    finally:
        os.chdir(cwd)
        shutil.rmtree(workdir, ignore_errors=True)

    def ctx_chars(name):
        return int(med([sum(len(c["chunk"]) for c in ctx) for ctx in contexts[name]]))

    # 4. LLM — GPU / CPU
    configs = [("GPU", None)] + ([] if args.skip_cpu else [("CPU", "0")])
    for label, num_gpu in configs:
        if num_gpu is None:
            os.environ.pop("OLLAMA_NUM_GPU", None)
        else:
            os.environ["OLLAMA_NUM_GPU"] = num_gpu
        unload(llm)
        t_warm, _ = timed(llm._generate_sync, "안녕")   # 모델 (재)로드 — 측정에서 제외
        share = gpu_share(llm)
        print(f"[{label}] 모델 로드 {t_warm:.1f}s, 실제 배치: {share}")

        t_sel = [timed(llm.select_best_paper, q, papers)[0] for q in sqs]
        rows.append((f"LLM {label}", f"논문 선택 (후보 {len(papers)}편)", med(t_sel), share))

        names = ["5청크+압축"] if label == "CPU" else list(contexts)
        for name in names:
            t_ans = [timed(llm.generate_rag_response, q, ctx, target, "ko")[0]
                     for q, ctx in zip(qs, contexts[name])]
            note = f"문맥 {ctx_chars(name):,}자" + (" (현재 설정)" if name == "5청크+압축" else "")
            rows.append((f"LLM {label}", f"답변 생성 — {name}", med(t_ans), note))

    if not args.skip_cpu:
        os.environ.pop("OLLAMA_NUM_GPU", None)
        unload(llm)                  # 앱이 CPU 모드로 남지 않도록 내린 뒤
        llm._generate_sync("안녕")   # 기본 설정으로 다시 올린다
        print(f"[복구] 기본 설정으로 재로드: {gpu_share(llm)}")

    # 출력
    lines = [
        f"# 속도 측정 ({datetime.now():%Y-%m-%d %H:%M})",
        "",
        f"질문 {len(qs)}개 기준 중앙값. 대상 논문: {target['title'][:80]}",
        "네트워크 단계(Semantic Scholar 검색, PDF 다운로드)는 제외.",
        "",
        "| 구분 | 항목 | 시간(초) | 비고 |",
        "|---|---|---:|---|",
    ] + [f"| {a} | {b} | {t:.2f} | {c} |" for a, b, t, c in rows]

    by = {(a, b): t for a, b, t, _ in rows}
    gpu_sel = by.get(("LLM GPU", f"논문 선택 (후보 {len(papers)}편)"))
    cpu_sel = by.get(("LLM CPU", f"논문 선택 (후보 {len(papers)}편)"))
    gpu_ans = by.get(("LLM GPU", "답변 생성 — 5청크+압축"))
    cpu_ans = by.get(("LLM CPU", "답변 생성 — 5청크+압축"))
    if gpu_sel and cpu_sel and gpu_ans and cpu_ans:
        lines += ["", f"GPU 가속: 논문 선택 {cpu_sel / gpu_sel:.1f}배, "
                      f"답변 생성 {cpu_ans / gpu_ans:.1f}배 빠름"]
    report = "\n".join(lines)
    print()
    print(report)

    out_dir = os.path.join(ROOT, "eval", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "speed_benchmark.md"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\n저장: eval/results/speed_benchmark.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
