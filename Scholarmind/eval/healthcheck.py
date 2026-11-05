"""
eval/healthcheck.py
-------------------
ScholarMind 시스템 상태 점검. 사람의 입력 없이 한 번 실행으로 판정한다.

설계 원칙 (이전 체크리스트의 실패에서 도출):
  1. 외부 서비스(Semantic Scholar) 상태는 "환경 정보"로만 보고하고 프로젝트
     판정에 넣지 않는다. 이전 버전은 S2의 429를 프로젝트 결함으로 오보했다.
  2. 체크끼리 상태를 공유하지 않는다. 각 체크는 자기 데이터를 직접 만든다.
  3. 체크 안에서 예외가 나면 그 체크는 FAIL이다. 절대 삼켜서 통과시키지 않는다.
  4. "예외 없이 실행됨"이 아니라 실제 출력값을 검증한다
     (길이 > 0, 차원 일치, 기대 문자열 포함 등).
  5. LLM 답변의 "품질"은 판정하지 않는다 — 비결정적이라 안정된 pass/fail이
     불가능하다. 서비스가 실제로 응답하는지만 확인한다.

실행:
    python eval/healthcheck.py

종료 코드: 실패한 체크가 있으면 1, 전부 통과면 0.
"""

import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
from typing import Callable, List, Tuple

import numpy as np
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 결과 타입: (통과 여부, 근거 문자열)
Result = Tuple[bool, str]

_embedder = None


def embedder():
    """임베딩 모델은 로드가 무거워 한 번만 만들어 재사용한다."""
    global _embedder
    if _embedder is None:
        from embedder import Embedder
        _embedder = Embedder()
    return _embedder


def _sample_store():
    """검색 계열 체크용 인덱스를 매번 새로 만든다 (체크 간 상태 공유 금지)."""
    from vector_store import VectorStore
    chunks = [
        "Transformer models use self-attention to process token sequences.",
        "The rare identifier XZQFOO7 appears only in this sentence.",
        "Neural networks are trained with gradient descent optimization.",
        "Retrieval augmented generation grounds answers in source documents.",
        "Quantum computing uses qubits for superposition based calculation.",
    ]
    vecs = embedder().embed_documents(chunks)
    meta = [{"chunk_idx": i} for i in range(len(chunks))]
    store = VectorStore()
    store.add_chunks(chunks, vecs, meta)
    return store, chunks


# ---------------------------------------------------------------------------
# 의존성
# ---------------------------------------------------------------------------

def check_dependencies() -> Result:
    """코드가 실제로 import 하는 서드파티 패키지가 설치되어 있는가."""
    required = [
        "streamlit", "sentence_transformers", "faiss", "fitz",
        "requests", "numpy", "rank_bm25", "feedparser",
    ]
    missing = []
    for mod in required:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        return False, f"미설치: {', '.join(missing)}"
    return True, f"{len(required)}개 모두 import 가능"


def check_requirements_file() -> Result:
    """requirements.txt가 실제 import 목록과 일치하는가."""
    pkg_to_req = {
        "rank_bm25": "rank-bm25",
        "feedparser": "feedparser",
        "faiss": "faiss-cpu",
        "fitz": "PyMuPDF",
        "sentence_transformers": "sentence-transformers",
    }
    req_path = os.path.join(ROOT, "requirements.txt")
    if not os.path.exists(req_path):
        return False, "requirements.txt 없음"
    with open(req_path, encoding="utf-8") as fh:
        listed = fh.read().lower()

    problems = []
    for mod, req_name in pkg_to_req.items():
        try:
            __import__(mod)
        except ImportError:
            continue
        if req_name.lower() not in listed:
            problems.append(f"누락:{req_name}")

    # 더 이상 import 되지 않는데 남아있는 항목
    for stale in ("arxiv",):
        if re.search(rf"^{stale}\b", listed, re.MULTILINE):
            try:
                __import__(stale)
                used = any(
                    re.search(rf"^\s*(import {stale}|from {stale})", open(
                        os.path.join(ROOT, f), encoding="utf-8").read(), re.MULTILINE)
                    for f in os.listdir(ROOT) if f.endswith(".py")
                )
                if not used:
                    problems.append(f"미사용:{stale}")
            except ImportError:
                problems.append(f"미설치인데 명시됨:{stale}")

    if problems:
        return False, "; ".join(problems)
    return True, "requirements.txt가 실제 의존성과 일치"


# ---------------------------------------------------------------------------
# Ollama + EXAONE
# ---------------------------------------------------------------------------

def check_ollama_service() -> Result:
    """프로세스 존재가 아니라 실제 HTTP 응답으로 확인한다."""
    from llm_client import OLLAMA_BASE_URL
    resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}"
    models = resp.json().get("models")
    if not isinstance(models, list):
        return False, f"응답에 models 리스트 없음: {str(resp.json())[:100]}"
    return True, f"HTTP 200, 모델 {len(models)}개 등록됨"


def check_exaone_model() -> Result:
    """설정된 모델에 실제 프롬프트를 보내고 응답 내용을 검증한다."""
    from llm_client import LLMClient, MODEL_NAME, OLLAMA_BASE_URL

    tags = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10).json()
    names = [m.get("name", "") for m in tags.get("models", [])]
    if not any(MODEL_NAME in n for n in names):
        return False, f"'{MODEL_NAME}' 미설치. 보유: {names}"

    client = LLMClient()
    # 정답이 명확한 프롬프트 — 응답이 실제 추론 결과인지 확인 가능
    out = client._generate_sync("What is 2+2? Reply with the number only.", temperature=0.0)
    if not out or not out.strip():
        return False, "빈 응답"
    if "4" not in out:
        return False, f"'4'를 포함하지 않는 응답: {out[:80]!r}"
    return True, f"'{MODEL_NAME}' 응답 정상: {out.strip()[:40]!r}"


# ---------------------------------------------------------------------------
# 임베딩
# ---------------------------------------------------------------------------

def check_embedder() -> Result:
    """벡터가 실제로 나오는지 — 차원, 정규화, 유한값까지 검증."""
    from embedder import EMBEDDING_DIM
    emb = embedder()
    docs = emb.embed_documents(["first document", "second document"])
    qv = emb.embed_query("a query")

    if docs.shape != (2, EMBEDDING_DIM):
        return False, f"문서 벡터 shape {docs.shape}, 기대 (2, {EMBEDDING_DIM})"
    if qv.shape != (EMBEDDING_DIM,):
        return False, f"쿼리 벡터 shape {qv.shape}, 기대 ({EMBEDDING_DIM},)"
    if not np.isfinite(docs).all() or not np.isfinite(qv).all():
        return False, "NaN/Inf 포함"
    norms = np.linalg.norm(docs, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        return False, f"L2 정규화 안 됨: norms={norms}"
    if np.allclose(docs[0], docs[1], atol=1e-6):
        return False, "서로 다른 문서가 동일 벡터 — 인코딩 이상"
    return True, f"shape {docs.shape}, L2 norm≈1.0, 문서별 벡터 상이"


def check_e5_prefixes() -> Result:
    """
    e5는 query:/passage: 접두사가 없으면 성능이 조용히 떨어진다.
    상수값과 실제 교차언어 정렬 동작을 함께 확인한다.
    """
    import embedder as emb_mod
    if emb_mod._QUERY_PREFIX != "query: ":
        return False, f"_QUERY_PREFIX={emb_mod._QUERY_PREFIX!r}"
    if emb_mod._PASSAGE_PREFIX != "passage: ":
        return False, f"_PASSAGE_PREFIX={emb_mod._PASSAGE_PREFIX!r}"

    emb = embedder()
    qv = emb.embed_query("환각 현상이란 무엇인가")
    passages = emb.embed_documents([
        "Hallucination is when a language model generates false information.",
        "Baking bread requires flour, water, yeast, and salt.",
    ])
    sims = passages @ qv
    if sims[0] <= sims[1]:
        return False, f"교차언어 정렬 실패: 관련={sims[0]:.3f} <= 무관={sims[1]:.3f}"
    return True, f"접두사 정상, 한국어질의-영어본문 정렬 {sims[0]:.3f} > 무관 {sims[1]:.3f}"


# ---------------------------------------------------------------------------
# 검색 (FAISS / BM25 / RRF)
# ---------------------------------------------------------------------------

def check_faiss_search() -> Result:
    """인덱스를 만들고 실제 검색 결과의 개수·구조·정확도를 확인한다."""
    store, chunks = _sample_store()
    if store.size != len(chunks):
        return False, f"인덱스 크기 {store.size}, 기대 {len(chunks)}"

    qv = embedder().embed_query("How do transformers use attention?")
    hits = store.search(qv, query_text="", top_k=3, candidate_k=5)

    if len(hits) == 0:
        return False, "검색 결과 0건"
    for h in hits:
        if not {"chunk", "metadata", "score"} <= set(h):
            return False, f"결과 구조 이상: {list(h)}"
        if not h["chunk"].strip():
            return False, "빈 청크 반환"
        if not np.isfinite(h["score"]):
            return False, f"score가 유한하지 않음: {h['score']}"
    top = hits[0]["chunk"]
    if "attention" not in top.lower():
        return False, f"의미 검색 부정확 — 1위: {top[:60]!r}"
    return True, f"{len(hits)}건 반환, 1위에 'attention' 포함, score 유한"


def check_bm25_index() -> Result:
    """add_chunks가 BM25 인덱스를 실제로 구축했는지, 어휘 검색이 되는지."""
    store, _ = _sample_store()
    if store._bm25 is None:
        return False, "add_chunks 후에도 _bm25 가 None"

    from vector_store import _tokenize
    scores = store._bm25.get_scores(_tokenize("XZQFOO7"))
    if len(scores) != store.size:
        return False, f"BM25 점수 개수 {len(scores)}, 기대 {store.size}"
    best = int(np.argmax(scores))
    if "XZQFOO7" not in store._chunks[best]:
        return False, f"희귀어 매칭 실패 — 최고점 청크: {store._chunks[best][:50]!r}"
    if scores[best] <= 0:
        return False, f"매칭됐는데 점수 {scores[best]}"

    store.reset()
    if store._bm25 is not None:
        return False, "reset 후에도 _bm25가 남아있음"
    return True, f"BM25 구축·희귀어 매칭 정상(score={scores[best]:.2f}), reset 시 해제"


def check_rrf_fusion() -> Result:
    """
    RRF 산식이 두 순위를 실제로 합산하는지 직접 검증한다.

    "희귀 토큰이 검색되는가" 같은 창발적 동작으로는 검증이 안 된다 —
    dense 임베딩도 희귀 토큰을 어느 정도 잡아서, 어휘 순위를 빼도
    결과가 같아 보일 수 있기 때문이다(실제로 그렇게 통과해버렸다).
    그래서 dense 순위와 BM25 순위를 각각 직접 구해 기대 점수를 계산하고,
    search()가 낸 점수와 대조한다. query_text를 무시하면 점수가 어긋난다.
    """
    from vector_store import _tokenize
    store, chunks = _sample_store()
    query = "XZQFOO7 attention"
    qv = embedder().embed_query(query)
    cand = store.size
    K = 60

    # dense 순위: query_text 없이 호출하면 밀집 검색만 수행된다
    dense_hits = store.search(qv, query_text="", top_k=cand, candidate_k=cand)
    dense_order = [chunks.index(h["chunk"]) for h in dense_hits]

    # 어휘 순위: BM25 점수에서 직접 계산
    bm25_scores = store._bm25.get_scores(_tokenize(query))
    sparse_order = list(np.argsort(bm25_scores)[::-1][:cand])

    expected = {}
    for rank, idx in enumerate(dense_order):
        expected[idx] = expected.get(idx, 0.0) + 1.0 / (K + rank + 1)
    for rank, idx in enumerate(sparse_order):
        expected[int(idx)] = expected.get(int(idx), 0.0) + 1.0 / (K + rank + 1)

    hybrid = store.search(qv, query_text=query, top_k=cand, candidate_k=cand)
    if not hybrid:
        return False, "하이브리드 검색 결과 0건"

    for h in hybrid:
        idx = chunks.index(h["chunk"])
        want = expected.get(idx)
        if want is None:
            return False, f"기대 순위에 없는 청크 반환: idx={idx}"
        if abs(h["score"] - want) > 1e-9:
            return False, (
                f"RRF 점수 불일치 idx={idx}: 실제={h['score']:.6f} 기대={want:.6f} "
                "(어휘 순위가 합산되지 않았을 가능성)"
            )

    scores = [h["score"] for h in hybrid]
    if scores != sorted(scores, reverse=True):
        return False, f"RRF 점수 내림차순 아님: {scores}"

    dense_top = dense_order[0]
    fused_top = chunks.index(hybrid[0]["chunk"])
    note = "융합이 순위를 바꿈" if dense_top != fused_top else "두 순위 1위가 동일"
    return True, f"RRF 산식 일치(k={K}, {len(hybrid)}건), 내림차순 정상, {note}"


# ---------------------------------------------------------------------------
# 문서 처리
# ---------------------------------------------------------------------------

def check_pdf_extraction() -> Result:
    """캐시된 실제 PDF에서 텍스트가 추출되는지."""
    from pdf_extractor import extract_text
    pdf_dir = os.path.join(ROOT, "pdfs")
    if not os.path.isdir(pdf_dir):
        return False, "pdfs/ 디렉터리 없음"
    pdfs = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    if not pdfs:
        return False, "캐시된 PDF 없음 — 추출 검증 불가"

    path = os.path.join(pdf_dir, sorted(pdfs)[0])
    text = extract_text(path)
    if not text or len(text) < 500:
        return False, f"추출 텍스트가 너무 짧음: {len(text or '')}자"
    letters = sum(c.isalpha() for c in text[:2000])
    if letters < 500:
        return False, f"문자 비율 이상 — 앞 2000자 중 알파벳 {letters}자"
    return True, f"{os.path.basename(path)} → {len(text):,}자 추출"


def check_text_cleaning() -> Result:
    """제어문자는 제거하고 한글/영문은 보존하는지."""
    from pdf_extractor import clean_llm_output
    out = clean_llm_output("한글 test\x00\x07 문장\x1f 유지")
    if "\x00" in out or "\x07" in out or "\x1f" in out:
        return False, f"제어문자 잔존: {out!r}"
    if "한글" not in out or "test" not in out or "유지" not in out:
        return False, f"정상 텍스트 손실: {out!r}"
    return True, f"제어문자 제거·본문 보존: {out!r}"


def check_chunking() -> Result:
    """split_text가 크기와 겹침을 지키는지."""
    from rag_pipeline import split_text, CHUNK_SIZE, CHUNK_OVERLAP
    text = "".join(f"sentence number {i}. " for i in range(400))
    chunks = split_text(text)
    if len(chunks) < 2:
        return False, f"청크 {len(chunks)}개 — 분할 안 됨"
    if any(len(c) > CHUNK_SIZE for c in chunks):
        return False, f"CHUNK_SIZE({CHUNK_SIZE}) 초과 청크 존재"
    tail = chunks[0][-CHUNK_OVERLAP:]
    if tail[:50] not in chunks[1]:
        return False, "인접 청크 간 겹침이 확인되지 않음"
    return True, f"{len(chunks)}청크, 크기≤{CHUNK_SIZE}, 겹침 확인"


# ---------------------------------------------------------------------------
# 파이프라인 로직 (외부 호출 없음)
# ---------------------------------------------------------------------------

def check_language_detection() -> Result:
    from llm_client import LLMClient
    cases = [("뉴로 심볼릭 ai", "ko"), ("multi-agent learning", "en"), ("환각과 의학", "ko")]
    for text, want in cases:
        got = LLMClient.detect_language(text)
        if got != want:
            return False, f"{text!r} → {got!r}, 기대 {want!r}"
    return True, f"{len(cases)}건 모두 정확"


def check_keyword_parsing() -> Result:
    """구분자 ' | '를 보존해야 rag_pipeline이 구문별로 검색할 수 있다."""
    from llm_client import LLMClient
    out = LLMClient._parse_keywords("a phrase | b phrase | c phrase", "fallback")
    if " | " not in out:
        return False, f"구분자 소실: {out!r}"
    if len(out.split(" | ")) != 3:
        return False, f"구문 수 이상: {out!r}"
    numbered = LLMClient._parse_keywords("1. alpha\n2. beta\n3. gamma", "fallback")
    if " | " not in numbered:
        return False, f"번호목록 분기에서 구분자 소실: {numbered!r}"
    return True, f"파이프/번호목록 모두 구분자 보존: {out!r}"


def check_selection_parsing() -> Result:
    """모델이 마커를 느슨하게 써도 번호와 이유를 뽑아내는지."""
    from llm_client import LLMClient
    idx, reason = LLMClient._parse_selection("번호: 2\n이유: 이 논문이 적합합니다.", 5)
    if idx != 2 or not reason:
        return False, f"한국어 느슨한 마커 실패: idx={idx}, reason={reason!r}"
    idx_en, reason_en = LLMClient._parse_selection("Number: 3\nReason: It fits.", 5)
    if idx_en != 3 or not reason_en:
        return False, f"영어 마커 실패: idx={idx_en}, reason={reason_en!r}"
    idx_oob, _ = LLMClient._parse_selection("번호: 99\n이유: x", 5)
    if idx_oob != 0:
        return False, f"범위 초과 번호를 0으로 보정하지 않음: {idx_oob}"
    return True, "한/영 느슨한 마커 파싱·범위 보정 정상"


def check_bullet_normalization() -> Result:
    """한국어 응답의 번호목록을 하이픈 불릿으로 통일하되 소수점은 보존."""
    from llm_client import normalize_bullets
    out = normalize_bullets("1. 첫째\n2. 둘째")
    if out != "- 첫째\n- 둘째":
        return False, f"번호→하이픈 변환 실패: {out!r}"
    keep = normalize_bullets("성능이 3.5배 향상")
    if keep != "성능이 3.5배 향상":
        return False, f"문장 중 소수점 훼손: {keep!r}"
    return True, "번호목록 변환·소수점 보존 정상"


def check_bibliography_filter() -> Result:
    """참고문헌 청크는 걸러내고, 본문 인용이 많은 청크는 남겨야 한다."""
    from rag_pipeline import _looks_like_bibliography
    bib = (
        "[12] Smith, J., Lee, K. (2020). Deep learning for graphs. In Proceedings of "
        "NeurIPS, pp. 1-9. [13] Wang, Y., Chen, L. (2021). Attention is all. arXiv "
        "preprint arXiv:2101.00001. [14] Kim, S., Park, H. (2019). Retrieval models. "
        "Journal of AI, vol. 3, pp. 22-30. doi:10.1000/xyz. J. Doe, A. Roe, 2018."
    )
    body = (
        "Prior work has studied retrieval for question answering (Smith et al., 2020; "
        "Wang and Chen, 2021). Unlike these methods, we combine lexical and dense "
        "signals (Kim et al., 2019), which improves recall on rare terms such as "
        "algorithm names. As shown in Table 2, our model outperforms the baseline."
    )
    if not _looks_like_bibliography(bib):
        return False, "참고문헌 청크를 걸러내지 못함"
    if _looks_like_bibliography(body):
        return False, "본문 인용이 많은 청크를 참고문헌으로 오판"
    return True, "참고문헌 검출, 본문 인용 문단은 유지"


def check_acronym_and_weak_match() -> Result:
    """
    키워드에서 사라진 약어 보존 + "직접 다루지 않는다" 신호 판정.
    refine_query를 실제로 거쳐서 약어 보존이 배선돼 있는지도 본다.
    """
    from rag_pipeline import RAGPipeline

    class FakeLLM:
        def refine_query(self, q, avoid_tokens=None):
            return "Multi-Agent Systems | Machine Learning Agents | Collaborative Computing"

    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe._llm, pipe._last_keywords = FakeLLM(), None
    kw = pipe.refine_query("mcp가 뭐야?")
    if not kw.startswith("MCP AI"):
        return False, f"사라진 약어가 보존되지 않음: {kw!r}"
    if len(kw.split(" | ")) != 3:
        return False, f"구문 수 3개 유지 안 됨: {kw!r}"

    class KeepLLM:
        def refine_query(self, q, avoid_tokens=None):
            return "LLM self-improvement | model adaptation | self-play"

    pipe._llm = KeepLLM()
    if pipe.refine_query("llm이 진화할 수 있어?") != "LLM self-improvement | model adaptation | self-play":
        return False, "이미 들어 있는 약어인데 키워드를 바꿈"

    weak = "사용자의 질문인 'mcp'에 대한 직접적인 정의나 설명은 없지만, 관련성이 있습니다."
    strong = "이 논문은 MAST라는 실패 분류 체계를 제안하여 직접적인 근거를 제공합니다."
    if not RAGPipeline.is_weak_match(weak):
        return False, "'직접 다루지 않음' 신호를 못 잡음"
    if RAGPipeline.is_weak_match(strong):
        return False, "정상적인 선택 이유를 약한 일치로 오판"
    return True, f"약어 보존({kw.split(' | ')[0]})·기존 약어 유지·약한 일치 판정 정상"


def check_bm25_keywords() -> Result:
    """
    한국어 질문에서도 BM25가 동작하도록, 검색 단계가 영어 키워드를 함께 넘기는가.
    _retrieve가 실제로 store.search에 넘기는 값을 가로채서 확인한다(배선 검사).
    """
    from rag_pipeline import RAGPipeline
    store, _ = _sample_store()
    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe._store, pipe._embedder = store, embedder()
    pipe._paper_keywords = "XZQFOO7 | retrieval augmented generation"

    seen = {}
    orig = store.search

    def spy(qv, query_text="", **kw):
        seen["text"] = query_text
        return orig(qv, query_text=query_text, **kw)

    store.search = spy
    q = "이 방법의 핵심 아이디어는?"
    hits = pipe._retrieve(q, embedder().embed_query(q))
    text = seen.get("text", "")
    if "XZQFOO7" not in text:
        return False, f"BM25에 영어 키워드가 전달되지 않음: {text!r}"
    if not any("XZQFOO7" in h["chunk"] for h in hits):
        return False, "키워드를 넘겼는데도 어휘 일치 청크가 결과에 없음"
    return True, "한국어 질문에 영어 키워드를 붙여 BM25 정확 매치 동작"


def check_definition_overview() -> Result:
    """정의 질문이면 survey 검색어를 맨 앞에 넣고, 선택 시 개요 논문을 표시·우선하는가."""
    from rag_pipeline import RAGPipeline
    from llm_client import LLMClient

    class FakeLLM:
        def refine_query(self, q, avoid_tokens=None):
            return "MCP AI | Multi-Agent Systems | Machine Learning Agents"

    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe._llm, pipe._last_keywords = FakeLLM(), None
    kw = pipe.refine_query("mcp가 뭐야?")
    if not kw.split(" | ")[0].endswith("survey"):
        return False, f"정의 질문인데 survey 검색어가 맨 앞에 없음: {kw!r}"
    kw2 = pipe.refine_query("AI 등장에 따른 미래 직업의 변화는 무엇인가")
    if "survey" in kw2:
        return False, f"정의 질문이 아닌데 survey를 붙임: {kw2!r}"
    from rag_pipeline import _is_definition_intent
    if not _is_definition_intent("end to end automation이 뭐야?"):
        return False, "영어 용어가 긴 정의 질문을 놓침 ('end to end automation이 뭐야?')"
    # 여러 단어 영어 용어에는 survey를 붙이지 않는다 — 붙였더니 "Kilo Degree Survey" 같은
    # 무관한 논문이 후보를 채우고 원래 2위였던 논문이 사라졌다(관찰)
    class E2ELLM:
        def refine_query(self, q, avoid_tokens=None):
            return "end-to-end automation | workflow orchestration | AI workflow management"

    pipe._llm = E2ELLM()
    kw3 = pipe.refine_query("end to end automation이 뭐야?")
    if "survey" in kw3:
        return False, f"여러 단어 용어인데 survey를 붙임: {kw3!r}"
    pipe._llm = FakeLLM()
    if not pipe.refine_query("what is MCP").split(" | ")[0].endswith("survey"):
        return False, "영어 정의 질문의 한 단어 용어(what is MCP)에 survey가 붙지 않음"
    from llm_client import _OVERVIEW_TITLE
    real = ["A Survey of MCP", "YOLOv1 to YOLOv11: A Comprehensive Survey of Real-Time Detection",
            "Graph Neural Networks: A Review", "An Introduction to Diffusion Models"]
    fake = ["Automation of finding strong lenses in the Kilo Degree Survey",
            "The End of Work is Near? Survey Evidence on Automation Angst"]
    if [t for t in real if not _OVERVIEW_TITLE.search(t)] or [t for t in fake if _OVERVIEW_TITLE.search(t)]:
        return False, "개요 논문 제목 판정 이상(서베이 형태가 아닌 'Survey' 단어에 반응)"

    prompts = []
    client = LLMClient()
    client._generate_sync = lambda prompt, **k: prompts.append(prompt) or "번호: 0\n이유: x"
    papers = [{"title": "A Survey of MCP", "authors": ["a"], "abstract": "x"},
              {"title": "MCP in Healthcare", "authors": ["b"], "abstract": "y"}]
    pipe._llm = client
    pipe.select_paper("mcp가 뭐야?", papers)
    pipe.select_paper("mcp를 의료에 적용한 사례", papers)
    if "[개요 논문] A Survey of MCP" not in prompts[0] or "개념의 정의를 묻고" not in prompts[0]:
        return False, "정의 질문인데 선택 프롬프트에 개요 논문 표시·안내가 없음"
    if "[개요 논문]" in prompts[1]:
        return False, "정의 질문이 아닌데 개요 논문 표시를 붙임"
    return True, f"survey 검색어({kw.split(' | ')[0]})·개요 논문 표시·비정의 질문 무변경"


def check_abstract_in_context() -> Result:
    """답변 생성 프롬프트(일반·스트리밍·폴백)에 논문 초록이 들어가는가."""
    from llm_client import LLMClient
    abstract = "We propose ZETA-FRAME, a protocol for tool use."
    paper = {"title": "t", "authors": ["a"], "published": "2025-01-01", "abstract": abstract}
    chunks = [{"chunk": "Section 4 describes a clinical case study."}]
    client = LLMClient()
    got = {}
    client._generate_sync = lambda prompt, **k: got.setdefault("sync", []).append(prompt) or "- x"
    client._generate_stream = lambda prompt, **k: (got.setdefault("stream", []).append(prompt), iter(["- x"]))[1]
    client.generate_rag_response("q", chunks, paper, "ko")
    list(client.stream_rag_response("q", chunks, paper, "ko"))
    client.generate_fallback_summary("q", chunks, paper, "en")
    paths = {"일반": got["sync"][0], "스트리밍": got["stream"][0], "폴백": got["sync"][1]}
    missing = [k for k, p in paths.items() if abstract not in p]
    if missing:
        return False, f"초록이 빠진 답변 경로: {missing}"
    return True, "일반·스트리밍·폴백 답변 모두 초록 포함"


def check_new_search_intent() -> Result:
    """
    "최신 논문을 가져와줘"는 직전 논문의 후속 질문이 아니라 새 검색이어야 한다
    (관찰: "논문"이 들어 있어 후속 질문으로 분류돼 이미 읽은 논문으로 답함).
    LLM 분류기가 무조건 followup이라고 답하게 해서, 판정 순서까지 확인한다.
    """
    from rag_pipeline import RAGPipeline

    class AlwaysFollowup:
        def classify_query(self, q, title):
            return "followup"

    class Store:
        size = 10

    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe._llm, pipe._store, pipe._current_paper = AlwaysFollowup(), Store(), {"title": "t"}
    new = ["ai 제일 최신 논문을 가져와줘", "다른 논문 찾아줘", "이 논문 말고 다른 논문 추천해줘",
           "find another paper on RAG"]
    follow = ["이 논문에서 실험 결과 찾아줘", "방금 내용 더 설명해줘", "explain this section"]
    wrong_new = [q for q in new if pipe.is_followup(q)]
    if wrong_new:
        return False, f"새 검색 요청을 후속 질문으로 처리: {wrong_new}"
    wrong_follow = [q for q in follow if not pipe.is_followup(q)]
    if wrong_follow:
        return False, f"후속 질문을 새 검색으로 처리: {wrong_follow}"
    return True, f"새 검색 {len(new)}건·후속 질문 {len(follow)}건 모두 올바르게 분기"


def check_recency_filter() -> Result:
    """
    최신 의도면 S2에 연도 필터가 실제로 전달되는가(질의 → refine → search → HTTP 파라미터).
    S2 검색 API는 날짜 정렬을 무시하므로 연도 필터가 유일한 최신화 수단이다.
    결과가 0건이면 필터 없이 다시 검색해야 한다.
    """
    from datetime import date
    from rag_pipeline import RAGPipeline
    from semantic_scholar_tool import SemanticScholarTool

    want = f"{date.today().year - 1}-"

    class FakeLLM:
        def refine_query(self, q, avoid_tokens=None):
            return "artificial intelligence trends | machine learning | deep learning"

    class FakeS2:
        def __init__(self, empty_when_filtered=False):
            self.years, self.empty = [], empty_when_filtered

        def search(self, query, max_results=10, backfill=True, year=None):
            self.years.append(year)
            if year and self.empty:
                return []
            return [{"id": f"p{i}", "doi": "10.1/x", "venue": "", "citation_count": 0,
                     "published": "2025-05-01", "pdf_url": "u", "abstract": "a"} for i in range(5)]

        def backfill_abstracts(self, papers):
            pass

        def is_pdf_likely_available(self, p):
            return True

    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe._llm, pipe._last_keywords = FakeLLM(), None
    pipe._s2 = FakeS2()
    kw = pipe.refine_query("ai 최신 트렌드 알려줘")
    pipe.search_papers(kw)
    if not pipe._s2.years or any(y != want for y in pipe._s2.years):
        return False, f"최신 의도인데 연도 필터 미전달: {pipe._s2.years}"

    pipe._s2 = FakeS2()
    kw = pipe.refine_query("트랜스포머 구조 설명해줘")
    pipe.search_papers(kw)
    if any(pipe._s2.years):
        return False, f"최신 의도가 아닌데 연도 필터 적용: {pipe._s2.years}"

    pipe._s2 = FakeS2(empty_when_filtered=True)
    kw = pipe.refine_query("요즘 나온 논문 보여줘")
    papers, _ = pipe.search_papers(kw)
    if not papers or pipe._s2.years[-1] is not None or pipe.search_year is not None:
        return False, f"연도 필터 결과 0건인데 필터 없이 재검색하지 않음: {pipe._s2.years}"

    # 도구 단계: year가 실제 HTTP 파라미터로 나가는가
    sent = {}

    class Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": []}

    tool = SemanticScholarTool.__new__(SemanticScholarTool)
    tool._rate_delay = 0
    tool._s2_get = lambda url, params=None, timeout=None: (sent.update(params), Resp())[1]
    tool._s2_search("x", 10, year=want)
    if sent.get("year") != want:
        return False, f"S2 요청 파라미터에 year 없음: {sent}"
    return True, f"최신 의도 → year={want} 전달, 비최신 무필터, 0건 시 필터 해제 재검색"


def check_newest_request() -> Result:
    """
    "제일 최신 논문" 요청: 검색어의 시기 표현 제거, 올해부터 찾고 단계적으로 넓히기,
    선택 시 날짜순 정렬·날짜 표시. 관찰된 실패(제목에 "Advancements"가 든 분야 무관
    논문이 후보를 채우고, 10월 논문을 두고 5월 논문을 고름)를 재현 입력으로 쓴다.
    """
    from datetime import date
    import rag_pipeline as rp
    from rag_pipeline import RAGPipeline
    from llm_client import LLMClient

    seen = "AI latest advancements | cutting-edge AI research | contemporary machine learning breakthroughs"
    got = rp._strip_recency_words(seen)
    if re.search(r"latest|cutting|contemporary|advancement|breakthrough", got, re.IGNORECASE):
        return False, f"시기 표현이 남음: {got!r}"
    keep = "modern Hopfield networks | novel view synthesis | drug development"
    if rp._strip_recency_words(keep) != keep:
        return False, f"기술 이름의 일부를 지움: {rp._strip_recency_words(keep)!r}"
    if rp._strip_recency_words("latest advancements") != "latest advancements":
        return False, "전부 지워질 때 원래 검색어로 돌아가지 않음"

    class FakeLLM:
        def refine_query(self, q, avoid_tokens=None):
            return seen

    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe._llm, pipe._last_keywords = FakeLLM(), None
    kw = pipe.refine_query("ai관련된 제일 최신 논문을 찾아줘")
    y = date.today().year
    if "latest" in kw.lower():
        return False, f"refine_query에서 시기 표현 제거가 호출되지 않음: {kw!r}"
    if pipe.search_year != f"{y}-" or not pipe.wants_newest:
        return False, f"'제일 최신'인데 올해 필터/날짜 우선이 아님: {pipe.search_year}, {pipe.wants_newest}"
    if pipe.refine_query("ai 트렌드 알려줘") != seen or pipe.wants_newest:
        return False, "최신 의도가 아닌데 검색어를 바꾸거나 날짜 우선을 켬"
    if rp._relax_year(f"{y}-") != f"{y - 1}-" or rp._relax_year(f"{y - 1}-") is not None:
        return False, "연도 필터 완화 단계가 올해 → 작년 이후 → 제한 없음이 아님"

    prompts = []
    client = LLMClient()
    client._generate_sync = lambda prompt, **k: prompts.append(prompt) or "번호: 0\n이유: x"
    papers = [{"title": "Older", "authors": ["a"], "abstract": "x", "published": "2026-03-01"},
              {"title": "Newer", "authors": ["b"], "abstract": "y", "published": "2026-09-20"}]
    pipe._llm = client
    pipe.wants_newest = True
    chosen, _ = pipe.select_paper("ai관련된 제일 최신 논문을 찾아줘", papers)
    p = prompts[-1]
    if "2026-09-20" not in p or "가장 최근" not in p:
        return False, "'제일 최신'인데 선택 프롬프트에 날짜·안내가 없음"
    if p.index("Newer") > p.index("Older"):
        return False, "후보가 최신순으로 정렬되지 않음"
    pipe.wants_newest = False
    pipe.select_paper("트랜스포머 구조", papers)
    if "[논문_날짜]" in prompts[-1]:
        return False, "평소 질문에도 날짜를 보여줌(선택이 날짜에 끌릴 수 있음)"
    return True, f"시기 표현 제거({got!r})·기술명 보존·올해→작년→제한없음·날짜순 선택"


def check_fallback_order() -> Result:
    """PDF 실패 시 대체 후보를 목록 순서가 아니라 질문과 가까운 순으로 시도하는가."""
    from rag_pipeline import RAGPipeline

    papers = [
        {"id": "sel", "title": "Selected paper", "abstract": ""},
        {"id": "chem", "title": "Bismuth nanostructures for catalysis", "abstract": "Chemistry of bismuth."},
        {"id": "xai", "title": "Explainable AI methods for deep neural networks",
         "abstract": "We review explanation methods for neural network predictions."},
    ]

    class Pipe(RAGPipeline):
        def __init__(self):
            self._embedder = embedder()
            self.tried = []

        def index_paper(self, paper):
            self.tried.append(paper["id"])
            return 100 if paper["id"] == "xai" else 0

    pipe = Pipe()
    got, n, sub = pipe.index_with_fallback(papers, papers[0], query="설명 가능한 AI 연구")
    if pipe.tried[:2] != ["sel", "xai"]:
        return False, f"질문과 가까운 후보를 먼저 시도하지 않음: {pipe.tried}"
    plain = Pipe()
    plain.index_with_fallback(papers, papers[0])
    if plain.tried != ["sel", "chem", "xai"]:
        return False, f"질문 없이 부르면 목록 순서여야 함: {plain.tried}"
    return True, "질문 있으면 유사도 순(sel→xai), 없으면 목록 순"


def check_vague_query() -> Result:
    """
    한 단어 질문은 검색 전에 되묻고, 선택지는 각각 정의·최신 검색 경로로 이어지는가.
    (관찰: "컴퓨터" → computer architecture 검색 → 이미지 인식 논문 Inception)
    """
    from rag_pipeline import RAGPipeline, _is_definition_intent, _recency_year
    vague = ["컴퓨터", "인공지능", "llm", "AI", "computer", "컴퓨터?"]
    specific = ["QMIX", "RLHF", "컴퓨터 구조", "mcp가 뭐야?", "최신 논문",
                "이 논문 요약해줘", "트랜스포머의 어텐션 구조 설명해줘", "what is RAG"]
    miss = [q for q in vague if not RAGPipeline.is_vague_query(q)]
    if miss:
        return False, f"한 단어 질문을 되묻지 않음: {miss}"
    over = [q for q in specific if RAGPipeline.is_vague_query(q)]
    if over:
        return False, f"구체적인 질문까지 되물음: {over}"
    opts = RAGPipeline.clarify_options("컴퓨터", "ko")
    define, recent, as_is = (q for _, q in opts)
    if not _is_definition_intent(define):
        return False, f"'개념 설명' 선택지가 정의 질문 경로로 가지 않음: {define!r}"
    if not _recency_year(recent):
        return False, f"'최신 연구' 선택지가 연도 필터로 이어지지 않음: {recent!r}"
    if as_is != "컴퓨터":
        return False, f"'그대로 검색'이 원래 질문이 아님: {as_is!r}"
    with open(os.path.join(ROOT, "app.py"), encoding="utf-8") as fh:
        src = fh.read()
    if "pipeline.is_vague_query(" not in src or "_skip_clarify" not in src:
        return False, "app.py에 되묻기(또는 '그대로 검색' 우회)가 연결되지 않음"
    return True, f"한 단어 {len(vague)}건 되묻기·구체적 질문 {len(specific)}건 통과·선택지 경로 정상"


def check_prompt_consistency() -> Result:
    """
    답변·선택 프롬프트끼리 서로 반대 방향을 지시하지 않는가.
      - 출력 예시가 "기존 대비 D만큼 향상"을 유도하면 수치 규칙과 충돌한다
        (관찰: "top-1 오류율 21.2%"를 "21.2% 향상된 오류율"로 씀)
      - 범위 규칙("이 논문은 ~ 관점에서")이 헤지 금지 규칙과 충돌하지 않게 명시
      - 개요 우선·최신 우선 안내가 동시에 붙지 않음
    """
    import llm_client as lc
    ko, en = lc._RAG_PROMPT_KO, lc._RAG_PROMPT_EN
    if "낮을수록 좋은 지표" not in ko or "lower-is-better" not in en:
        return False, "답변 프롬프트에 수치 의미 규칙이 없음"
    example = ko.split("출력 형식 예시")[1].split("<사용자_질문>")[0]
    if re.search(r"대비.{0,10}향상", example):
        return False, f"출력 예시가 '기존 대비 향상' 표현을 유도함: {example.strip()!r}"
    if "헤지)가 아니" not in ko or "not a hedge" not in en:
        return False, "범위 규칙과 헤지 금지 규칙의 관계가 명시되지 않음"
    if "개념이 무엇인지 한 문장으로 정의" not in ko or "define the concept in one sentence" not in en:
        return False, "정의 질문에 첫 줄 정의 규칙이 없음"
    if "범위 규칙" not in ko.split("개념이 무엇인지 한 문장으로 정의")[1][:200]:
        return False, "정의 규칙과 범위 규칙이 함께 걸릴 때의 처리 방법이 없음"
    if "metric names are the exception" not in en:
        return False, "영어 '원문 그대로 쓰지 말 것' 규칙이 수치까지 바꾸게 둠"

    prompts = []
    client = lc.LLMClient()
    client._generate_sync = lambda prompt, **k: prompts.append(prompt) or "번호: 0\n이유: x"
    papers = [{"title": "A Survey of MCP", "authors": ["a"], "abstract": "x", "published": "2026-01-01"}]
    client.select_best_paper("mcp 제일 최신이 뭐야", papers, prefer_overview=True, prefer_recent=True)
    if "[개요 논문]" in prompts[-1] or "개념의 정의를 묻고" in prompts[-1]:
        return False, "개요 우선·최신 우선 안내가 동시에 붙음"
    if "가장 최근" not in prompts[-1]:
        return False, "둘 다 켜졌을 때 최신 우선 안내가 빠짐"
    return True, "수치 규칙·예시 일치·범위/헤지 관계 명시·선택 안내 동시 적용 방지"


def check_metric_misread() -> Result:
    """
    답변이 원문의 지표 값(오류율 21.2%)을 변화량(21.2% 향상)으로 쓴 경우를 잡는가.
    원문에 실제 변화량("by 3.2%")이 있으면 잡지 않는가. 스트리밍 경로에서 근거를
    남겨 두고 app.py가 경고를 띄우는지(배선)도 본다.
    """
    from rag_pipeline import RAGPipeline, _metric_misreads
    ctx = ("We demonstrate substantial gains over the state of the art: 21:2% top-1 and 5:6% "
           "top-5 error for single frame evaluation.")
    wrong = ["- 기존 최첨단 모델 대비 21.2% 향상된 top-1 오류율을 달성했습니다.",
             "- top-1 오류율이 기존 대비 21.2% 감소하였습니다.",
             "- The method achieves a 21.2% improvement in top-1 error."]
    missed = [a for a in wrong if not _metric_misreads(a, ctx)]
    if missed:
        return False, f"지표 값을 변화량으로 쓴 답변을 못 잡음: {missed}"
    right = [("- top-1 오류율 21.2%, top-5 오류율 5.6%를 기록했습니다.", ctx),
             ("- 기존 대비 정확도가 3.2% 향상되었습니다.", "Our method improves accuracy by 3.2%."),
             ("- WER이 기존 대비 5% 감소했습니다.", "The model reduces WER by 5% relative to the baseline.")]
    over = [a for a, c in right if _metric_misreads(a, c)]
    if over:
        return False, f"올바른 수치 표현을 오류로 잡음: {over}"

    class FakeLLM:
        def stream_rag_response(self, q, chunks, paper, lang):
            yield "- 기존 대비 21.2% 향상된 top-1 오류율을 달성했습니다."

    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe._llm, pipe._embedder = FakeLLM(), embedder()
    pipe._retrieve = lambda q, v: [{"chunk": ctx}]
    answer = "".join(pipe.stream_answer("실험 결과는?", {"abstract": ""}, "ko"))
    if not pipe.metric_misreads(answer):
        return False, "스트리밍 답변의 근거가 남지 않아 수치 검사가 동작하지 않음"
    with open(os.path.join(ROOT, "app.py"), encoding="utf-8") as fh:
        if "pipeline.metric_misreads(" not in fh.read():
            return False, "app.py에 수치 경고가 연결되지 않음"
    return True, f"잘못 쓴 수치 {len(wrong)}건 검출·올바른 표현 {len(right)}건 통과·스트리밍 경로 배선"


def check_pdf_ligatures() -> Result:
    """
    합자(ﬁ ﬀ ﬃ)·수학용 이탤릭 문자가 공백이 되지 않고 글자로 풀리는가.
    (실측: 캐시 논문 40편 중 28편에서 "efficient"가 "e cient"로 깨져 있었다)
    """
    from pdf_extractor import clean_pdf_text
    got = clean_pdf_text("classiﬁcation is eﬃcient and ﬂexible; 𝑡 = 10² ms")
    want = "classification is efficient and flexible; t = 10² ms"
    if got != want:
        return False, f"합자/수학 문자 처리 이상: {got!r}"
    return True, "합자 3종·수학 이탤릭 복원, 위첨자(10²)는 보존"


def check_embedding_cache_key() -> Result:
    """
    같은 논문이라도 청크 내용이 바뀌면 옛 임베딩을 쓰지 않는가.
    (예전 캐시는 논문 ID+청크 개수만 봐서, 정제 방식을 고쳐도 옛 벡터를 그대로 썼다)
    """
    import tempfile
    import rag_pipeline as rp
    from rag_pipeline import RAGPipeline

    texts = iter(["Alpha beta gamma. " * 60, "Quantum qubit entanglement. " * 39])

    class S2:
        def download_pdf(self, p):
            return "unused.pdf"

    from vector_store import VectorStore
    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe._s2, pipe._embedder, pipe._store = S2(), embedder(), VectorStore()
    pipe._last_keywords, pipe._paper_keywords = None, None
    paper = {"id": "cache-key-test", "title": "Alpha beta gamma"}
    cwd, tmp = os.getcwd(), tempfile.mkdtemp()
    orig = rp.extract_pdf_text
    rp.extract_pdf_text = lambda path: next(texts)
    os.chdir(tmp)
    try:
        n1 = pipe.index_paper(dict(paper))
        n2 = pipe.index_paper(dict(paper))
        stored = pipe._store._index.reconstruct(0)
        fresh = embedder().embed_documents([pipe._store._chunks[0]])[0]
    finally:
        os.chdir(cwd)
        rp.extract_pdf_text = orig
        shutil.rmtree(tmp, ignore_errors=True)
    if n1 != n2:
        return False, f"검사 전제 실패 — 두 텍스트의 청크 수가 달라짐({n1}, {n2})"
    if float(stored @ fresh) < 0.999:
        return False, "청크 내용이 바뀌었는데 옛 임베딩을 재사용함"
    return True, f"같은 ID·같은 청크 수({n1})에서도 내용이 바뀌면 새로 임베딩"


def check_answer_echo() -> Result:
    """
    답변 앞머리의 "[최종 답변]:"만 지우고, 본문 중간의 "답변:"이 있는 정상 답변은
    자르지 않는가. (예전 Thinking/Answer 파서는 그 앞을 통째로 버렸다)
    """
    from llm_client import LLMClient, strip_answer_echo
    body = "- 저자들은 사용자 질문의 답변: 생성 과정을 3단계로 나눕니다.\n- 결과는 B입니다."
    client = LLMClient()
    client._generate_sync = lambda prompt, **k: body
    _, ans = client.generate_rag_response("q", [{"chunk": "c"}],
                                          {"title": "t", "published": "2025", "abstract": ""}, "ko")
    if ans != body:
        return False, f"본문 중간의 '답변:' 때문에 답변이 잘림: {ans!r}"
    if strip_answer_echo("[최종 답변]: - 이 논문은 A를 제안합니다.") != "- 이 논문은 A를 제안합니다.":
        return False, "앞머리에 따라 쓴 답변 표시를 지우지 못함"
    with open(os.path.join(ROOT, "app.py"), encoding="utf-8") as fh:
        if "strip_answer_echo(" not in fh.read():
            return False, "스트리밍 경로(app.py)에 답변 표시 제거가 연결되지 않음"
    return True, "앞머리 표시만 제거, 본문의 '답변:'은 보존, 스트리밍 경로 배선"


def check_followup_keywords() -> Result:
    """
    "더 알려줘", "what does"처럼 새 주제에도 쓰는 말은 키워드로 곧장 후속 질문 처리하지
    않고 LLM 분류기에 넘기는가. 지금 논문을 분명히 가리키는 말은 바로 후속 질문인가.
    """
    from rag_pipeline import RAGPipeline

    class LLM:
        calls = 0

        def classify_query(self, q, title):
            LLM.calls += 1
            return "new"

    class Store:
        size = 10

    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe._llm, pipe._store, pipe._current_paper = LLM(), Store(), {"title": "MCP survey"}
    for q in ("강화학습에 대해 더 알려줘", "what does RLHF stand for", "계속 알려줘"):
        if pipe.is_followup(q):
            return False, f"새 주제일 수 있는 질문을 키워드만으로 후속 질문 처리: {q!r}"
    asked = LLM.calls
    for q in ("방금 논문 요약해줘", "이 논문의 한계는?", "summarize this paper"):
        if not pipe.is_followup(q):
            return False, f"지금 논문을 가리키는데 후속 질문으로 보지 않음: {q!r}"
    if LLM.calls != asked:
        return False, "명확한 후속 질문인데 LLM 분류기를 불렀음(불필요한 지연)"
    return True, "애매한 표현 3건은 LLM 판단, 명확한 지칭 3건은 즉시 후속 처리"


def check_network_waits() -> Result:
    """
    S2 호출 간격은 '남은 시간만' 기다리는가(첫 호출은 바로). PDF 사전 확인은 동시에
    하면서 순서를 유지하는가. 둘 다 느린 와이파이에서 체감 시간을 좌우한다.
    """
    import semantic_scholar_tool as s2mod
    from semantic_scholar_tool import SemanticScholarTool
    from rag_pipeline import RAGPipeline

    tool = SemanticScholarTool.__new__(SemanticScholarTool)
    tool._rate_delay, tool._s2_headers = 0.4, {}
    orig_get = s2mod.requests.get
    s2mod.requests.get = lambda *a, **k: None
    try:
        t0 = time.perf_counter(); tool._s2_get("u"); first = time.perf_counter() - t0
        t0 = time.perf_counter(); tool._s2_get("u"); second = time.perf_counter() - t0
    finally:
        s2mod.requests.get = orig_get
    if first > 0.1:
        return False, f"첫 S2 호출인데 {first:.2f}초 기다림"
    if second < 0.3:
        return False, f"연속 호출 간격 미준수({second:.2f}초 < 0.4초)"

    class S2:
        def is_pdf_likely_available(self, p):
            time.sleep(0.2)
            return p["id"] != "x2"

    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe._s2 = S2()
    papers = [{"id": f"x{i}"} for i in range(5)]
    t0 = time.perf_counter()
    got = pipe._pdf_prefilter(papers)
    took = time.perf_counter() - t0
    if [p["id"] for p in got] != ["x0", "x1", "x3", "x4"]:
        return False, f"PDF 사전 확인 결과 순서/내용 이상: {[p['id'] for p in got]}"
    if took > 0.6:
        return False, f"PDF 사전 확인이 차례로 실행됨({took:.2f}초, 5건×0.2초)"
    return True, f"첫 S2 호출 {first:.2f}초·연속 {second:.2f}초, PDF 확인 5건 {took:.2f}초(순서 유지)"


def check_candidate_switch() -> Result:
    """
    후보 목록에서 다른 논문을 골라 다시 답하기:
      - 고른 논문으로 인덱스가 바뀌는가
      - PDF를 못 받으면 원래 논문이 그대로 남는가(후속 질문이 끊기지 않게)
      - 목록 순서: 최신 의도면 최신순, 아니면 검색 관련도순
      - TLDR이 후보 정보에 담기는가, app.py에 패널·교체가 연결돼 있는가
    """
    from rag_pipeline import RAGPipeline
    from semantic_scholar_tool import SemanticScholarTool as T

    class Pipe(RAGPipeline):
        def __init__(self, fail_ids):
            self.fail_ids, self._current_paper, self.loaded = fail_ids, None, []

        def index_paper(self, paper):
            if paper["id"] in self.fail_ids:
                self._current_paper = paper        # 실제 index_paper도 실패 전에 바꿔 둔다
                return 0
            self._current_paper = paper
            self.loaded.append(paper["id"])
            return 50

    a, b, c = {"id": "A", "title": "a"}, {"id": "B", "title": "b"}, {"id": "C", "title": "c"}
    pipe = Pipe({"C"})
    pipe.index_paper(a)
    if pipe.switch_paper(b) != 50 or pipe.current_paper["id"] != "B":
        return False, "고른 논문으로 바뀌지 않음"
    if pipe.switch_paper(c) != 0 or pipe.current_paper["id"] != "B":
        return False, f"PDF 실패 시 원래 논문으로 복구되지 않음: {pipe.current_paper}"

    papers = [{"id": "x", "published": "2021-01-01"}, {"id": "y", "published": "2025-06-01"}]
    pipe.search_year = None
    if [p["id"] for p in pipe.candidates_for_display(papers)] != ["x", "y"]:
        return False, "평소 질문인데 검색 관련도 순서를 바꿈"
    pipe.search_year = "2025-"
    if [p["id"] for p in pipe.candidates_for_display(papers)] != ["y", "x"]:
        return False, "최신 의도인데 최신순이 아님"

    d = T._paper_to_dict({"paperId": "z", "title": "t", "tldr": {"text": "One-line summary."}})
    if d.get("tldr") != "One-line summary.":
        return False, f"TLDR이 후보 정보에 담기지 않음: {d.get('tldr')!r}"
    if T._paper_to_dict({"paperId": "z", "title": "t", "tldr": None}).get("tldr") != "":
        return False, "TLDR이 없을 때 빈 문자열이 아님"

    with open(os.path.join(ROOT, "app.py"), encoding="utf-8") as fh:
        src = fh.read()
    for needle in ("pipeline.switch_paper(", "_render_candidate_panel()", "candidates_for_display("):
        if needle not in src:
            return False, f"app.py에 후보 목록·교체가 연결되지 않음: {needle}"
    return True, "교체 성공·실패 시 원래 논문 유지·목록 순서·TLDR·화면 배선 정상"


def check_stage_timer() -> Result:
    """
    단계별 시간 측정: 단계 합이 전체와 같은가, 기록 파일에 한 줄이 남는가,
    app.py의 각 단계에 연결돼 있는가. Ollama 주소가 localhost(요청당 2초 지연)가 아닌가.
    """
    import json
    import tempfile
    import timing
    import llm_client

    t = timing.StageTimer("새 검색")
    time.sleep(0.05)
    t.lap("a")
    time.sleep(0.03)
    t.lap("b")
    if abs(sum(s for _, s in t.laps) - t.total) > 1e-6 or not (0.07 < t.total < 1.0):
        return False, f"단계 합과 전체가 다름: {t.laps}, {t.total}"
    orig = timing.LOG_PATH
    timing.LOG_PATH = os.path.join(tempfile.mkdtemp(), "logs", "timings.jsonl")
    try:
        t.save("q")
        with open(timing.LOG_PATH, encoding="utf-8") as fh:
            rec = json.loads(fh.read().strip().splitlines()[-1])
    finally:
        timing.LOG_PATH = orig
    if set(rec["laps"]) != {"a", "b"} or len(rec["version"]) != 8:
        return False, f"기록 내용 이상: {rec}"
    with open(os.path.join(ROOT, "app.py"), encoding="utf-8") as fh:
        src = fh.read()
    missing = [n for n in ("질문 판별", "검색어 생성", "논문 검색", "논문 선택", "PDF·인덱싱",
                           "근거 검색 · 첫 글자", "답변 출력", "_show_timing(")
               if n not in src]
    if missing:
        return False, f"app.py에 연결되지 않은 단계: {missing}"
    if "localhost" in llm_client.OLLAMA_BASE_URL:
        return False, f"Ollama 주소가 localhost — 요청마다 약 2초 지연: {llm_client.OLLAMA_BASE_URL}"
    return True, f"단계 합 = 전체, 기록 1줄(버전 {rec['version']}), 화면 7단계 연결, 주소 {llm_client.OLLAMA_BASE_URL}"


def check_scope_rule() -> Result:
    """분야 전체를 묻는 질문에서 논문 한 편을 일반화하지 말라는 규칙이 답변 프롬프트에 들어가는가."""
    from llm_client import LLMClient
    paper = {"title": "t", "authors": ["a"], "published": "2025-01-01", "abstract": "x"}
    client = LLMClient()
    got = []
    client._generate_sync = lambda prompt, **k: got.append(prompt) or "- x"
    client.generate_rag_response("ai 트렌드", [{"chunk": "c"}], paper, "ko")
    client.generate_rag_response("ai trends", [{"chunk": "c"}], paper, "en")
    if "관점에서" not in got[0] or "일반화하지" not in got[0]:
        return False, "한국어 답변 프롬프트에 범위 규칙 없음"
    if "perspective of this paper" not in got[1]:
        return False, "영어 답변 프롬프트에 범위 규칙 없음"
    return True, "한국어·영어 답변 프롬프트 모두 범위 규칙 포함"


def check_bullet_dedup() -> Result:
    """
    어미만 다른 반복 불릿은 하나로 합치고, 단어 하나라도 다른 불릿은 남기는가.
    generate_answer와 app.py(스트리밍 경로)에 실제로 연결돼 있는지도 본다.
    """
    from rag_pipeline import RAGPipeline
    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe._embedder = embedder()

    dup = ("이 논문은 다음을 다룹니다.\n"
           "- 이 논문은 생성형 AI가 노동 시장에 미치는 영향을 분석합니다.\n"
           "- 고숙련 직종일수록 AI 노출도가 높게 나타났습니다.\n"
           "- 이 논문은 생성형 AI가 노동 시장에 미치는 영향을 분석하고 있습니다.")
    out = pipe.dedup_bullets(dup).splitlines()
    if len(out) != 3:
        return False, f"반복 불릿이 합쳐지지 않음({len(out)}줄): {out}"
    if out[0] != "이 논문은 다음을 다룹니다." or "분석하고 있습니다" not in out[1]:
        return False, f"도입 문장 유실 또는 긴 쪽이 남지 않음: {out}"

    for a, b in (("이 논문은 AI 에이전트의 계획 능력을 평가합니다.",
                  "이 논문은 AI 에이전트의 도구 사용 능력을 평가합니다."),
                 ("생성형 AI는 사무직 일자리를 줄입니다.", "생성형 AI는 사무직 일자리를 늘립니다."),
                 ("정확도가 5% 향상되었습니다.", "정확도가 2% 향상되었습니다.")):
        if len(pipe.dedup_bullets(f"- {a}\n- {b}").splitlines()) != 2:
            return False, f"서로 다른 내용을 중복으로 지움: {a!r} / {b!r}"

    # 배선: generate_answer가 dedup을 거치는가
    class FakeLLM:
        def generate_rag_response(self, q, chunks, paper, lang):
            return "", "- 데이터 품질이 성능에 중요합니다.\n- 데이터 품질이 성능에 중요하다고 합니다."

    pipe._llm = FakeLLM()
    pipe._retrieve = lambda q, v: [{"chunk": "c"}]
    _, ans = pipe.generate_answer("q", {"title": "t"}, "ko")
    if len(ans.splitlines()) != 1:
        return False, "generate_answer 경로에 중복 제거 미적용"
    with open(os.path.join(ROOT, "app.py"), encoding="utf-8") as fh:
        if "pipeline.dedup_bullets(" not in fh.read():
            return False, "app.py 스트리밍 경로에 중복 제거 미적용"
    return True, "반복 불릿 합침(긴 쪽 유지)·단어 다른 3쌍 보존·답변 두 경로 배선"


def check_summary_routing() -> Result:
    """
    요약형 질의는 유사도 검색 대신 문서 앞부분을 써야 한다.
    (유사도로는 부록의 실험설정이 뽑혀 엉뚱한 요약이 나온 사례가 있었음)
    """
    from rag_pipeline import RAGPipeline, _is_summary_intent, TOP_K
    for q in ("이 논문 요약해줘", "summarize this paper"):
        if not _is_summary_intent(q):
            return False, f"요약 의도 미감지: {q!r}"
    for q in ("뉴로 심볼릭 ai", "multi-agent reinforcement learning"):
        if _is_summary_intent(q):
            return False, f"일반 질의를 요약으로 오탐: {q!r}"

    store, chunks = _sample_store()
    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe._store = store
    pipe._embedder = embedder()
    got = pipe._retrieve("이 논문 요약해줘", embedder().embed_query("이 논문 요약해줘"))
    if [g["chunk"] for g in got] != chunks[:len(got)]:
        return False, f"앞부분이 아닌 청크 반환: {[g['chunk'][:25] for g in got]}"
    if len(got) != min(TOP_K, len(chunks)):
        return False, f"청크 수 {len(got)}, 기대 {min(TOP_K, len(chunks))}"
    return True, f"요약 질의 → 문서 앞부분 {len(got)}청크 사용"


def check_metadata_suspect() -> Result:
    """arXiv ID의 연도와 S2의 연도가 크게 어긋나면 플래그가 서야 한다."""
    from semantic_scholar_tool import SemanticScholarTool as T
    bad = T._paper_to_dict({
        "paperId": "x", "title": "t", "year": 2014,
        "externalIds": {"ArXiv": "2203.00667"},
    })
    if bad["metadata_suspect"] is not True:
        return False, "8년 차이인데 플래그가 서지 않음"
    ok = T._paper_to_dict({
        "paperId": "y", "title": "t", "year": 2022,
        "externalIds": {"ArXiv": "2203.00667"},
    })
    if ok["metadata_suspect"] is not False:
        return False, "연도 일치인데 오탐"
    none_id = T._paper_to_dict({"paperId": "z", "title": "t", "year": 2020, "externalIds": {}})
    if "metadata_suspect" not in none_id or none_id["metadata_suspect"] is not False:
        return False, "arXiv ID 없을 때 키 누락 또는 오탐"
    return True, "연도 불일치 탐지·정상 케이스 무탐지·키 항상 존재"


def check_reliability_filter() -> Result:
    """STRICT/BALANCED/OFF 판정과 자동 완화 사다리."""
    from rag_pipeline import RAGPipeline, MIN_CITATIONS
    pipe = RAGPipeline.__new__(RAGPipeline)
    doi_paper = {"doi": "10.1/x", "venue": "", "citation_count": 0, "published": "2018-01-01"}
    weak = {"doi": None, "venue": "", "citation_count": 0, "published": "2015-01-01"}
    cited = {"doi": None, "venue": "", "citation_count": MIN_CITATIONS, "published": "2015-01-01"}

    if not pipe._passes_reliability(doi_paper, "STRICT"):
        return False, "DOI 보유 논문이 STRICT 불통과"
    if pipe._passes_reliability(weak, "STRICT"):
        return False, "근거 없는 논문이 STRICT 통과"
    if not pipe._passes_reliability(cited, "BALANCED"):
        return False, f"인용 {MIN_CITATIONS}회 논문이 BALANCED 불통과"
    if not pipe._passes_reliability(weak, "OFF"):
        return False, "OFF인데 탈락"

    # 사다리: STRICT로 걸러지지 않는 논문만 주면 BALANCED/OFF까지 완화되어야 한다
    out = pipe._reliability_filter([weak])
    if len(out) != 1:
        return False, f"완화 사다리가 동작하지 않음 — 결과 {len(out)}건"
    return True, "모드별 판정 정확, 0건 시 자동 완화 동작"


def check_ratelimit_propagation() -> Result:
    """
    429는 '결과 없음'이 아니라 예외로 전파되어야 한다.
    (이전 체크리스트가 이 둘을 혼동해 API 장애를 프로젝트 결함으로 오보했다)
    """
    from rag_pipeline import RAGPipeline
    from semantic_scholar_tool import S2RateLimitError

    class Fake:
        def __init__(self):
            self.calls = 0

        def search(self, *a, **k):
            self.calls += 1
            raise S2RateLimitError("simulated")

        def backfill_abstracts(self, papers):
            pass

    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe._s2 = Fake()
    try:
        pipe.search_papers("a | b | c", max_results=10)
        return False, "429인데 예외 없이 반환됨"
    except S2RateLimitError:
        pass
    if pipe._s2.calls != 1:
        return False, f"첫 429 후에도 계속 호출: {pipe._s2.calls}회"
    return True, "예외 전파 + 첫 실패에서 중단(1회 호출)"


def check_pdf_fallback() -> Result:
    """선택 논문의 PDF 확보가 실패하면 다음 후보로 넘어가야 한다."""
    from rag_pipeline import RAGPipeline

    papers = [{"id": f"p{i}", "title": f"Paper {i}"} for i in range(4)]

    class Pipe(RAGPipeline):
        def __init__(self, fail):
            self.fail = fail
            self.tried = []

        def index_paper(self, paper):
            self.tried.append(paper["id"])
            return 0 if paper["id"] in self.fail else 100

    ok = Pipe(set())
    got, n, sub = ok.index_with_fallback(papers, papers[1])
    if got["id"] != "p1" or n != 100 or sub or ok.tried != ["p1"]:
        return False, f"정상 케이스 이상: {got}, {n}, {sub}, {ok.tried}"

    degraded = Pipe({"p1"})
    got2, n2, sub2 = degraded.index_with_fallback(papers, papers[1])
    if not got2 or got2["id"] == "p1" or n2 != 100 or not sub2:
        return False, f"대체 실패: {got2}, {n2}, {sub2}, {degraded.tried}"

    dead = Pipe({p["id"] for p in papers})
    got3, n3, _ = dead.index_with_fallback(papers, papers[0])
    if got3 is not None or n3 != 0 or len(dead.tried) != 3:
        return False, f"전멸 케이스 이상: {got3}, {n3}, {dead.tried}"
    return True, "정상/대체/전멸 3케이스 모두 기대대로"


# ---------------------------------------------------------------------------
# 전체 파이프라인 (오프라인 E2E)
# ---------------------------------------------------------------------------

def check_end_to_end() -> Result:
    """
    질의 -> 키워드 -> 검색 -> 필터 -> 선택 -> 인덱싱 -> 검색 -> 답변을
    실제 파이프라인 메서드로 관통한다.

    다른 체크들은 함수를 하나씩 직접 불러 "그 함수가 올바른가"를 본다.
    이 체크는 "그 함수들이 실제로 호출되도록 배선되어 있는가"를 본다.
    예: search_papers에서 self._reliability_filter(...) 호출 줄이 사라져도
    check_reliability_filter는 통과한다(함수 자체는 멀쩡하므로). 여기서는
    무근거 논문이 결과에 남는 것으로 즉시 드러난다.

    네트워크가 필요한 부분(S2 검색, PDF 다운로드)만 가짜로 대체하고
    임베딩/FAISS/BM25/LLM은 모두 실제를 쓴다.
    """
    from llm_client import LLMClient
    from rag_pipeline import RAGPipeline

    pdf_dir = os.path.join(ROOT, "pdfs")
    pdfs = sorted(f for f in os.listdir(pdf_dir) if f.endswith(".pdf")) if os.path.isdir(pdf_dir) else []
    if not pdfs:
        return False, "캐시된 PDF 없음 — E2E 실행 불가"
    pdf_path = os.path.join(pdf_dir, pdfs[0])

    def paper(pid, *, doi=None, cites=0, published="2015-01-01"):
        return {
            "id": pid, "title": f"Paper {pid}", "abstract": f"Abstract of {pid}.",
            "authors": ["A. Author"], "published": published,
            "url": f"http://example.invalid/{pid}", "pdf_url": f"http://example.invalid/{pid}.pdf",
            "arxiv_id": None, "doi": doi, "venue": "", "pub_types": [],
            "citation_count": cites, "influential_citations": 0, "metadata_suspect": False,
        }

    # DOI 보유 -> BALANCED 통과 / DOI·venue 없고 인용 0에 오래된 논문 -> 탈락
    reliable = [paper(f"good{i}", doi=f"10.1000/{i}") for i in range(3)]
    unreliable = [paper(f"bad{i}") for i in range(2)]
    bad_ids = {p["id"] for p in unreliable}

    class FakeS2:
        """네트워크만 대체 — 반환 내용은 고정이라 판정이 확정적이다."""

        def search(self, query, max_results=10, backfill=True, year=None):
            return [dict(p) for p in reliable + unreliable]

        def backfill_abstracts(self, papers):
            pass

        def is_pdf_likely_available(self, p):
            return bool(p.get("pdf_url") or p.get("arxiv_id"))

        def download_pdf(self, p):
            return pdf_path

    pipe = RAGPipeline(embedder=embedder(), llm=LLMClient())
    pipe._s2 = FakeS2()

    # 1) 키워드 정제 — 파이프라인 진입점
    keywords = pipe.refine_query("이 논문의 주요 방법론은 무엇인가?")
    if not keywords.strip():
        return False, "refine_query가 빈 문자열 반환"

    # 2) 검색 + 신뢰도 필터 배선
    papers, _ = pipe.search_papers(keywords, max_results=10)
    if not papers:
        return False, "search_papers가 후보 0건 반환(가짜 S2는 5건을 준다)"
    leaked = [p["id"] for p in papers if p["id"] in bad_ids]
    if leaked:
        return False, f"신뢰도 필터 미배선 — 무근거 논문이 후보에 남음: {leaked}"
    if len(papers) != len(reliable):
        return False, f"후보 {len(papers)}건, 기대 {len(reliable)}건: {[p['id'] for p in papers]}"

    # 3) 논문 선택 — 반드시 후보 안에서 골라야 한다
    selected, reasoning = pipe.select_paper("이 논문의 주요 방법론은 무엇인가?", papers)
    if selected["id"] not in {p["id"] for p in papers}:
        return False, f"후보에 없는 논문 선택: {selected['id']}"
    if not reasoning.strip():
        return False, "선택 이유가 비어 있음"

    # 4) 다운로드 -> 추출 -> 청킹 -> 임베딩 -> 인덱싱 사슬
    selected["id"] = "e2e-test-paper"   # 임베딩 캐시를 실제 논문과 분리
    indexed, n_chunks, _ = pipe.index_with_fallback(papers, selected)
    if n_chunks <= 0 or indexed is None:
        return False, f"인덱싱 실패 — n_chunks={n_chunks}"
    if pipe._store.size != n_chunks:
        return False, f"store 크기 {pipe._store.size} != 인덱싱 청크 {n_chunks}"
    if not getattr(pipe, "_paper_keywords", None):
        return False, "인덱싱 후 논문 키워드가 남지 않음 — BM25에 영어 키워드가 전달되지 않음"
    from rag_pipeline import _looks_like_bibliography
    leaked_bib = sum(_looks_like_bibliography(c) for c in pipe._store._chunks)
    if leaked_bib:
        return False, f"참고문헌 필터 미배선 — 참고문헌 청크 {leaked_bib}개가 인덱스에 들어감"

    # 5) 검색 -> 압축 -> 답변 사슬
    answer_q = "이 논문의 주요 방법론은 무엇인가?"
    retrieved = pipe._retrieve(answer_q, embedder().embed_query(answer_q))
    if not retrieved:
        return False, "_retrieve가 0건 반환 — 인덱스가 비었거나 검색 미배선"

    _, answer = pipe.generate_answer(answer_q, indexed, "ko")
    if not answer.strip():
        return False, "generate_answer가 빈 답변 반환"
    numbered = [l for l in answer.splitlines() if re.match(r"^\s*\d+[\.\)]\s", l)]
    if numbered:
        return False, f"번호목록 잔존 — 답변 경로에 normalize_bullets 미적용: {numbered[0][:60]!r}"

    return True, (
        f"키워드→검색({len(papers)}건, 무근거 {len(unreliable)}건 제외)→선택→"
        f"인덱싱({n_chunks}청크)→검색({len(retrieved)}건)→답변({len(answer)}자) 전 구간 정상"
    )


# ---------------------------------------------------------------------------
# Streamlit
# ---------------------------------------------------------------------------

def check_streamlit_boot() -> Result:
    """
    서버 기동 + 앱 스크립트가 실제로 오류 없이 실행됐는지 확인한다.

    주의: Streamlit은 app.py가 깨져도 서버 셸은 띄우고 HTTP 200을 돌려준다
    (오류는 브라우저 UI에만 표시된다). HTTP 200만 보면 문법 오류·런타임
    예외를 전부 놓친다 — 실제로 고장 주입에서 그렇게 통과해버렸다.
    그래서 (1) 문법 파싱 (2) 프로세스 출력의 traceback 스캔을 함께 한다.
    """
    import ast
    import threading

    app_path = os.path.join(ROOT, "app.py")
    if not os.path.exists(app_path):
        return False, "app.py 없음"

    # (1) 문법 검사 — 문법 오류는 여기서 확정적으로 걸린다
    with open(app_path, encoding="utf-8") as fh:
        source = fh.read()
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return False, f"app.py 문법 오류: line {exc.lineno}: {exc.msg}"

    exe = shutil.which("streamlit")
    if not exe:
        return False, "streamlit 실행 파일을 찾을 수 없음"

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    proc = subprocess.Popen(
        [exe, "run", "app.py", "--server.headless=true", f"--server.port={port}",
         "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace",
    )
    captured: List[str] = []
    reader = threading.Thread(
        target=lambda: captured.extend(iter(proc.stdout.readline, "")), daemon=True
    )
    reader.start()

    try:
        deadline = time.time() + 90
        status = None
        while time.time() < deadline:
            if proc.poll() is not None:
                return False, (f"프로세스 조기 종료(code={proc.returncode}): "
                               f"{''.join(captured).strip()[-300:]}")
            try:
                status = requests.get(f"http://127.0.0.1:{port}/", timeout=3).status_code
                if status == 200:
                    break
            except requests.RequestException:
                time.sleep(1)
        if status != 200:
            return False, f"90초 내 HTTP 200 응답 없음 (마지막 status={status})"

        health = requests.get(f"http://127.0.0.1:{port}/healthz", timeout=5)
        if health.status_code != 200:
            return False, f"/healthz 응답 {health.status_code}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    # (3) 스크립트를 실제로 실행시켜 미처리 예외가 없는지 본다.
    #     서버는 웹소켓 세션이 붙어야 스크립트를 돌리므로 HTTP 200만으로는
    #     런타임 오류를 못 잡는다. AppTest는 스크립트를 직접 실행한다.
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(app_path, default_timeout=120).run()
    if at.exception:
        exc = at.exception[0]
        return False, f"앱 실행 중 미처리 예외: {getattr(exc, 'value', exc)}"[:200]
    return True, f"문법 정상, 포트 {port} HTTP 200, 스크립트 실행 시 예외 없음"


# ---------------------------------------------------------------------------
# 환경 정보 (판정에 포함하지 않음)
# ---------------------------------------------------------------------------

def environment_info() -> List[str]:
    lines = []
    key = os.environ.get("S2_API_KEY", "")
    lines.append(f"S2_API_KEY: {'설정됨' if key else '없음 (익명 풀 — 429 빈발)'}")
    lines.append(
        f"UNPAYWALL_EMAIL: {'설정됨' if os.environ.get('UNPAYWALL_EMAIL') else '없음 (PDF 확보율 저하)'}"
    )
    lines.append(f"CORE_API_KEY: {'설정됨' if os.environ.get('CORE_API_KEY') else '없음 (선택)'}")
    try:
        r = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": "test", "limit": 1, "fields": "title"},
            headers={"x-api-key": key} if key else {},
            timeout=15,
        )
        lines.append(f"Semantic Scholar 응답: HTTP {r.status_code}"
                     + (" (요청 한도 — 프로젝트 결함 아님)" if r.status_code == 429 else ""))
    except Exception as exc:
        lines.append(f"Semantic Scholar 응답: 연결 실패 ({type(exc).__name__})")

    pdf_dir = os.path.join(ROOT, "pdfs")
    n_pdf = len([f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]) if os.path.isdir(pdf_dir) else 0
    cache_dir = os.path.join(ROOT, "cache", "emb")
    n_emb = len(os.listdir(cache_dir)) if os.path.isdir(cache_dir) else 0
    lines.append(f"캐시: PDF {n_pdf}개, 임베딩 {n_emb}개")
    return lines


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

CHECKS: List[Tuple[str, Callable[[], Result]]] = [
    ("의존성 설치", check_dependencies),
    ("requirements.txt 정확성", check_requirements_file),
    ("Ollama 서비스 응답", check_ollama_service),
    ("EXAONE 모델 응답", check_exaone_model),
    ("임베딩 벡터 생성", check_embedder),
    ("e5 접두사/교차언어 정렬", check_e5_prefixes),
    ("FAISS 검색", check_faiss_search),
    ("BM25 인덱스", check_bm25_index),
    ("RRF 하이브리드 융합", check_rrf_fusion),
    ("PDF 텍스트 추출", check_pdf_extraction),
    ("텍스트 정제", check_text_cleaning),
    ("청킹", check_chunking),
    ("언어 감지", check_language_detection),
    ("키워드 파싱", check_keyword_parsing),
    ("논문 선택 파싱", check_selection_parsing),
    ("불릿 정규화", check_bullet_normalization),
    ("참고문헌 필터", check_bibliography_filter),
    ("약어 보존·약한 일치 판정", check_acronym_and_weak_match),
    ("BM25 영어 키워드 전달", check_bm25_keywords),
    ("정의 질문 개요 논문 우선", check_definition_overview),
    ("답변 문맥에 초록 포함", check_abstract_in_context),
    ("새 검색 의도 우선", check_new_search_intent),
    ("최신 의도 연도 필터", check_recency_filter),
    ("'제일 최신' 요청 처리", check_newest_request),
    ("PDF 실패 시 대체 순서", check_fallback_order),
    ("답변 범위 규칙", check_scope_rule),
    ("한 단어 질문 되묻기", check_vague_query),
    ("프롬프트 규칙 충돌", check_prompt_consistency),
    ("수치 의미 오독 경고", check_metric_misread),
    ("PDF 합자 복원", check_pdf_ligatures),
    ("임베딩 캐시 내용 기준", check_embedding_cache_key),
    ("답변 앞머리 표시 제거", check_answer_echo),
    ("후속 질문 키워드 범위", check_followup_keywords),
    ("네트워크 대기 최소화", check_network_waits),
    ("후보 목록에서 다시 답하기", check_candidate_switch),
    ("단계별 시간 측정", check_stage_timer),
    ("중복 불릿 제거", check_bullet_dedup),
    ("요약 질의 라우팅", check_summary_routing),
    ("메타데이터 불일치 탐지", check_metadata_suspect),
    ("신뢰도 필터", check_reliability_filter),
    ("429 예외 전파", check_ratelimit_propagation),
    ("PDF 폴백", check_pdf_fallback),
    ("전체 파이프라인 E2E", check_end_to_end),
    ("Streamlit 기동", check_streamlit_boot),
]


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "-v"]
    verbose = "-v" in sys.argv
    only = args[0] if args else None
    checks = [(n, f) for n, f in CHECKS if only is None or only in n or only in f.__name__]

    # 여러 체크가 고장을 일부러 주입해서 검증한다(예: PDF 폴백은 실패를 만들어
    # 대체 동작을 확인한다). 그때 프로젝트 코드가 찍는 WARNING이 섞이면 진짜
    # 실패처럼 보인다. 판정은 반환값으로만 하므로 검사 중에는 로그를 끈다.
    # 디버깅이 필요하면 -v 로 다시 켠다.
    if not verbose:
        logging.disable(logging.WARNING)

    print("=" * 78)
    print("ScholarMind 상태 점검")
    print("=" * 78)
    for line in environment_info():
        print(f"  {line}")
    print("-" * 78)

    failures = []
    for i, (name, fn) in enumerate(checks, 1):
        t0 = time.time()
        try:
            passed, detail = fn()
        except Exception as exc:
            # 체크 내부 예외는 무조건 실패로 처리한다 (삼켜서 통과시키지 않는다)
            passed = False
            detail = f"{type(exc).__name__}: {exc} | {traceback.format_exc(limit=1).splitlines()[-1]}"
        dt = time.time() - t0
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {i:2d}. {name} ({dt:.1f}s)")
        print(f"        {detail}")
        if not passed:
            failures.append(name)

    print("-" * 78)
    print(f"결과: {len(checks) - len(failures)}/{len(checks)} 통과")
    if failures:
        print("실패: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
