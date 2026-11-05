"""
eval/verify_healthcheck.py
--------------------------
healthcheck.py의 각 체크가 "실제로 고장을 잡아내는지" 검증한다.

방법: 체크가 검사하는 대상을 고의로 망가뜨린 뒤 그 체크를 호출해서
FAIL이 나오는지 확인한다. 고장을 냈는데 PASS가 나오면 그 체크는 무용지물이다.

실행:
    python eval/verify_healthcheck.py
"""

import importlib.util
import os
import shutil
import sys
import tempfile
import textwrap
from contextlib import contextmanager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location(
    "healthcheck", os.path.join(os.path.dirname(os.path.abspath(__file__)), "healthcheck.py")
)
hc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hc)

results = []


def expect_fail(label, check_fn):
    """고장 주입 상태에서 check_fn이 FAIL을 반환해야 한다."""
    try:
        passed, detail = check_fn()
    except Exception as exc:
        passed, detail = False, f"{type(exc).__name__}: {exc}"
    caught = not passed
    results.append((label, caught, detail))
    print(f"{'[OK]  ' if caught else '[MISS]'} {label}")
    print(f"        -> {detail[:150]}")


def expect_pass(label, check_fn):
    """정상 상태에서 PASS를 반환해야 한다 (항상-실패 체크가 아님을 증명)."""
    try:
        passed, detail = check_fn()
    except Exception as exc:
        passed, detail = False, f"{type(exc).__name__}: {exc}"
    results.append((label, passed, detail))
    print(f"{'[OK]  ' if passed else '[MISS]'} {label}")
    print(f"        -> {detail[:150]}")


@contextmanager
def patched(obj, attr, value):
    old = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, old)


class BlockImport:
    """지정한 모듈의 import를 차단하는 meta path finder."""

    def __init__(self, name):
        self.name = name

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.name:
            raise ImportError(f"blocked: {fullname}")
        return None


print("=" * 78)
print("고장 주입 검증 — 각 체크가 실패를 실제로 잡아내는가")
print("=" * 78)

# 1. 의존성
blocker = BlockImport("rank_bm25")
sys.meta_path.insert(0, blocker)
saved = sys.modules.pop("rank_bm25", None)
expect_fail("의존성: rank_bm25 import 차단", hc.check_dependencies)
sys.meta_path.remove(blocker)
if saved:
    sys.modules["rank_bm25"] = saved

# 2. requirements.txt — 현재 실제로 실패 중이므로, 올바른 파일을 주면
#    PASS가 나오는지(항상-실패 체크가 아닌지) 확인한다.
req = os.path.join(ROOT, "requirements.txt")
backup = req + ".bak"
shutil.copy(req, backup)
try:
    with open(req, "w", encoding="utf-8") as fh:
        fh.write("streamlit>=1.28.0\nsentence-transformers>=2.7.0\nfaiss-cpu>=1.7.4\n"
                 "PyMuPDF>=1.23.0\nrequests>=2.31.0\nnumpy>=1.24.0\n"
                 "rank-bm25>=0.2.2\nfeedparser>=6.0.0\n")
    expect_pass("requirements: 올바른 파일이면 통과(항상-실패 아님)", hc.check_requirements_file)
finally:
    shutil.move(backup, req)

# 3. Ollama 서비스 — 죽은 포트로 향하게 한다
import llm_client
with patched(llm_client, "OLLAMA_BASE_URL", "http://127.0.0.1:1"):
    expect_fail("Ollama: 죽은 포트", hc.check_ollama_service)

# 4. EXAONE 모델 — 없는 모델명
with patched(llm_client, "MODEL_NAME", "no-such-model:999b"):
    expect_fail("EXAONE: 존재하지 않는 모델명", hc.check_exaone_model)

# 5. 임베딩 — 차원이 틀린 벡터를 내놓게 한다
import numpy as np


class BadEmbedder:
    def embed_documents(self, texts):
        return np.ones((len(texts), 128), dtype=np.float32)

    def embed_query(self, q):
        return np.ones(128, dtype=np.float32)


with patched(hc, "embedder", lambda: BadEmbedder()):
    expect_fail("임베딩: 차원 불일치(128 != 384)", hc.check_embedder)

# 5b. 정규화 안 된 벡터
class UnnormEmbedder:
    def embed_documents(self, texts):
        return np.full((len(texts), 384), 5.0, dtype=np.float32)

    def embed_query(self, q):
        return np.ones(384, dtype=np.float32)


with patched(hc, "embedder", lambda: UnnormEmbedder()):
    expect_fail("임베딩: L2 정규화 누락", hc.check_embedder)

# 6. e5 접두사 — 잘못된 값
import embedder as emb_mod
with patched(emb_mod, "_QUERY_PREFIX", "wrong: "):
    expect_fail("e5: 접두사 오설정", hc.check_e5_prefixes)

# 7. FAISS 검색 — 빈 결과
import vector_store
with patched(vector_store.VectorStore, "search", lambda self, *a, **k: []):
    expect_fail("FAISS: 검색 결과 0건", hc.check_faiss_search)

# 7b. 구조가 깨진 결과
with patched(vector_store.VectorStore, "search",
             lambda self, *a, **k: [{"chunk": "", "metadata": {}, "score": 1.0}]):
    expect_fail("FAISS: 빈 청크 반환", hc.check_faiss_search)

# 8. BM25 — 인덱스를 만들지 않게 한다
_orig_add = vector_store.VectorStore.add_chunks


def add_without_bm25(self, chunks, embeddings, metadata):
    _orig_add(self, chunks, embeddings, metadata)
    self._bm25 = None


with patched(vector_store.VectorStore, "add_chunks", add_without_bm25):
    expect_fail("BM25: 인덱스 미구축", hc.check_bm25_index)

# 9. RRF — query_text를 무시해 어휘 순위가 융합되지 않게 한다
_orig_search = vector_store.VectorStore.search


def dense_only(self, query_vector, query_text="", top_k=5, candidate_k=20):
    return _orig_search(self, query_vector, query_text="", top_k=top_k, candidate_k=candidate_k)


with patched(vector_store.VectorStore, "search", dense_only):
    expect_fail("RRF: 어휘 순위 미융합(dense only)", hc.check_rrf_fusion)

# 10. PDF 추출 — 빈 문자열
import pdf_extractor
with patched(pdf_extractor, "extract_text", lambda p: ""):
    expect_fail("PDF: 추출 결과 빈 문자열", hc.check_pdf_extraction)

# 11. 텍스트 정제 — 제어문자를 남기게 한다
with patched(pdf_extractor, "clean_llm_output", lambda t: t):
    expect_fail("정제: 제어문자 미제거", hc.check_text_cleaning)

# 12. 청킹 — 분할하지 않게 한다
import rag_pipeline
with patched(rag_pipeline, "split_text", lambda t, **k: [t]):
    expect_fail("청킹: 분할 안 됨", hc.check_chunking)

# 13. 언어 감지 — 항상 en
with patched(llm_client.LLMClient, "detect_language", staticmethod(lambda t: "en")):
    expect_fail("언어감지: 한국어를 en으로 오판", hc.check_language_detection)

# 14. 키워드 파싱 — 공백 join (원래 있던 실제 버그 재현)
with patched(llm_client.LLMClient, "_parse_keywords",
             staticmethod(lambda raw, fb: " ".join(p.strip() for p in raw.split("|")))):
    expect_fail("키워드: 구분자 소실(공백 join)", hc.check_keyword_parsing)

# 15. 선택 파싱 — 대괄호 마커만 인식 (원래 있던 실제 버그 재현)
import re as _re
with patched(llm_client.LLMClient, "_parse_selection",
             staticmethod(lambda raw, n: (
                 int(m.group(1)) if (m := _re.search(r"\[번호\]:\s*(\d+)", raw)) else 0, ""))):
    expect_fail("선택파싱: 느슨한 마커 미인식", hc.check_selection_parsing)

# 16. 불릿 정규화 — 항등 함수
with patched(llm_client, "normalize_bullets", lambda t: t):
    expect_fail("불릿: 번호목록 그대로 통과", hc.check_bullet_normalization)

# 16b. 참고문헌 필터 — 판별 무력화
with patched(rag_pipeline, "_looks_like_bibliography", lambda c: False):
    expect_fail("참고문헌: 판별 무력화", hc.check_bibliography_filter)

# 16c. 약어 보존 — refine_query에서 호출이 빠짐 (배선 끊김)
with patched(rag_pipeline, "_keep_acronyms", lambda q, kw: kw):
    expect_fail("약어: 보존 호출 누락", hc.check_acronym_and_weak_match)

# 16d. 약한 일치 판정 — 항상 False
with patched(rag_pipeline.RAGPipeline, "is_weak_match", staticmethod(lambda r: False)):
    expect_fail("약한 일치: 신호 무시", hc.check_acronym_and_weak_match)

# 16e. BM25 키워드 — 검색 단계가 사용자 질문만 넘김 (원래 상태)
with patched(rag_pipeline.RAGPipeline, "_bm25_text", lambda self, q: q):
    expect_fail("BM25: 영어 키워드 미전달", hc.check_bm25_keywords)

# 16f. 정의 질문 — survey 검색어 누락 / 개요 표시 누락
with patched(rag_pipeline, "_with_survey_phrase", lambda kw: kw):
    expect_fail("정의 질문: survey 검색어 누락", hc.check_definition_overview)
with patched(llm_client, "_OVERVIEW_HINT_KO", ""):
    expect_fail("정의 질문: 선택 안내 누락", hc.check_definition_overview)

# 16g. 초록 — 문맥 구성에서 초록이 빠짐
with patched(llm_client, "_build_context",
             lambda chunks, paper, lang: "\n\n---\n\n".join(c["chunk"] for c in chunks)):
    expect_fail("초록: 답변 문맥에서 누락", hc.check_abstract_in_context)

# 16h. 새 검색 의도 — 판정 무력화(원래 상태: 후속 질문 분류기로 넘어감)
with patched(rag_pipeline, "_is_new_search_intent", lambda q: False):
    expect_fail("새 검색: 의도 무시", hc.check_new_search_intent)

# 16i. 새 검색 의도 — "이 논문에서 ~ 찾아줘"까지 새 검색으로 처리(과잉 판정)
with patched(rag_pipeline, "_is_new_search_intent",
             lambda q: bool(rag_pipeline._NEW_SEARCH_INTENT.search(q))):
    expect_fail("새 검색: 현재 논문 지칭 무시", hc.check_new_search_intent)

# 16j. 최신 의도 — 연도 필터 미생성 / 도구가 year를 요청에 안 넣음 / 0건 재검색 누락
with patched(rag_pipeline, "_recency_year", lambda q: None):
    expect_fail("최신: 연도 필터 미생성", hc.check_recency_filter)

import semantic_scholar_tool as s2
_orig_s2_search = s2.SemanticScholarTool._s2_search


def s2_search_ignoring_year(self, query, limit, year=None):
    return _orig_s2_search(self, query, limit, year=None)


with patched(s2.SemanticScholarTool, "_s2_search", s2_search_ignoring_year):
    expect_fail("최신: S2 요청에 year 누락", hc.check_recency_filter)

def search_without_year_retry(self, keywords, max_results=10):
    # 연도 필터는 전달하지만, 결과가 0건이면 그대로 0건을 돌려주는 (재검색 없는) 상태
    year = getattr(self, "search_year", None)
    first = keywords.split("|")[0].strip()
    return self._s2.search(first, max_results=max_results, backfill=False, year=year), keywords


with patched(rag_pipeline.RAGPipeline, "search_papers", search_without_year_retry):
    expect_fail("최신: 0건 시 필터 해제 재검색 누락", hc.check_recency_filter)

# 16j-2. '제일 최신' — 시기 표현 제거 안 함 / 날짜 우선 미감지 / 완화 단계 건너뜀 / 날짜 미표시
with patched(rag_pipeline, "_strip_recency_words", lambda kw: kw):
    expect_fail("제일 최신: 검색어의 시기 표현 유지", hc.check_newest_request)
with patched(rag_pipeline, "_wants_newest", lambda q: False):
    expect_fail("제일 최신: 날짜 우선 미감지", hc.check_newest_request)
with patched(rag_pipeline, "_relax_year", lambda y: None):
    expect_fail("제일 최신: 올해 0건이면 바로 제한 해제", hc.check_newest_request)
_orig_fmt = llm_client.LLMClient._format_papers_for_selection
with patched(llm_client.LLMClient, "_format_papers_for_selection",
             staticmethod(lambda papers, overview_label="", show_date=False:
                          _orig_fmt(papers, overview_label=overview_label))):
    expect_fail("제일 최신: 선택 후보에 날짜 미표시", hc.check_newest_request)

# 16j-3. PDF 대체 순서 — 유사도 정렬을 안 함(원래 상태: 목록 순서)
with patched(rag_pipeline.RAGPipeline, "_rank_by_query", lambda self, q, papers: papers):
    expect_fail("PDF 대체: 목록 순서로 시도", hc.check_fallback_order)

# 16k. 범위 규칙 — 프롬프트에서 빠짐
with patched(llm_client, "_RAG_PROMPT_KO",
             _re.sub(r"\n- 질문이 분야 전체.*?\n", "\n", llm_client._RAG_PROMPT_KO)):
    expect_fail("범위 규칙: 한국어 프롬프트 누락", hc.check_scope_rule)

# 16k-2. 되묻기 — 판정 무력화 / 대문자 약어까지 되물음
with patched(rag_pipeline, "_is_vague_query", lambda q: False):
    expect_fail("되묻기: 한 단어 질문 그대로 검색", hc.check_vague_query)
with patched(rag_pipeline, "_is_vague_query", lambda q: " " not in q.strip()):
    expect_fail("되묻기: 구체적인 약어까지 되물음", hc.check_vague_query)

# 16k-3. 프롬프트 충돌 — 예전 출력 예시 복원 / 수치 규칙 누락 / 선택 안내 동시 적용
with patched(llm_client, "_RAG_PROMPT_KO", llm_client._RAG_PROMPT_KO.replace(
        "- C 데이터셋에서 D 지표 E를 기록했습니다.",
        "- 실험 결과 C 지표에서 기존 방법 대비 D만큼 향상되었습니다.")):
    expect_fail("프롬프트: 예시가 '대비 향상'을 유도", hc.check_prompt_consistency)
with patched(llm_client, "_RAG_PROMPT_EN",
             _re.sub(r"\n- Report numbers.*?\n", "\n", llm_client._RAG_PROMPT_EN)):
    expect_fail("프롬프트: 영어 수치 규칙 누락", hc.check_prompt_consistency)
_orig_select = llm_client.LLMClient.select_best_paper


def select_both_hints(self, query, papers, prefer_overview=False, prefer_recent=False):
    # 개요 안내를 최신 안내와 함께 붙이는(우선순위 처리가 빠진) 상태
    label = "[개요 논문]"
    hint = llm_client._OVERVIEW_HINT_KO + (llm_client._RECENT_HINT_KO if prefer_recent else "")
    block = self._format_papers_for_selection(papers, overview_label=label, show_date=prefer_recent)
    self._generate_sync(llm_client._SELECT_PROMPT_KO.format(
        query=query, papers_block=block, overview_hint=hint))
    return papers[0], "x"


with patched(llm_client.LLMClient, "select_best_paper", select_both_hints):
    expect_fail("프롬프트: 개요·최신 안내 동시 적용", hc.check_prompt_consistency)

# 16k-4. 수치 오독 — 검사 무력화 / 원문 대조 없이 "향상 X%"면 무조건 경고 / 근거 미보관
with patched(rag_pipeline, "_metric_misreads", lambda a, c: []):
    expect_fail("수치: 오독 검사 무력화", hc.check_metric_misread)
with patched(rag_pipeline, "_metric_misreads",
             lambda a, c: [("x", "")] if _re.search(r"%\s*(?:향상|감소)|대비", a) else []):
    expect_fail("수치: 원문 대조 없이 경고", hc.check_metric_misread)
with patched(rag_pipeline.RAGPipeline, "_remember_context", lambda self, chunks, paper: None):
    expect_fail("수치: 스트리밍 근거 미보관", hc.check_metric_misread)
with patched(rag_pipeline, "fix_decimal_colons", lambda t: t):
    expect_fail("수치: '21:2%' 소수점 복원 누락", hc.check_metric_misread)

# 16k-5. 정리 작업으로 고친 것들 — 원래 상태로 되돌리면 잡히는가
import pdf_extractor as _pe
with patched(_pe, "_expand_compat_letters", lambda t: t):
    expect_fail("합자: 공백으로 지워짐(원래 상태)", hc.check_pdf_ligatures)


class _ConstHash:
    @staticmethod
    def sha1(data):
        import hashlib as _h
        return _h.sha1(b"same-key-regardless-of-content")


with patched(rag_pipeline, "hashlib", _ConstHash):
    expect_fail("임베딩 캐시: 내용과 무관한 키(원래 상태)", hc.check_embedding_cache_key)


def _old_thinking_parser(text):
    # 예전 _parse_thinking_answer: 어디든 "답변:"이 있으면 그 앞을 버림
    for marker in ("답변:", "Answer:"):
        if marker in (text or ""):
            return text.partition(marker)[2].strip()
    return text


with patched(llm_client, "strip_answer_echo", _old_thinking_parser):
    expect_fail("답변 표시: 본문의 '답변:'에서 잘림(원래 상태)", hc.check_answer_echo)
with patched(llm_client, "strip_answer_echo", lambda t: t):
    expect_fail("답변 표시: 앞머리 표시 미제거", hc.check_answer_echo)

with patched(rag_pipeline, "_FOLLOWUP_KW_KO", rag_pipeline._FOLLOWUP_KW_KO | {"더 알려", "계속"}):
    expect_fail("후속 질문: 애매한 키워드로 즉시 판정(원래 상태)", hc.check_followup_keywords)

import time as _time
_orig_s2_get = s2.SemanticScholarTool._s2_get


def _always_sleep(self, url, **kw):
    _time.sleep(self._rate_delay)
    return _orig_s2_get(self, url, **kw)


with patched(s2.SemanticScholarTool, "_s2_get", _always_sleep):
    expect_fail("S2 대기: 매 호출 전 무조건 대기(원래 상태)", hc.check_network_waits)


def _sequential_prefilter(self, papers):
    return [p for p in papers if self._s2.is_pdf_likely_available(p)] or papers


with patched(rag_pipeline.RAGPipeline, "_pdf_prefilter", _sequential_prefilter):
    expect_fail("PDF 사전 확인: 차례로 실행(원래 상태)", hc.check_network_waits)

# 16k-6. 후보 목록에서 다시 답하기 — 실패 시 복구 안 함 / 항상 최신순 / TLDR 누락 / 정의 판정 원래 상태
def _switch_no_restore(self, paper):
    return self.index_paper(paper)


with patched(rag_pipeline.RAGPipeline, "switch_paper", _switch_no_restore):
    expect_fail("다시 답하기: PDF 실패 시 원래 논문 유실", hc.check_candidate_switch)
with patched(rag_pipeline.RAGPipeline, "candidates_for_display",
             lambda self, papers: sorted(papers, key=lambda p: p.get("published") or "", reverse=True)):
    expect_fail("다시 답하기: 평소에도 최신순으로 바꿈", hc.check_candidate_switch)


_to_dict_before = s2.SemanticScholarTool._paper_to_dict


def _to_dict_without_tldr(raw):
    d = _to_dict_before(raw)
    d.pop("tldr", None)
    return d


with patched(s2.SemanticScholarTool, "_paper_to_dict", staticmethod(_to_dict_without_tldr)):
    expect_fail("다시 답하기: TLDR 누락", hc.check_candidate_switch)


def _old_definition_intent(query):
    q = query.strip()
    if not rag_pipeline._DEFINITION_INTENT.search(q):
        return False
    return len(q) <= 16 if llm_client.LLMClient.detect_language(q) == "ko" else len(q.split()) <= 6


with patched(rag_pipeline, "_is_definition_intent", _old_definition_intent):
    expect_fail("정의 질문: 전체 길이로 판정(원래 상태)", hc.check_definition_overview)
with patched(rag_pipeline, "_is_single_term", lambda q: True):
    expect_fail("정의 질문: 여러 단어 용어에도 survey 추가", hc.check_definition_overview)
with patched(llm_client, "_OVERVIEW_TITLE", _re.compile(r"\bsurvey\b|\breview\b", _re.I)):
    expect_fail("정의 질문: 'Survey' 단어만으로 개요 논문 표시(원래 상태)", hc.check_definition_overview)

# 16k-7. 시간 측정 — 주소를 localhost로 되돌림(원래 상태) / 단계가 누적 시간으로 기록됨
with patched(llm_client, "OLLAMA_BASE_URL", "http://localhost:11434"):
    expect_fail("시간 측정: Ollama 주소 localhost(원래 상태)", hc.check_stage_timer)
import timing as _timing
_orig_lap = _timing.StageTimer.lap


def _lap_cumulative(self, name):
    # 직전 단계부터가 아니라 처음부터 잰 시간을 기록하는 실수
    import time as _t
    self.laps.append((name, _t.perf_counter() - self.start))
    self._last = _t.perf_counter()


with patched(_timing.StageTimer, "lap", _lap_cumulative):
    expect_fail("시간 측정: 단계별이 아닌 누적 시간 기록", hc.check_stage_timer)

# 16l. 중복 불릿 — 제거 안 함 / 임베딩만 보고 지움(오탐) / generate_answer 배선 끊김
with patched(rag_pipeline.RAGPipeline, "dedup_bullets", lambda self, t, threshold=0.97: t):
    expect_fail("중복 불릿: 제거 안 함", hc.check_bullet_dedup)

with patched(rag_pipeline, "_same_words", lambda a, b: True):
    expect_fail("중복 불릿: 단어 확인 없이 임베딩만으로 지움", hc.check_bullet_dedup)

_orig_generate = rag_pipeline.RAGPipeline.generate_answer


def generate_without_dedup(self, query, paper, lang):
    # dedup_bullets 자체는 멀쩡하고, generate_answer 안에서만 호출이 빠진 상태
    self.dedup_bullets = lambda t, threshold=0.97: t
    try:
        return _orig_generate(self, query, paper, lang)
    finally:
        del self.dedup_bullets


with patched(rag_pipeline.RAGPipeline, "generate_answer", generate_without_dedup):
    expect_fail("중복 불릿: generate_answer 배선 끊김", hc.check_bullet_dedup)

# 17. 요약 라우팅 — 의도 감지 무력화
with patched(rag_pipeline, "_is_summary_intent", lambda q: False):
    expect_fail("요약라우팅: 의도 미감지", hc.check_summary_routing)

# 18. 메타데이터 탐지 — 플래그를 절대 세우지 않게 한다
import semantic_scholar_tool as s2
_orig_to_dict = s2.SemanticScholarTool._paper_to_dict


def no_flag(raw):
    d = _orig_to_dict(raw)
    d["metadata_suspect"] = False
    return d


with patched(s2.SemanticScholarTool, "_paper_to_dict", staticmethod(no_flag)):
    expect_fail("메타데이터: 불일치 탐지 무력화", hc.check_metadata_suspect)

# 19. 신뢰도 필터 — 전부 통과시키게 한다
with patched(rag_pipeline.RAGPipeline, "_passes_reliability", lambda self, p, m: True):
    expect_fail("신뢰도: STRICT가 무근거 논문 통과", hc.check_reliability_filter)

# 20. 429 전파 — 예외를 삼키고 빈 리스트 반환 (원래 있던 실제 버그 재현)
with patched(rag_pipeline.RAGPipeline, "search_papers",
             lambda self, kw, max_results=10: ([], kw)):
    expect_fail("429: 예외를 삼키고 빈 결과 반환", hc.check_ratelimit_propagation)

# 21. PDF 폴백 — 대체하지 않고 바로 포기
with patched(rag_pipeline.RAGPipeline, "index_with_fallback",
             lambda self, papers, selected, max_attempts=3: (
                 (selected, 100, False) if self.index_paper(selected) else (None, 0, False))):
    expect_fail("PDF폴백: 대체 후보 시도 안 함", hc.check_pdf_fallback)

# 21b. E2E — 배선을 끊었을 때 잡아내는지 (이 체크의 존재 이유)
#      각 케이스는 "함수는 멀쩡한데 호출되지 않는" 상황을 흉내낸다.
with patched(rag_pipeline.RAGPipeline, "_reliability_filter",
             lambda self, papers: papers):
    expect_fail("E2E: 신뢰도 필터 호출이 빠짐", hc.check_end_to_end)

with patched(vector_store.VectorStore, "add_chunks", lambda self, c, e, m: None):
    expect_fail("E2E: 인덱싱 결과가 store에 안 들어감", hc.check_end_to_end)

with patched(llm_client, "normalize_bullets", lambda t: t):
    expect_fail("E2E: 답변 경로에 불릿 정규화 미적용", hc.check_end_to_end)

with patched(rag_pipeline.RAGPipeline, "_retrieve", lambda self, q, v: []):
    expect_fail("E2E: 검색 단계가 결과를 못 냄", hc.check_end_to_end)

# 인덱싱에서 참고문헌 필터 호출이 빠진 상황: 판별 함수는 그대로 두고
# index_paper가 쓰는 이름만 무력화해야 "배선 끊김"이 된다. index_paper는
# 모듈 전역 이름으로 호출하므로, 원본을 따로 보관해 E2E의 판정에는 원본을 쓴다.
_orig_bib = rag_pipeline._looks_like_bibliography
_orig_index = rag_pipeline.RAGPipeline.index_paper


def index_without_bib_filter(self, paper):
    rag_pipeline._looks_like_bibliography = lambda c: False
    try:
        return _orig_index(self, paper)
    finally:
        rag_pipeline._looks_like_bibliography = _orig_bib


with patched(rag_pipeline.RAGPipeline, "index_paper", index_without_bib_filter):
    expect_fail("E2E: 인덱싱에서 참고문헌 필터 누락", hc.check_end_to_end)


def index_forgetting_keywords(self, paper):
    n = _orig_index(self, paper)
    self._paper_keywords = None
    return n


with patched(rag_pipeline.RAGPipeline, "index_paper", index_forgetting_keywords):
    expect_fail("E2E: 인덱싱 후 논문 키워드 유실", hc.check_end_to_end)

# 22. Streamlit — 깨진 app.py를 가리키게 한다
tmpdir = tempfile.mkdtemp()
try:
    with open(os.path.join(tmpdir, "app.py"), "w", encoding="utf-8") as fh:
        fh.write("this is not valid python !!!(\n")
    with patched(hc, "ROOT", tmpdir):
        expect_fail("Streamlit: 문법 오류 app.py", hc.check_streamlit_boot)

    # 런타임 오류(임포트 실패)도 잡는지 — 더 어려운 케이스
    with open(os.path.join(tmpdir, "app.py"), "w", encoding="utf-8") as fh:
        fh.write(textwrap.dedent("""
            import streamlit as st
            raise RuntimeError("boom at runtime")
        """))
    with patched(hc, "ROOT", tmpdir):
        expect_fail("Streamlit: 런타임 예외 app.py", hc.check_streamlit_boot)
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

print()
print("=" * 78)
caught = sum(1 for _, ok, _ in results if ok)
print(f"결과: {caught}/{len(results)} — 고장을 정상적으로 검출")
missed = [label for label, ok, _ in results if not ok]
if missed:
    print("검출 실패(이 체크들은 고장을 못 잡음):")
    for m in missed:
        print(f"  - {m}")
sys.exit(1 if missed else 0)
