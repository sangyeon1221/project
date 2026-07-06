# Ragxiv

로컬 CPU 환경에서 동작하는 arXiv 논문 검색·요약 RAG(Retrieval-Augmented Generation) 어시스턴트입니다.
질문을 입력하면 arXiv에서 관련 논문을 자동으로 검색·선택하고, 논문 PDF를 인덱싱한 뒤 그 내용에
근거해 인용과 함께 답변합니다.

- 🎓 4학년 졸업 캡스톤 프로젝트
- 💻 클라우드 API 없이 완전히 로컬(CPU-only)에서 동작
- 📄 arXiv 자동 검색 + 자율 논문 선택 2단계 에이전트

> 이 프로젝트는 이후 [ScholarMind](https://github.com/sangyeon1221/scholarmind)로 발전했습니다.
> ScholarMind는 논문 소스를 Semantic Scholar로 전환하고 신뢰도 필터, 하이브리드 검색(BM25+벡터),
> 다중 소스 PDF 확보, 답변 스트리밍 등을 추가한 버전입니다.

---

## 아키텍처

```
사용자 질문 (한/영)
   │
   ▼
[0단계] 후속질문 판별(Follow-up Detection) ── 새 검색 없이 기존 논문 인덱스 재사용
   │ (새 주제인 경우)
   ▼
[1단계] LLM이 질문을 arXiv 검색 키워드로 정제 → arXiv 검색 (최대 10편, 3초 레이트리밋)
   │
   ▼
[2단계] LLM이 최적 논문 자율 선택 → PDF 다운로드 → 텍스트 추출 → 1,000자 단위 청킹
        → 임베딩 → FAISS 색인
   │
   ▼
[3단계] Top-K 청크 검색 → LLM이 인용 포함 최종 답변 생성
```

## 핵심 기술

| 영역 | 기술 |
|---|---|
| 논문 검색 | arXiv API (`arxiv` 파이썬 라이브러리), 3초 레이트리밋 |
| PDF 처리 | PyMuPDF 텍스트 추출 + 유니코드 노이즈 정제(베트남어 성조 부호, PUA 글리프 등 제거) |
| 임베딩 | `intfloat/multilingual-e5-small` — 한국어 질의 ↔ 영어 논문 본문 교차 언어 검색 |
| 검색 | FAISS `IndexFlatIP` (코사인 유사도 기반 정확 최근접 이웃 검색) |
| 에이전트 흐름 | 자율 2단계 에이전트: (1) 검색어 정제·검색 (2) 논문 선택·심층 RAG |
| 컨텍스트 인식 | 키워드 휴리스틱 + LLM 분류를 결합한 후속 질문(follow-up) 자동 감지 |
| LLM 추론 | Ollama + EXAONE 3.5 2.4B, CPU 전용 로컬 추론 |
| UI | Streamlit |

## 실행 환경

- CPU: Intel i7-1355U (또는 동급) — GPU 불필요
- RAM: 16GB
- [Ollama](https://ollama.com) 설치 후 `ollama pull exaone3.5:2.4b`

## 설치 및 실행

```bash
git clone https://github.com/sangyeon1221/ragxiv.git
cd ragxiv
pip install -r requirements.txt

# Ollama 서버 실행 (별도 터미널)
ollama serve

# 앱 실행
streamlit run app.py
```

## 프로젝트 기여 범위

이 프로젝트는 Claude(Claude Code)를 활용해 코드 생성을 진행했습니다. 본인의 기여는 다음과 같습니다:

- 문제 정의 및 요구사항 설계 (로컬 CPU 환경 제약 하의 RAG 시스템 설계)
- 아키텍처 결정 (임베딩 모델 선정 및 전환(BGE → multilingual-e5-small), 데이터 파이프라인 설계)
- 반복적인 실제 질의 결과 평가 및 문제 진단, 프롬프트 개선
- 발표 자료 및 평가표 작성

## 라이선스

이 저장소에는 논문 PDF나 저작권이 있는 자료를 포함하지 않습니다. 코드는 개인/학술 목적으로 자유롭게 사용 가능합니다.
