"""
semantic_scholar_tool.py
------------------------
Semantic Scholar Academic Graph API wrapper for Ragxiv.

Replaces arxiv_tool.py as the paper search and PDF acquisition layer.
Public interface is identical to ArxivTool so rag_pipeline.py requires
only a one-line import swap.

Rate limiting
-------------
S2 without a key: ~100 requests / 5 min (~3–4 s per request).
All api.semanticscholar.org calls go through _s2_get(), which sleeps
RATE_LIMIT_DELAY seconds before every request.

Non-S2 hosts (arXiv abstract backfill, Unpaywall, CORE, PDF downloads)
go through _http_get(), which carries NO throttle.  arXiv volume is kept
safe by batching (one Atom feed call per search() invocation instead of
one per paper) — not by per-call sleeping.

IMPORTANT: if _backfill_abstracts is ever changed back to a per-paper
loop, a small per-call delay on export.arxiv.org must be reintroduced to
avoid triggering arXiv's burst detection.

Environment variables
---------------------
S2_API_KEY       (optional) — raises S2 rate limit via x-api-key header.
UNPAYWALL_EMAIL  (optional) — enables Unpaywall step in PDF cascade and
                              the pre-filter is_oa check.
CORE_API_KEY     (optional) — enables CORE step in PDF cascade.
"""

import logging
import os
import re
import time
from typing import Dict, List, Optional

import feedparser
import requests

logger = logging.getLogger(__name__)

# S2 호출 간 대기 시간 — 키 유무에 따라 달라진다.
#   무키(익명): 전 세계 공유 풀이라 제한이 매우 빡빡하고 상시 포화 → 보수적으로 3초
#   키 보유:    공식 제한이 "초당 1회, 모든 엔드포인트 합산"이므로 1.1초면 충분
#               (1.0초 정확히 맞추면 네트워크/클럭 지터로 429가 날 수 있어 여유를 둠)
RATE_LIMIT_DELAY: float = 3.0        # seconds — 무키 기본값
RATE_LIMIT_DELAY_KEYED: float = 1.1  # seconds — S2_API_KEY 설정 시
PDF_DOWNLOAD_TIMEOUT: int = 120  # seconds

_S2_BASE = "https://api.semanticscholar.org/graph/v1"
_S2_FIELDS = (
    "paperId,title,abstract,authors,year,publicationDate,url,"
    "venue,publicationVenue,publicationTypes,citationCount,"
    "influentialCitationCount,externalIds,openAccessPdf"
)


class S2RateLimitError(RuntimeError):
    """
    S2 API가 429(Too Many Requests)로 호출을 거부한 상태.

    "검색 결과가 없음"과 반드시 구분해야 한다.  빈 리스트를 반환하면 호출자가
    "이 주제의 논문이 없다"고 오해하고 사용자에게 '질문을 바꿔보라'고 잘못
    안내하게 된다.  실제로는 질문과 무관하게 API 자체가 막힌 상태이므로,
    예외로 전파해서 (1) 정확한 메시지를 보여주고 (2) 남은 검색 시도를 즉시
    중단해 쿼터 낭비를 막는다.

    S2_API_KEY 환경변수가 없으면 전 세계 익명 사용자가 공유하는 풀을 쓰게 되어
    상시 429가 발생할 수 있다.  무료 키 발급:
    https://www.semanticscholar.org/product/api#api-key-form
    """


class SemanticScholarTool:
    """
    Rate-limited wrapper around the Semantic Scholar Academic Graph API.

    Exposes the same public interface as ArxivTool:
        search(query, max_results, also_recent)  -> List[Dict]
        download_pdf(paper)                      -> Optional[str]
        is_pdf_likely_available(paper)           -> bool
    """

    def __init__(self, save_dir: str = "./pdfs") -> None:
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        api_key = os.environ.get("S2_API_KEY", "")
        self._s2_headers: Dict[str, str] = {}
        if api_key:
            self._s2_headers["x-api-key"] = api_key
            self._rate_delay = RATE_LIMIT_DELAY_KEYED
            logger.info(
                "SemanticScholarTool: S2_API_KEY 감지 — 호출 간격 %.1fs (초당 1회 제한 준수)",
                self._rate_delay,
            )
        else:
            self._rate_delay = RATE_LIMIT_DELAY
            logger.warning(
                "SemanticScholarTool: S2_API_KEY 없음 — 익명 공유 풀 사용, 429가 자주 발생할 수 "
                "있습니다. 호출 간격 %.1fs. 키 발급: "
                "https://www.semanticscholar.org/product/api#api-key-form",
                self._rate_delay,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        max_results: int = 10,
        also_recent: bool = False,
        backfill: bool = True,
    ) -> List[Dict]:
        """
        Search Semantic Scholar for papers matching *query*.

        Each phrase from the "|"-separated keyword string is passed as a
        plain-text relevance query.  Do NOT pass arXiv-style quoted phrases
        or boolean OR operators — S2 uses its own relevance ranker and does
        not interpret arXiv query grammar.

        Args:
            query:        Plain-text relevance query (already refined by LLM).
            max_results:  Maximum number of papers to return.
            also_recent:  When True, run a second S2 call sorted by
                          publicationDate and merge results (deduped). Useful
                          when relevance sort buries brand-new papers.
            backfill:     When True (default), backfill missing abstracts from
                          arXiv and sort abstract-first before returning. Set
                          False for intermediate/exploratory calls (e.g. the
                          multi-phrase loop in rag_pipeline.search_papers) so
                          the arXiv batch call only runs once, on the final
                          filtered candidate list, instead of once per call.

        Returns:
            List of paper dicts.  When backfill=True, abstracts are backfilled
            from arXiv in a single batched Atom feed call where needed, and
            papers with non-empty abstracts are sorted first.
        """
        logger.info("S2 search: '%s'", query[:80])
        raw = self._s2_search(query, max_results, sort="relevance")
        results = [self._paper_to_dict(r) for r in raw]
        seen_ids = {p["id"] for p in results}

        if also_recent:
            logger.info("S2 recency pass for: '%s'", query[:80])
            recent_raw = self._s2_search(
                query, max(max_results // 2, 5), sort="date"
            )
            for r in recent_raw:
                d = self._paper_to_dict(r)
                if d["id"] not in seen_ids and len(results) < max_results:
                    seen_ids.add(d["id"])
                    results.append(d)

        if backfill:
            # Backfill missing abstracts via ONE batched arXiv Atom call (H)
            self._backfill_abstracts(results)
            # Papers with abstracts first so LLM selection always sees useful content
            results.sort(key=lambda p: 0 if p.get("abstract") else 1)

        logger.info("S2 search complete: %d papers for '%s'", len(results), query[:60])
        return results

    def backfill_abstracts(self, papers: List[Dict]) -> None:
        """
        Public wrapper around the batched arXiv abstract backfill.

        Intended to be called once, on the final filtered candidate list
        (after reliability + PDF pre-filtering), rather than on every
        intermediate search() call — see the backfill parameter above.
        """
        self._backfill_abstracts(papers)

    def download_pdf(self, paper: Dict) -> Optional[str]:
        """
        Resolve and download a PDF for *paper* via a multi-source cascade.

        The cache is checked first — the entire cascade is skipped on a hit.

        Cascade order (normal case — metadata_suspect is False/absent):
          1. arXiv         — if arxiv_id present:
                             https://arxiv.org/pdf/{arxiv_id}.pdf
          2. openAccessPdf — if pdf_url present in the paper dict
          3. Unpaywall     — if doi present + UNPAYWALL_EMAIL env var set
          4. CORE          — if doi present + CORE_API_KEY env var set

        Cascade order (metadata_suspect is True — Part 2):
          When _paper_to_dict's arXiv-ID/year cross-check flagged this record
          as suspect, the arxiv_id linkage itself is what's in doubt, so it's
          demoted to position 3.  openAccessPdf and Unpaywall are keyed off
          this specific S2 record/its DOI rather than the suspect external
          link, so they're tried first instead:
            1. openAccessPdf  2. Unpaywall  3. arXiv  4. CORE

        Each downloaded file is validated against the %PDF magic number.
        HTML error pages are rejected and the cascade continues.

        Returns local file path on success, None if all sources fail.
        """
        safe_id = paper["id"].replace("/", "_").replace(".", "_")
        filepath = os.path.join(self.save_dir, f"{safe_id}.pdf")

        # Cache check wraps the ENTIRE cascade — no re-download on hit (C)
        if os.path.exists(filepath):
            logger.info("PDF cache hit: %s", filepath)
            return filepath

        normal_sources = [
            ("arXiv",        lambda: self._try_arxiv(paper)),
            ("openAccessPdf", lambda: self._try_open_access(paper)),
            ("Unpaywall",    lambda: self._try_unpaywall(paper)),
            ("CORE",         lambda: self._try_core(paper)),
        ]
        suspect_sources = [
            ("openAccessPdf", lambda: self._try_open_access(paper)),
            ("Unpaywall",    lambda: self._try_unpaywall(paper)),
            ("arXiv",        lambda: self._try_arxiv(paper)),
            ("CORE",         lambda: self._try_core(paper)),
        ]

        if paper.get("metadata_suspect"):
            sources = suspect_sources
            logger.info(
                "PDF cascade for %s: metadata_suspect=True — using reordered "
                "order (openAccessPdf → Unpaywall → arXiv → CORE)",
                paper["id"],
            )
        else:
            sources = normal_sources
            logger.info(
                "PDF cascade for %s: using normal order "
                "(arXiv → openAccessPdf → Unpaywall → CORE)",
                paper["id"],
            )

        for name, attempt in sources:
            logger.info("PDF cascade — trying %s for paper %s", name, paper["id"])
            content = attempt()
            if content is None:
                logger.info("PDF cascade — %s: skipped (no applicable URL/key)", name)
                continue
            if self._is_valid_pdf(content):
                with open(filepath, "wb") as fh:
                    fh.write(content)
                logger.info("PDF saved via %s: %s", name, filepath)
                return filepath
            logger.warning(
                "PDF cascade — %s returned non-PDF content for %s (rejected)",
                name, paper["id"],
            )

        logger.error(
            "PDF cascade exhausted for %s — all sources failed", paper["id"]
        )
        return None

    def is_pdf_likely_available(self, paper: Dict) -> bool:
        """
        Decide without downloading whether a paper has a resolvable PDF.

        Returns True when any of these signals is present:
          - arxiv_id present (arXiv is ~100% open access)
          - pdf_url present  (openAccessPdf.url was set in S2 metadata)
          - Unpaywall reports is_oa == True (only checked when
            UNPAYWALL_EMAIL env var is set)

        Never downloads. Used by the pre-selection PDF availability filter
        in rag_pipeline.
        """
        if paper.get("arxiv_id"):
            return True
        if paper.get("pdf_url"):
            return True
        doi = paper.get("doi")
        email = os.environ.get("UNPAYWALL_EMAIL", "")
        if doi and email:
            return self._unpaywall_is_oa(doi, email)
        return False

    # ------------------------------------------------------------------
    # Internal — S2 API calls
    # ------------------------------------------------------------------

    def _s2_search(
        self, query: str, limit: int, sort: str = "relevance"
    ) -> List[Dict]:
        """Issue one S2 paper/search request. Returns the raw JSON data list."""
        params: Dict = {
            "query": query,
            "limit": min(limit, 100),
            "fields": _S2_FIELDS,
        }
        if sort == "date":
            params["sort"] = "publicationDate"

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                resp = self._s2_get(
                    f"{_S2_BASE}/paper/search",
                    params=params,
                    timeout=30,
                )
                if resp.status_code == 429:
                    backoff = self._rate_delay * (2 ** attempt)
                    logger.warning(
                        "S2 429 (attempt %d/%d) — backing off %.0f s",
                        attempt + 1, max_attempts, backoff,
                    )
                    if attempt == max_attempts - 1:
                        # 재시도를 모두 소진 — 빈 리스트가 아니라 예외로 전파해서
                        # "결과 없음"과 구분하고 남은 검색 시도를 중단시킨다.
                        raise S2RateLimitError(
                            f"S2 API rate limit (429) after {max_attempts} attempts"
                        )
                    time.sleep(backoff)
                    continue
                resp.raise_for_status()
                return resp.json().get("data", [])
            except S2RateLimitError:
                raise
            except requests.exceptions.HTTPError as exc:
                logger.error("S2 HTTP error: %s", exc)
                if attempt == max_attempts - 1:
                    return []
            except Exception as exc:
                logger.error("S2 search error: %s", exc)
                return []
        return []

    # ------------------------------------------------------------------
    # Internal — dict mapping (pure, no network calls)
    # ------------------------------------------------------------------

    @staticmethod
    def _paper_to_dict(raw: Dict) -> Dict:
        """
        Map a raw S2 API result to the Ragxiv paper dict schema.

        Pure mapper — no network calls. Abstract backfill is handled
        separately in _backfill_abstracts() after all papers are collected.

        Existing keys (preserved for downstream compatibility):
            id, title, abstract, authors, published, url, pdf_url

        New keys:
            arxiv_id, doi, venue, pub_types,
            citation_count, influential_citations, metadata_suspect

        metadata_suspect
            S2 occasionally links externalIds.ArXiv to an unrelated record
            sharing the same title (observed: arxiv_id resolved to a 2022
            paper while this record's own year/authors were from an
            unrelated 2021/2014 paper).  Modern arXiv IDs encode their
            submission year in the first 4 digits (YYMM.NNNNN), so we cross
            check that against this record's own year/publicationDate.  A
            >1 year gap flags the record as suspect (1 year tolerance avoids
            false positives from ordinary preprint-to-publication delay).
        """
        paper_id = raw.get("paperId") or ""

        # published: prefer publicationDate ("%Y-%m-%d"); fall back to year;
        # then "n.d.-01-01" ([:4] slice yields "n.d." in the citation card).
        pub_date: str = raw.get("publicationDate") or ""
        if not pub_date:
            year = raw.get("year")
            pub_date = f"{year}-01-01" if year else "n.d.-01-01"

        external_ids = raw.get("externalIds") or {}
        arxiv_id: Optional[str] = external_ids.get("ArXiv")
        doi: Optional[str] = external_ids.get("DOI")

        open_access = raw.get("openAccessPdf") or {}
        pdf_url: Optional[str] = open_access.get("url") or None

        pub_venue_obj = raw.get("publicationVenue") or {}
        venue: str = pub_venue_obj.get("name") or raw.get("venue") or ""

        # arXiv-ID / year cross-check (Part 1)
        metadata_suspect = False
        if arxiv_id:
            m = re.match(r"^(\d{2})(\d{2})\.", arxiv_id)
            if m:
                yy = int(m.group(1))
                implied_year = 2000 + yy  # arXiv's new-style IDs all postdate 2007
                pub_year_str = pub_date[:4]
                if pub_year_str.isdigit():
                    if abs(implied_year - int(pub_year_str)) > 1:
                        metadata_suspect = True
                        logger.warning(
                            "arXiv-ID/year mismatch for paperId=%s title=%r: "
                            "arxiv_id implies %d but S2 year=%s — flagging metadata_suspect",
                            paper_id, raw.get("title"), implied_year, pub_year_str,
                        )

        return {
            # --- existing schema keys ---
            "id":        paper_id,
            "title":     raw.get("title") or "",
            "abstract":  (raw.get("abstract") or "").replace("\n", " ").strip(),
            "authors":   [
                a.get("name", "") for a in (raw.get("authors") or [])[:5]
            ],
            "published": pub_date,
            "url":       (
                raw.get("url")
                or f"https://www.semanticscholar.org/paper/{paper_id}"
            ),
            "pdf_url":   pdf_url,
            # --- new keys ---
            "arxiv_id":              arxiv_id,
            "doi":                   doi,
            "venue":                 venue,
            "pub_types":             raw.get("publicationTypes") or [],
            "citation_count":        raw.get("citationCount") or 0,
            "influential_citations": raw.get("influentialCitationCount") or 0,
            "metadata_suspect":      metadata_suspect,
        }

    # ------------------------------------------------------------------
    # Internal — abstract backfill (H)
    # ------------------------------------------------------------------

    def _backfill_abstracts(self, papers: List[Dict]) -> None:
        """
        Fill empty abstracts in-place via ONE batched arXiv Atom API call.

        arXiv's id_list parameter accepts comma-separated IDs and returns all
        entries in a single Atom feed response.  Batching is the mechanism
        that keeps arXiv traffic safe — no per-call sleep is used here.

        IMPORTANT: if this is ever changed to a per-paper loop, a per-call
        delay on export.arxiv.org must be reintroduced to avoid triggering
        arXiv's burst detection.
        """
        need = [p for p in papers if not p.get("abstract") and p.get("arxiv_id")]
        if not need:
            return

        norm = lambda a: re.sub(r"v\d+$", "", a)   # strip version suffix e.g. "v2"
        id_list = ",".join(norm(p["arxiv_id"]) for p in need)
        url = (
            "https://export.arxiv.org/api/query"
            f"?id_list={id_list}&max_results={len(need)}"
        )
        logger.info("Abstract backfill: fetching %d abstracts from arXiv", len(need))
        try:
            resp = self._http_get(url, timeout=30)  # non-S2 domain — no S2 throttle
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
            by_id: Dict[str, str] = {}
            for entry in feed.entries:
                # entry.id is like "http://arxiv.org/abs/2301.00001v2"
                short = norm(entry.id.rsplit("/abs/", 1)[-1])
                by_id[short] = (entry.summary or "").replace("\n", " ").strip()
            filled = 0
            for p in need:
                key = norm(p["arxiv_id"])
                if by_id.get(key):
                    p["abstract"] = by_id[key]
                    filled += 1
            logger.info("Abstract backfill: %d/%d abstracts filled", filled, len(need))
        except Exception as exc:
            logger.warning(
                "Abstract backfill failed (leaving empty): %s", exc
            )

    # ------------------------------------------------------------------
    # Internal — PDF cascade sources
    # ------------------------------------------------------------------

    def _try_arxiv(self, paper: Dict) -> Optional[bytes]:
        arxiv_id = paper.get("arxiv_id")
        if not arxiv_id:
            return None
        bare = re.sub(r"v\d+$", "", arxiv_id)
        url = f"https://arxiv.org/pdf/{bare}.pdf"
        try:
            resp = self._http_get(
                url,
                headers={"User-Agent": "Ragxiv/1.0 (research assistant)"},
                timeout=PDF_DOWNLOAD_TIMEOUT,
                allow_redirects=True,
            )
            resp.raise_for_status()
            return resp.content
        except Exception as exc:
            logger.warning("arXiv PDF fetch failed for '%s': %s", arxiv_id, exc)
            return None

    def _try_open_access(self, paper: Dict) -> Optional[bytes]:
        url = paper.get("pdf_url")
        if not url:
            return None
        try:
            resp = self._http_get(
                url,
                headers={"User-Agent": "Ragxiv/1.0 (research assistant)"},
                timeout=PDF_DOWNLOAD_TIMEOUT,
                allow_redirects=True,
            )
            resp.raise_for_status()
            return resp.content
        except Exception as exc:
            logger.warning(
                "openAccessPdf fetch failed for '%s': %s", paper["id"], exc
            )
            return None

    def _try_unpaywall(self, paper: Dict) -> Optional[bytes]:
        doi = paper.get("doi")
        email = os.environ.get("UNPAYWALL_EMAIL", "")
        if not doi or not email:
            return None
        try:
            meta = self._http_get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={"email": email},
                timeout=15,
            )
            meta.raise_for_status()
            data = meta.json()
            pdf_url = (data.get("best_oa_location") or {}).get("url_for_pdf")
            if not pdf_url:
                logger.info("Unpaywall: no OA PDF URL for DOI %s", doi)
                return None
            resp = self._http_get(
                pdf_url,
                headers={"User-Agent": "Ragxiv/1.0 (research assistant)"},
                timeout=PDF_DOWNLOAD_TIMEOUT,
                allow_redirects=True,
            )
            resp.raise_for_status()
            return resp.content
        except Exception as exc:
            logger.warning("Unpaywall fetch failed for DOI '%s': %s", doi, exc)
            return None

    def _try_core(self, paper: Dict) -> Optional[bytes]:
        doi = paper.get("doi")
        api_key = os.environ.get("CORE_API_KEY", "")
        if not doi or not api_key:
            return None
        try:
            search_resp = self._http_get(
                "https://api.core.ac.uk/v3/search/works",
                params={"q": f'doi:"{doi}"', "limit": 1},
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=15,
            )
            search_resp.raise_for_status()
            results = search_resp.json().get("results", [])
            if not results:
                logger.info("CORE: no results for DOI %s", doi)
                return None
            pdf_url = (
                results[0].get("downloadUrl") or results[0].get("fullTextLink")
            )
            if not pdf_url:
                logger.info("CORE: no PDF URL found for DOI %s", doi)
                return None
            resp = self._http_get(
                pdf_url,
                headers={"User-Agent": "Ragxiv/1.0 (research assistant)"},
                timeout=PDF_DOWNLOAD_TIMEOUT,
                allow_redirects=True,
            )
            resp.raise_for_status()
            return resp.content
        except Exception as exc:
            logger.warning("CORE fetch failed for DOI '%s': %s", doi, exc)
            return None

    def _unpaywall_is_oa(self, doi: str, email: str) -> bool:
        """Lightweight is_oa check via Unpaywall — no download, for pre-filter."""
        try:
            resp = self._http_get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={"email": email},
                timeout=10,
            )
            resp.raise_for_status()
            return bool(resp.json().get("is_oa", False))
        except Exception as exc:
            logger.warning(
                "Unpaywall is_oa check failed for DOI '%s': %s", doi, exc
            )
            return False

    # ------------------------------------------------------------------
    # Internal — request helpers (E)
    # ------------------------------------------------------------------

    def _s2_get(self, url: str, **kwargs) -> requests.Response:
        """
        Rate-limited GET for api.semanticscholar.org endpoints only.
        Sleeps self._rate_delay seconds before every call
        (키 유무에 따라 3.0s 또는 1.1s — __init__ 참조).
        """
        time.sleep(self._rate_delay)
        return requests.get(url, headers=self._s2_headers, **kwargs)

    def _http_get(self, url: str, **kwargs) -> requests.Response:
        """
        Unthrottled GET for non-S2 hosts: arXiv Atom feed, Unpaywall,
        CORE, and direct PDF downloads.

        arXiv safety is maintained by batching, not throttling — see
        _backfill_abstracts.  Unpaywall and CORE have no per-second caps.
        """
        return requests.get(url, **kwargs)

    # ------------------------------------------------------------------
    # Internal — PDF validation
    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid_pdf(content: bytes) -> bool:
        """Return True if content starts with the PDF magic number (%PDF)."""
        return len(content) > 4 and content[:4] == b"%PDF"
