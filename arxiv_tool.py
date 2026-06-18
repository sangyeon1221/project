"""
arxiv_tool.py
-------------
arXiv API wrapper with strict 3-second rate limiting.

All public methods that touch the arXiv network automatically wait
RATE_LIMIT_DELAY seconds *before* each request so the caller never
has to think about throttling.
"""

import os
import time
import logging
from typing import Dict, List, Optional

import arxiv
import requests

logger = logging.getLogger(__name__)

RATE_LIMIT_DELAY: float = 3.0  # seconds — mandatory between every arXiv call
PDF_DOWNLOAD_TIMEOUT: int = 120  # seconds for HTTP PDF download


class ArxivTool:
    """Thin, rate-limited wrapper around the arXiv Python library."""

    def __init__(self, save_dir: str = "./pdfs") -> None:
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        # Built-in client-level delay is a safety net; we also sleep explicitly.
        self._client = arxiv.Client(
            page_size=10,
            delay_seconds=RATE_LIMIT_DELAY,
            num_retries=3,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        max_results: int = 10,
        also_recent: bool = False,
    ) -> List[Dict]:
        """
        Search arXiv for papers matching *query* with optimal throughput.

        The single rate-limit sleep fires once before the network request.
        It is NOT repeated inside the results loop — doing so caused a
        cumulative 30-second freeze per 10-paper batch, triggering HTTP
        ReadTimeouts and "No papers found" crashes.

        Args:
            query:        English search keywords (already refined by LLM).
            max_results:  Maximum number of results to fetch.
            also_recent:  When True, run a second search sorted by
                          SubmittedDate and merge the results (deduped).
                          Useful when relevance sort buries brand-new papers.
                          Default False — behaviour is unchanged unless set.

        Returns:
            List of paper metadata dicts. Each dict has:
            id, title, abstract, authors, published, url, pdf_url.
        """
        logger.info("arXiv search starting: %s", query)
        time.sleep(RATE_LIMIT_DELAY)  # one-time rate-limit before network call

        search_obj = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results: List[Dict] = []
        seen_ids: set = set()
        try:
            for paper in self._client.results(search_obj):
                d = self._paper_to_dict(paper)
                seen_ids.add(d["id"])
                results.append(d)
        except Exception as exc:
            logger.error("arXiv search error: %s", exc)

        # Optional recency pass — catches brand-new papers relevance sort buries
        if also_recent:
            time.sleep(RATE_LIMIT_DELAY)
            recent_obj = arxiv.Search(
                query=query,
                max_results=max_results // 2 or 5,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )
            try:
                for paper in self._client.results(recent_obj):
                    d = self._paper_to_dict(paper)
                    if d["id"] not in seen_ids and len(results) < max_results:
                        seen_ids.add(d["id"])
                        results.append(d)
            except Exception as exc:
                logger.error("arXiv recent search error: %s", exc)

        logger.info("Found %d papers successfully", len(results))
        return results

    def download_pdf(self, paper: Dict) -> Optional[str]:
        """
        Download the PDF for *paper* and return the local file path.

        Uses a direct HTTP download so no extra arxiv API call is made.
        Returns None if the download fails.

        Args:
            paper: Dict produced by :meth:`search` (must contain 'id'
                   and 'pdf_url').
        """
        safe_id = paper["id"].replace("/", "_").replace(".", "_")
        filename = f"{safe_id}.pdf"
        filepath = os.path.join(self.save_dir, filename)

        if os.path.exists(filepath):
            logger.info("PDF cache hit: %s", filepath)
            return filepath

        logger.info("Downloading PDF: %s", paper["id"])
        time.sleep(RATE_LIMIT_DELAY)

        try:
            headers = {"User-Agent": "Ragxiv/1.0 (research assistant)"}
            resp = requests.get(
                paper["pdf_url"],
                headers=headers,
                timeout=PDF_DOWNLOAD_TIMEOUT,
                allow_redirects=True,
            )
            resp.raise_for_status()
            with open(filepath, "wb") as fh:
                fh.write(resp.content)
            logger.info("Saved PDF: %s", filepath)
            return filepath
        except Exception as exc:
            logger.error("PDF download failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _paper_to_dict(paper: arxiv.Result) -> Dict:
        return {
            "id": paper.get_short_id(),
            "title": paper.title,
            "abstract": paper.summary.replace("\n", " ").strip(),
            "authors": [str(a) for a in paper.authors[:5]],
            "published": paper.published.strftime("%Y-%m-%d"),
            "url": paper.entry_id,
            "pdf_url": paper.pdf_url,
        }
