"""
WebSearchRetriever: lightweight web retriever for BrowseComp-style tasks.

Uses Wikipedia public API at runtime (no local BM25 index required).
Implements the same interface as WikiBM25Retriever:
  - search(query, top_k) -> list[(title, text, score)]
  - lookup(title) -> text | None
"""

import html
import re
from typing import Dict, List, Optional, Tuple

import requests


class WebSearchRetriever:
    """Online retriever backed by Wikipedia search and extract APIs."""

    def __init__(self, timeout_sec: int = 12):
        self.timeout_sec = int(timeout_sec)
        self.session = requests.Session()
        self.api_url = "https://en.wikipedia.org/w/api.php"
        self._title_cache: Dict[str, str] = {}

    @staticmethod
    def _clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = html.unescape(text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _fetch_extract_by_title(self, title: str) -> str:
        key = title.strip().lower()
        if not key:
            return ""
        if key in self._title_cache:
            return self._title_cache[key]

        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": 1,
            "exintro": 0,
            "redirects": 1,
            "titles": title,
        }
        try:
            resp = self.session.get(self.api_url, params=params, timeout=self.timeout_sec)
            resp.raise_for_status()
            data = resp.json()
            pages = (data.get("query") or {}).get("pages") or {}
            extract_text = ""
            for page in pages.values():
                extract_text = self._clean_text(page.get("extract", ""))
                if extract_text:
                    break
            self._title_cache[key] = extract_text
            return extract_text
        except Exception:
            return ""

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """
        Search the web (Wikipedia API) for relevant pages.
        Returns list of (title, text, score) tuples.
        """
        if not isinstance(query, str) or not query.strip():
            return []

        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": max(1, int(top_k)),
            "srnamespace": 0,
            "srwhat": "text",
        }
        try:
            resp = self.session.get(self.api_url, params=params, timeout=self.timeout_sec)
            resp.raise_for_status()
            data = resp.json()
            hits = (data.get("query") or {}).get("search") or []
        except Exception:
            return []

        out: List[Tuple[str, str, float]] = []
        for rank, hit in enumerate(hits):
            title = str(hit.get("title", "")).strip()
            if not title:
                continue
            text = self._fetch_extract_by_title(title)
            # Rank-based proxy score; keeps interface consistent.
            score = float(max(0.0, 1.0 - 0.1 * rank))
            out.append((title, text, score))
        return out

    def lookup(self, title: str) -> Optional[str]:
        """Look up page text by title."""
        if not isinstance(title, str) or not title.strip():
            return None
        text = self._fetch_extract_by_title(title.strip())
        return text if text else None
