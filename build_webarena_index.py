"""
Build a BM25 index over WebArena's bundled Wikipedia (.zim / Kiwix archive).

Extracts article text from a ZIM file and writes a local BM25 index in the same
format used by BrowseCompBM25Retriever:
  - titles.json
  - aliases.json
  - corpus_texts.jsonl
  - bm25s index files
  - meta.json

Requires: pip install libzim bm25s PyStemmer beautifulsoup4

Usage (server, repo root):
    python build_webarena_index.py --zim_path /path/to/wikipedia_en_all_maxi_2022-05.zim
    # limit size / skip stubs while testing
    python build_webarena_index.py --zim_path ... --max_docs 200000 --min_chars 200
"""

import argparse
import json
import os
import re
import time
from typing import List, Optional

import bm25s
import Stemmer


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_html(html: str) -> str:
    """HTML -> plain text. Prefer BeautifulSoup, fall back to a regex strip."""
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "table", "sup", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
    except Exception:
        text = _TAG_RE.sub(" ", html)
    text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _iter_entries(archive):
    """Yield non-redirect entries across libzim API variants."""
    count = int(getattr(archive, "entry_count", 0) or 0)
    getter = None
    for name in ("_get_entry_by_id", "get_entry_by_id"):
        if hasattr(archive, name):
            getter = getattr(archive, name)
            break
    if getter is None:
        raise RuntimeError("Unsupported libzim version: no get_entry_by_id method.")
    for i in range(count):
        try:
            entry = getter(i)
        except Exception:
            continue
        if getattr(entry, "is_redirect", False):
            continue
        yield entry


def build_webarena_index(zim_path: str, index_dir: str, max_docs: int, min_chars: int) -> None:
    from libzim.reader import Archive

    if not os.path.exists(zim_path):
        raise FileNotFoundError(f"ZIM file not found: {zim_path}")
    os.makedirs(index_dir, exist_ok=True)
    start = time.time()

    print(f"[INFO] Opening ZIM: {zim_path}")
    archive = Archive(zim_path)
    print(f"[INFO] entry_count={getattr(archive, 'entry_count', '?')} "
          f"article_count={getattr(archive, 'article_count', '?')}")

    titles: List[str] = []
    corpus_texts: List[str] = []
    aliases: List[dict] = []
    seen_titles = set()
    skipped = 0
    processed = 0

    for entry in _iter_entries(archive):
        try:
            item = entry.get_item()
            mimetype = str(getattr(item, "mimetype", "") or "")
            if "text/html" not in mimetype:
                continue
            title = str(getattr(entry, "title", "") or "").strip()
            path = str(getattr(entry, "path", "") or "")
            if not title:
                continue
            key = title.lower()
            if key in seen_titles:
                continue
            html = bytes(item.content).decode("utf-8", errors="ignore")
            text = _strip_html(html)
            if len(text) < min_chars:
                skipped += 1
                continue
            seen_titles.add(key)
            titles.append(title)
            corpus_texts.append(text)
            aliases.append({"docid": path or title, "url": "", "title": title})
            processed += 1
            if processed % 100000 == 0:
                print(f"[INFO] Extracted {processed} articles "
                      f"(skipped {skipped} short) elapsed={time.time() - start:.0f}s")
            if max_docs and processed >= max_docs:
                print(f"[INFO] Reached max_docs={max_docs}, stopping extraction.")
                break
        except Exception:
            skipped += 1
            continue

    if not corpus_texts:
        raise ValueError(
            "No articles extracted. Check the ZIM file and that 'libzim' is installed correctly."
        )
    print(f"[INFO] Kept {len(corpus_texts)} articles (skipped {skipped})")

    with open(os.path.join(index_dir, "titles.json"), "w", encoding="utf-8") as f:
        json.dump(titles, f, ensure_ascii=False)
    with open(os.path.join(index_dir, "aliases.json"), "w", encoding="utf-8") as f:
        json.dump(aliases, f, ensure_ascii=False)
    with open(os.path.join(index_dir, "corpus_texts.jsonl"), "w", encoding="utf-8") as f:
        for text in corpus_texts:
            f.write(json.dumps(text, ensure_ascii=False) + "\n")

    print("[INFO] Tokenizing corpus (this can take a while for full Wikipedia)...")
    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(
        corpus_texts,
        stopwords="en",
        stemmer=stemmer,
        show_progress=True,
    )

    print("[INFO] Building BM25 index...")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=True)
    retriever.save(index_dir)

    meta = {
        "source": "webarena_wikipedia_zim",
        "zim_path": zim_path,
        "num_docs": len(corpus_texts),
        "skipped": skipped,
        "min_chars": min_chars,
        "max_docs": max_docs,
        "build_time_seconds": time.time() - start,
    }
    with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[DONE] WebArena Wikipedia BM25 index built.")
    print(f"  index_dir: {index_dir}")
    print(f"  num_docs:  {len(corpus_texts)}")
    print(f"  elapsed:   {time.time() - start:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 index from WebArena Wikipedia .zim")
    parser.add_argument("--zim_path", type=str, required=True, help="Path to the WebArena Wikipedia .zim file.")
    parser.add_argument("--index_dir", type=str, default="data/webarena_index",
                        help="Directory to save the index (default: data/webarena_index).")
    parser.add_argument("--max_docs", type=int, default=0,
                        help="Max articles to index (0 = all). Use a small value to test first.")
    parser.add_argument("--min_chars", type=int, default=200,
                        help="Skip articles shorter than this many characters (default: 200).")
    args = parser.parse_args()
    build_webarena_index(args.zim_path, args.index_dir, args.max_docs, args.min_chars)


if __name__ == "__main__":
    main()
