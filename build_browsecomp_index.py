"""
Build a BM25 index over BrowseComp-Plus corpus.

Usage:
    python build_browsecomp_index.py [--index_dir INDEX_DIR]

This builds a local BM25 index from `Tevatron/browsecomp-plus-corpus`,
so BrowseComp experiments can run without online web search.
"""

import argparse
import json
import os
import time
from typing import Any, Dict, Optional

import bm25s
import Stemmer
from datasets import load_dataset


DATA_CACHE_DIR = "data/browsecomp"
DEFAULT_INDEX_DIR = "data/browsecomp_index"


def _pick_text(item: Dict[str, Any]) -> str:
    for k in ("text", "contents", "content", "document", "body"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _pick_docid(item: Dict[str, Any], idx: int) -> str:
    for k in ("docid", "id", "_id", "uid"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return str(idx)


def _pick_url(item: Dict[str, Any]) -> str:
    for k in ("url", "source_url", "source", "link"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _pick_title(item: Dict[str, Any], docid: str, url: str) -> str:
    title = item.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    if url:
        # Keep a stable short display title from URL tail.
        tail = url.rstrip("/").split("/")[-1].strip()
        if tail:
            return tail
    return docid


def build_browsecomp_index(index_dir: str) -> None:
    os.makedirs(index_dir, exist_ok=True)
    start = time.time()
    print("[INFO] Loading Tevatron/browsecomp-plus-corpus...")
    ds = load_dataset(
        "Tevatron/browsecomp-plus-corpus",
        split="train",
        cache_dir=DATA_CACHE_DIR,
    )
    print(f"[INFO] Corpus loaded: {len(ds)} rows")

    titles = []
    corpus_texts = []
    aliases = []
    skipped = 0

    for i, item in enumerate(ds):
        text = _pick_text(item)
        if not text:
            skipped += 1
            continue
        docid = _pick_docid(item, i)
        url = _pick_url(item)
        title = _pick_title(item, docid, url)

        titles.append(title)
        corpus_texts.append(text)
        aliases.append({"docid": docid, "url": url, "title": title})

        if (i + 1) % 200000 == 0:
            print(f"[INFO] Processed {i + 1}/{len(ds)} rows...")

    print(f"[INFO] Kept {len(corpus_texts)} docs (skipped {skipped} empty docs)")

    titles_path = os.path.join(index_dir, "titles.json")
    aliases_path = os.path.join(index_dir, "aliases.json")
    corpus_texts_path = os.path.join(index_dir, "corpus_texts.jsonl")

    with open(titles_path, "w", encoding="utf-8") as f:
        json.dump(titles, f, ensure_ascii=False)
    with open(aliases_path, "w", encoding="utf-8") as f:
        json.dump(aliases, f, ensure_ascii=False)
    with open(corpus_texts_path, "w", encoding="utf-8") as f:
        for text in corpus_texts:
            f.write(json.dumps(text, ensure_ascii=False) + "\n")

    print("[INFO] Tokenizing corpus...")
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
        "dataset": "Tevatron/browsecomp-plus-corpus",
        "num_docs": len(corpus_texts),
        "skipped_empty_docs": skipped,
        "build_time_seconds": time.time() - start,
    }
    with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[DONE] BrowseComp BM25 index built.")
    print(f"  index_dir: {index_dir}")
    print(f"  num_docs:  {len(corpus_texts)}")
    print(f"  elapsed:   {time.time() - start:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 index over BrowseComp-Plus corpus")
    parser.add_argument(
        "--index_dir",
        type=str,
        default=DEFAULT_INDEX_DIR,
        help=f"Directory to save index (default: {DEFAULT_INDEX_DIR})",
    )
    args = parser.parse_args()
    build_browsecomp_index(args.index_dir)


if __name__ == "__main__":
    main()
