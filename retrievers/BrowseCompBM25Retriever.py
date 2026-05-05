"""
BrowseCompBM25Retriever: BM25 retriever for BrowseComp-Plus local index.

Expected files in index_dir:
  - titles.json
  - aliases.json
  - corpus_texts.jsonl
  - bm25s index files
"""

import json
import os
from typing import Dict, List, Tuple

import bm25s
import Stemmer


class BrowseCompBM25Retriever:
    def __init__(self, index_dir: str, load_corpus: bool = True):
        self.index_dir = index_dir
        titles_path = os.path.join(index_dir, "titles.json")
        aliases_path = os.path.join(index_dir, "aliases.json")
        if not os.path.exists(titles_path):
            raise FileNotFoundError(
                f"titles.json not found in {index_dir}. Run build_browsecomp_index.py first."
            )
        if not os.path.exists(aliases_path):
            raise FileNotFoundError(
                f"aliases.json not found in {index_dir}. Run build_browsecomp_index.py first."
            )

        with open(titles_path, "r", encoding="utf-8") as f:
            self.titles: List[str] = json.load(f)
        with open(aliases_path, "r", encoding="utf-8") as f:
            self.aliases: List[Dict[str, str]] = json.load(f)

        self.corpus_texts = None
        if load_corpus:
            self._load_corpus_texts()

        self.retriever = bm25s.BM25.load(index_dir, load_corpus=False)
        self.stemmer = Stemmer.Stemmer("english")
        self.alias_to_idx: Dict[str, int] = {}
        for i, item in enumerate(self.aliases):
            for key in ("title", "docid", "url"):
                v = item.get(key)
                if isinstance(v, str) and v.strip():
                    k = v.strip().lower()
                    if k not in self.alias_to_idx:
                        self.alias_to_idx[k] = i

    def _load_corpus_texts(self) -> None:
        corpus_path = os.path.join(self.index_dir, "corpus_texts.jsonl")
        if not os.path.exists(corpus_path):
            self.corpus_texts = None
            return
        self.corpus_texts = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.corpus_texts.append(json.loads(line))

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        query_tokens = bm25s.tokenize(
            [query],
            stopwords="en",
            stemmer=self.stemmer,
            show_progress=False,
        )
        results, scores = self.retriever.retrieve(query_tokens, k=top_k)
        out: List[Tuple[str, str, float]] = []
        for idx_val, score_val in zip(results[0], scores[0]):
            idx = int(idx_val)
            title = self.titles[idx] if 0 <= idx < len(self.titles) else f"doc_{idx}"
            text = ""
            if self.corpus_texts is not None and 0 <= idx < len(self.corpus_texts):
                text = self.corpus_texts[idx]
            out.append((title, text, float(score_val)))
        return out

    def lookup(self, title: str):
        key = title.strip().lower()
        idx = self.alias_to_idx.get(key)
        if idx is not None and self.corpus_texts is not None and 0 <= idx < len(self.corpus_texts):
            return self.corpus_texts[idx]

        # Fallback to bm25 search and exact alias match.
        for cand_title, cand_text, _ in self.search(title, top_k=10):
            if cand_title.strip().lower() == key:
                return cand_text
        return None
