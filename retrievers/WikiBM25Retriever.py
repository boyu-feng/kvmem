"""
WikiBM25Retriever: BM25 retriever backed by a large pre-built Wikipedia index.

Uses bm25s for fast retrieval over 5.2M+ Wikipedia articles.
The index must be pre-built using build_wiki_index.py.
"""

import os
import json
import time
import bm25s
import Stemmer


DEFAULT_INDEX_DIR = "/apdcephfs_szgm/share_303492287/ryanylsun/Projects/ReAct/data/wiki_index"


class WikiBM25Retriever:
    """
    BM25 retriever over a large Wikipedia corpus.

    Loads a pre-built bm25s index and title metadata from disk.
    Supports search (BM25 ranking) and lookup (exact title match).
    """

    def __init__(self, index_dir=DEFAULT_INDEX_DIR, load_corpus=True):
        """
        Args:
            index_dir: Path to the directory containing the bm25s index, titles.json, 
                       and corpus_texts.jsonl
            load_corpus: Whether to load corpus texts into memory (needed for returning
                         text in search results and for lookup).
        """
        print(f"[INFO] Loading WikiBM25Retriever from {index_dir}...")
        start_time = time.time()

        self.index_dir = index_dir

        # Load titles
        titles_path = os.path.join(index_dir, "titles.json")
        if not os.path.exists(titles_path):
            raise FileNotFoundError(
                f"titles.json not found in {index_dir}. Run build_wiki_index.py first."
            )
        with open(titles_path, "r", encoding="utf-8") as f:
            self.titles = json.load(f)

        # Build title-to-index mapping for fast lookup
        self.title_to_idx = {}
        for idx, title in enumerate(self.titles):
            key = title.lower().strip()
            if key not in self.title_to_idx:
                self.title_to_idx[key] = idx

        # Load corpus texts if needed
        self.corpus_texts = None
        if load_corpus:
            self._load_corpus_texts()

        # Load BM25 index
        self.retriever = bm25s.BM25.load(index_dir, load_corpus=False)

        # Initialize stemmer (must match the one used during indexing)
        self.stemmer = Stemmer.Stemmer("english")

        elapsed = time.time() - start_time
        print(f"[INFO] WikiBM25Retriever loaded: {len(self.titles)} docs, {elapsed:.1f}s")

        # Load metadata if available
        meta_path = os.path.join(index_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            print(f"[INFO] Index metadata: {meta}")

    def _load_corpus_texts(self):
        """Load corpus texts from JSONL file."""
        corpus_path = os.path.join(self.index_dir, "corpus_texts.jsonl")
        if not os.path.exists(corpus_path):
            print(f"[WARN] corpus_texts.jsonl not found in {self.index_dir}. "
                  "Text will not be available in search results.")
            return

        print("[INFO] Loading corpus texts from JSONL (this may take a moment)...")
        load_start = time.time()
        self.corpus_texts = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.corpus_texts.append(json.loads(line))
        print(f"[INFO] Loaded {len(self.corpus_texts)} corpus texts in {time.time() - load_start:.1f}s")

    def search(self, query, top_k=3):
        """
        Search for relevant documents given a query.
        Returns list of (title, text, score) tuples.
        """
        # Tokenize query with the same stemmer used during indexing
        query_tokens = bm25s.tokenize(
            [query],
            stopwords="en",
            stemmer=self.stemmer,
            show_progress=False,
        )

        # Retrieve top-k
        results, scores = self.retriever.retrieve(query_tokens, k=top_k)

        output = []
        for idx_val, score_val in zip(results[0], scores[0]):
            idx = int(idx_val)
            score = float(score_val)
            title = self.titles[idx] if 0 <= idx < len(self.titles) else "Unknown"

            # Get text if corpus is loaded
            text = ""
            if self.corpus_texts is not None and 0 <= idx < len(self.corpus_texts):
                text = self.corpus_texts[idx]

            output.append((title, text, score))

        return output

    def lookup(self, title):
        """
        Look up a document by its exact title.
        Returns the text if found, else None.
        """
        title_lower = title.lower().strip()

        # Direct lookup via title-to-index mapping
        if self.corpus_texts is not None:
            idx = self.title_to_idx.get(title_lower)
            if idx is not None and 0 <= idx < len(self.corpus_texts):
                return self.corpus_texts[idx]

        # Fallback: BM25 search with the title as query, match by title
        results = self.search(title, top_k=10)
        for r_title, r_text, r_score in results:
            if r_title.lower().strip() == title_lower:
                return r_text if r_text else None

        return None
