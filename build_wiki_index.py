"""
Build a BM25 index over the full Wikipedia corpus for HotpotQA.

Uses TIGER-Lab/LongRAG hotpot_qa_wiki dataset (~5.2M Wikipedia articles)
and bm25s library for efficient BM25 indexing with disk persistence.

Usage:
    python build_wiki_index.py [--index_dir INDEX_DIR] [--batch_size BATCH_SIZE]

This follows the approach used in SearchR1 and similar papers where a large
external Wikipedia corpus is used as the retrieval source for HotpotQA,
rather than the per-sample distractor context paragraphs.
"""

import os
import json
import time
import argparse
import bm25s
import Stemmer

from datasets import load_dataset


# ==================== Configuration ====================
DATA_CACHE_DIR = "data/hotpotqa"
DEFAULT_INDEX_DIR = "data/wiki_index"


def build_wiki_index(index_dir, batch_size=50000):
    """
    Build a BM25 index from the TIGER-Lab/LongRAG hotpot_qa_wiki corpus.

    Steps:
    1. Load the wiki corpus dataset (streaming for memory efficiency)
    2. Extract titles and texts
    3. Build bm25s index with stemming
    4. Save index + metadata to disk
    """
    os.makedirs(index_dir, exist_ok=True)

    print("[INFO] Loading TIGER-Lab/LongRAG hotpot_qa_wiki corpus...")
    start_time = time.time()

    # Load dataset - use non-streaming to allow random access
    # The dataset has ~5.2M entries, each with title and doc_dict (text)
    ds = load_dataset(
        "TIGER-Lab/LongRAG",
        "hotpot_qa_wiki",
        cache_dir=DATA_CACHE_DIR,
        split="train",
    )
    print(f"[INFO] Dataset loaded: {len(ds)} articles in {time.time() - start_time:.1f}s")

    # Extract titles and texts
    print("[INFO] Extracting titles and texts...")
    titles = []
    corpus_texts = []

    for i, item in enumerate(ds):
        title = item["title"]
        text = item["doc_dict"]
        if not text or not text.strip():
            continue
        titles.append(title)
        corpus_texts.append(text)

        if (i + 1) % 500000 == 0:
            print(f"  Processed {i + 1}/{len(ds)} articles...")

    print(f"[INFO] Extracted {len(corpus_texts)} non-empty articles (from {len(ds)} total)")

    # Save titles metadata
    titles_path = os.path.join(index_dir, "titles.json")
    print(f"[INFO] Saving titles to {titles_path}...")
    with open(titles_path, "w", encoding="utf-8") as f:
        json.dump(titles, f, ensure_ascii=False)

    # Build BM25 index using bm25s with English stemmer
    print("[INFO] Tokenizing corpus with stemming (this may take a while)...")
    stemmer = Stemmer.Stemmer("english")

    # bm25s tokenize supports batched processing
    corpus_tokens = bm25s.tokenize(
        corpus_texts,
        stopwords="en",
        stemmer=stemmer,
        show_progress=True,
    )

    print("[INFO] Building BM25 index...")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=True)

    # Save index to disk
    print(f"[INFO] Saving BM25 index to {index_dir}...")
    retriever.save(index_dir)

    # Save corpus texts as line-delimited JSONL for efficient loading
    # Each line: JSON-encoded string of the document text
    corpus_texts_path = os.path.join(index_dir, "corpus_texts.jsonl")
    print(f"[INFO] Saving corpus texts to {corpus_texts_path}...")
    with open(corpus_texts_path, "w", encoding="utf-8") as f:
        for text in corpus_texts:
            f.write(json.dumps(text, ensure_ascii=False) + "\n")

    # Also save a small metadata file
    meta = {
        "num_docs": len(corpus_texts),
        "dataset": "TIGER-Lab/LongRAG hotpot_qa_wiki",
        "build_time_seconds": time.time() - start_time,
    }
    with open(os.path.join(index_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    total_time = time.time() - start_time
    print(f"\n[DONE] Index built successfully!")
    print(f"  Articles indexed: {len(corpus_texts)}")
    print(f"  Index directory:  {index_dir}")
    print(f"  Total time:       {total_time:.1f}s ({total_time/3600:.2f}h)")


def main():
    parser = argparse.ArgumentParser(description="Build BM25 index over Wikipedia corpus for HotpotQA")
    parser.add_argument(
        "--index_dir", type=str, default=DEFAULT_INDEX_DIR,
        help=f"Directory to save the BM25 index (default: {DEFAULT_INDEX_DIR})"
    )
    parser.add_argument(
        "--batch_size", type=int, default=50000,
        help="Batch size for processing (default: 50000)"
    )
    args = parser.parse_args()

    build_wiki_index(args.index_dir, args.batch_size)


if __name__ == "__main__":
    main()
