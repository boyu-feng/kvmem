"""
2Wiki experiment runner (v2).

This script reuses the existing experiment implementations in
`run_all_wiki_experiments_v2.py` (Single / RAG / ReAct / ReAct-KV variants),
but switches data loading to 2WikiMultihopQA and writes outputs to a separate
2Wiki result directory.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import run_all_wiki_experiments_v2 as base


DEFAULT_2WIKI_CACHE_DIR = "data/2wiki"
DEFAULT_2WIKI_LOCAL_PATH = "data/2wiki/dev.json"
DEFAULT_OUTPUT_DIR = "results/2wiki_v2"


def _coerce_answer(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
    if value is None:
        return ""
    return str(value)


def _normalize_2wiki_item(item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    question = item.get("question")
    if not isinstance(question, str) or not question.strip():
        return None

    answer = _coerce_answer(item.get("answer", item.get("answers")))
    if not answer:
        answer = _coerce_answer(item.get("answer_text", ""))

    sample_id = item.get("id", item.get("_id", item.get("qid", str(idx))))
    if not isinstance(sample_id, str):
        sample_id = str(sample_id)

    return {
        "id": sample_id,
        "question": question.strip(),
        "answer": answer.strip(),
    }


def _load_2wiki_from_local(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        if isinstance(raw.get("data"), list):
            raw = raw["data"]
        elif isinstance(raw.get("examples"), list):
            raw = raw["examples"]
        else:
            raise ValueError(f"Unsupported 2Wiki JSON object format in {path}")

    if not isinstance(raw, list):
        raise ValueError(f"Expected list-like JSON data in {path}")

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        norm = _normalize_2wiki_item(item, i)
        if norm is not None:
            out.append(norm)
    return out


def load_2wiki_data(local_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load 2Wiki data as a list of {id, question, answer}.

    Priority:
    1) local json path if provided and exists
    2) default local path data/2wiki/dev.json if exists
    3) HuggingFace dataset fallback
    """
    candidate = local_path or DEFAULT_2WIKI_LOCAL_PATH
    if candidate and os.path.exists(candidate):
        print(f"[INFO] Loading 2Wiki from local file: {candidate}")
        data = _load_2wiki_from_local(candidate)
        print(f"[INFO] Loaded {len(data)} 2Wiki examples from local file.")
        return data

    print("[INFO] Local 2Wiki file not found. Falling back to HuggingFace datasets...")
    from datasets import load_dataset

    hf_candidates = [
        ("2wikimultihopqa", None),
        ("scholarly-shadows-syndicate/2wikimultihopqa", None),
    ]
    last_err: Optional[Exception] = None
    for ds_name, ds_cfg in hf_candidates:
        try:
            ds = load_dataset(ds_name, ds_cfg, cache_dir=DEFAULT_2WIKI_CACHE_DIR)
            split_name = "validation" if "validation" in ds else ("dev" if "dev" in ds else "train")
            split_data = ds[split_name]
            out: List[Dict[str, Any]] = []
            for i in range(len(split_data)):
                row = dict(split_data[i])
                norm = _normalize_2wiki_item(row, i)
                if norm is not None:
                    out.append(norm)
            print(f"[INFO] Loaded {len(out)} examples from HF dataset '{ds_name}' split '{split_name}'.")
            return out
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        "Failed to load 2Wiki data from local file and HuggingFace candidates. "
        f"Last error: {last_err}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run v2 experiments on 2WikiMultihopQA")
    parser.add_argument("--experiment", type=str, default="react_kv_step_aware_h2o", choices=[
        "single", "rag", "react",
        "react_kv_none", "react_kv_h2o", "react_kv_step_anchor_h2o",
        "react_kv_step_aware_h2o", "react_kv_snapkv", "ours", "all"
    ])
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--data_path", type=str, default=DEFAULT_2WIKI_LOCAL_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bm25_top_k", type=int, default=5)
    parser.add_argument("--wiki_index_dir", type=str, default=base.WIKI_INDEX_DIR)
    args = parser.parse_args()

    # Reuse base module config knobs so downstream functions stay unchanged.
    base.NUM_SAMPLES = int(args.num_samples)
    base.RANDOM_SEED = int(args.seed)
    base.BM25_TOP_K = int(args.bm25_top_k)
    base.WIKI_INDEX_DIR = args.wiki_index_dir

    os.makedirs(args.output_dir, exist_ok=True)

    val_data = load_2wiki_data(args.data_path)
    selected_samples = base.select_samples(val_data)

    needs_retriever = args.experiment in [
        "rag", "react", "react_kv_none", "react_kv_h2o",
        "react_kv_step_anchor_h2o", "react_kv_step_aware_h2o",
        "react_kv_snapkv", "ours", "all"
    ]
    retriever = None
    if needs_retriever:
        from retrievers.WikiBM25Retriever import WikiBM25Retriever
        if not os.path.exists(base.WIKI_INDEX_DIR):
            raise FileNotFoundError(
                f"Wiki index not found: {base.WIKI_INDEX_DIR}. "
                "Please build the BM25 index first."
            )
        retriever = WikiBM25Retriever(index_dir=base.WIKI_INDEX_DIR, load_corpus=True)

    if args.experiment in ("single", "all"):
        base.run_single_experiment(
            val_data, selected_samples,
            os.path.join(args.output_dir, "single_2wiki.json"),
            os.path.join(args.output_dir, "single_2wiki_checkpoint.json"),
        )
    if args.experiment in ("rag", "all"):
        base.run_rag_experiment(
            val_data, selected_samples, retriever,
            os.path.join(args.output_dir, "rag_2wiki.json"),
            os.path.join(args.output_dir, "rag_2wiki_checkpoint.json"),
        )
    if args.experiment in ("react", "all"):
        base.run_react_experiment(
            val_data, selected_samples, retriever,
            os.path.join(args.output_dir, "react_2wiki.json"),
            os.path.join(args.output_dir, "react_2wiki_checkpoint.json"),
        )
    if args.experiment in ("react_kv_none", "all"):
        base.run_react_kv_experiment(
            val_data, selected_samples, retriever, "none",
            os.path.join(args.output_dir, "react_kv_none_2wiki.json"),
            os.path.join(args.output_dir, "react_kv_none_2wiki_checkpoint.json"),
        )
    if args.experiment in ("react_kv_h2o", "all"):
        base.run_react_kv_experiment(
            val_data, selected_samples, retriever, "h2o",
            os.path.join(args.output_dir, "react_kv_h2o_2wiki.json"),
            os.path.join(args.output_dir, "react_kv_h2o_2wiki_checkpoint.json"),
        )
    if args.experiment in ("react_kv_step_anchor_h2o", "all"):
        base.run_react_kv_experiment(
            val_data, selected_samples, retriever, "step_anchor_h2o",
            os.path.join(args.output_dir, "react_kv_step_anchor_h2o_2wiki.json"),
            os.path.join(args.output_dir, "react_kv_step_anchor_h2o_2wiki_checkpoint.json"),
        )
    if args.experiment in ("react_kv_step_aware_h2o", "all"):
        base.run_react_kv_experiment(
            val_data, selected_samples, retriever, "step_aware_h2o",
            os.path.join(args.output_dir, "react_kv_step_aware_h2o_2wiki.json"),
            os.path.join(args.output_dir, "react_kv_step_aware_h2o_2wiki_checkpoint.json"),
        )
    if args.experiment in ("react_kv_snapkv", "all"):
        base.run_react_kv_experiment(
            val_data, selected_samples, retriever, "snapkv",
            os.path.join(args.output_dir, "react_kv_snapkv_2wiki.json"),
            os.path.join(args.output_dir, "react_kv_snapkv_2wiki_checkpoint.json"),
        )
    if args.experiment in ("ours", "all"):
        base.run_react_kv_experiment(
            val_data, selected_samples, retriever, "ours",
            os.path.join(args.output_dir, "react_kv_ours_2wiki.json"),
            os.path.join(args.output_dir, "react_kv_ours_2wiki_checkpoint.json"),
        )

    print("[DONE] 2Wiki experiments complete.")


if __name__ == "__main__":
    main()
