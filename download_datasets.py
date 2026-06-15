#!/usr/bin/env python3
"""
Download 2Wiki / MuSiQue dev sets to local JSON for offline experiment runs.

Designed for servers where huggingface.co is unreachable (e.g. autodl in CN):
it routes through the hf-mirror.com endpoint by default and disables Xet/CAS,
which is the failure source on unstable networks.

Output (matches the loaders' DEFAULT_*_LOCAL_PATH):
- data/2wiki/dev.json
- data/musique/dev.json

Each file is a JSON list of {"id", "question", "answer"}.

Usage:
    # default: use hf-mirror.com
    python download_datasets.py

    # only one dataset
    python download_datasets.py --datasets 2wiki

    # use the official endpoint (if your network can reach it)
    HF_ENDPOINT=https://huggingface.co python download_datasets.py
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def _ensure_endpoint(endpoint: str) -> None:
    # Only set if user did not already export one.
    os.environ.setdefault("HF_ENDPOINT", endpoint)
    # Plain HTTP path is far more reliable than Xet/CAS on blocked networks.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    print(f"[INFO] HF_ENDPOINT={os.environ['HF_ENDPOINT']}")


def _save_list_json(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrote {len(rows)} examples -> {path}")


def _pick_split(ds) -> str:
    for name in ("validation", "dev", "test", "train"):
        if name in ds:
            return name
    # Fallback to first available split.
    return list(ds.keys())[0]


def _download(
    hf_candidates: List[Tuple[str, Optional[str]]],
    normalize_fn,
    out_path: str,
    min_expected_rows: int = 1,
) -> bool:
    from datasets import load_dataset

    cache_dir = os.path.dirname(out_path)
    last_err: Optional[Exception] = None
    for ds_name, ds_cfg in hf_candidates:
        try:
            print(f"[INFO] Trying HF dataset '{ds_name}' (config={ds_cfg}) ...")
            ds = load_dataset(ds_name, ds_cfg, cache_dir=cache_dir)
            split_name = _pick_split(ds)
            split_data = ds[split_name]
            rows: List[Dict[str, Any]] = []
            for i in range(len(split_data)):
                norm = normalize_fn(dict(split_data[i]), i)
                if norm is not None:
                    rows.append(norm)
            if not rows:
                raise ValueError("normalized 0 examples")
            if len(rows) < min_expected_rows:
                # Likely the wrong source/split (e.g. an incomplete dev set).
                # Skip and try the next candidate instead of silently saving it.
                raise ValueError(
                    f"only {len(rows)} rows from split '{split_name}', "
                    f"expected >= {min_expected_rows}; trying next source"
                )
            print(f"[INFO] Loaded {len(rows)} examples from '{ds_name}' split '{split_name}'.")
            _save_list_json(out_path, rows)
            return True
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[WARN] '{ds_name}' failed: {e}")
            continue
    print(f"[ERROR] All candidates failed for {out_path}. Last error: {last_err}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download 2Wiki / MuSiQue dev sets locally.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["2wiki", "musique"],
        choices=["2wiki", "musique"],
        help="Which datasets to download.",
    )
    parser.add_argument(
        "--endpoint",
        default="https://hf-mirror.com",
        help="HF endpoint to use (default: hf-mirror.com). Ignored if HF_ENDPOINT is already set.",
    )
    parser.add_argument(
        "--data_root",
        default="/root/autodl-tmp/kvmem/data",
        help="Root dir for downloaded data; files go to <data_root>/2wiki/dev.json and <data_root>/musique/dev.json.",
    )
    args = parser.parse_args()

    _ensure_endpoint(args.endpoint)

    # Import after endpoint is set so the loaders' env defaults don't override us.
    import run_all_2wiki_experiments_v2 as runner_2wiki
    import run_all_musique_experiments_v2 as runner_musique

    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    path_2wiki = os.path.join(data_root, "2wiki", "dev.json")
    path_musique = os.path.join(data_root, "musique", "dev.json")
    print(f"[INFO] data_root={data_root}")

    ok = True
    if "2wiki" in args.datasets:
        ok &= _download(
            hf_candidates=[
                ("2wikimultihopqa", None),
                ("scholarly-shadows-syndicate/2wikimultihopqa", None),
                ("voidful/2WikiMultihopQA", None),
            ],
            normalize_fn=runner_2wiki._normalize_2wiki_item,
            out_path=path_2wiki,
        )

    if "musique" in args.datasets:
        # dgslibisey/MuSiQue has the proper MuSiQue-Answerable splits
        # (train=19938, validation=2417). Prefer it over bdsaglam/musique,
        # which is a 67k-row dump without a standard dev split and yields an
        # incomplete validation set.
        ok &= _download(
            hf_candidates=[
                ("dgslibisey/MuSiQue", None),
                ("bdsaglam/musique", "default"),
                ("musique", None),
            ],
            normalize_fn=runner_musique._normalize_musique_item,
            out_path=path_musique,
            min_expected_rows=2000,
        )

    if not ok:
        raise SystemExit(1)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
