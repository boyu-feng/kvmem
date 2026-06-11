#!/usr/bin/env python3
"""
Analyze categorical/type fields on the selected 500 samples for:
- HotpotQA
- 2Wiki
- MuSiQue

Sampling follows experiment logic: shuffle by seed, take first N.
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List

import run_all_wiki_experiments_v2 as base
import run_all_2wiki_experiments_v2 as runner_2wiki
import run_all_musique_experiments_v2 as runner_musique


def _load_hotpot_with_fallback() -> List[Dict[str, Any]]:
    try:
        return list(base.load_hotpotqa_data())
    except Exception:
        local = "data/dev.json"
        if not os.path.exists(local):
            raise
        with open(local, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k in ("data", "examples", "items"):
                if isinstance(data.get(k), list):
                    data = data[k]
                    break
        if not isinstance(data, list):
            raise ValueError("Unsupported local Hotpot format in data/dev.json")
        return [x for x in data if isinstance(x, dict)]


def _select_samples(rows: List[Dict[str, Any]], seed: int, num_samples: int) -> List[Dict[str, Any]]:
    idxs = list(range(len(rows)))
    random.Random(seed).shuffle(idxs)
    chosen = idxs[: min(num_samples, len(idxs))]
    return [rows[i] for i in chosen]


def _collect_categorical_stats(rows: List[Dict[str, Any]], max_unique: int = 20) -> Dict[str, Any]:
    n = len(rows)
    counters: Dict[str, Counter] = defaultdict(Counter)
    nonnull: Counter = Counter()

    for row in rows:
        if not isinstance(row, dict):
            continue
        for k, v in row.items():
            if v is None:
                continue
            if isinstance(v, str):
                s = v.strip()
                if not s:
                    continue
                nonnull[k] += 1
                counters[k][s] += 1
            elif isinstance(v, (int, float, bool)):
                nonnull[k] += 1
                counters[k][str(v)] += 1

    fields = []
    for k, c in counters.items():
        uniq = len(c)
        if uniq <= max_unique:
            fields.append(
                {
                    "field": k,
                    "unique_count": uniq,
                    "nonnull_count": int(nonnull[k]),
                    "nonnull_ratio": (nonnull[k] / n) if n else 0.0,
                    "top_values": c.most_common(20),
                }
            )

    fields.sort(key=lambda x: (x["unique_count"], -x["nonnull_count"], x["field"]))
    return {"n_selected": n, "categorical_fields": fields}


def _write_md(path: str, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Selected 500 Type-Field Analysis")
    lines.append("")
    lines.append(f"- generated_at_utc: {payload['generated_at_utc']}")
    lines.append(f"- seed: {payload['seed']}")
    lines.append(f"- num_samples: {payload['num_samples']}")
    lines.append("")

    for ds in payload["datasets"]:
        lines.append(f"## {ds['dataset']}")
        lines.append(f"- selected_samples: {ds['n_selected']}")
        if not ds["categorical_fields"]:
            lines.append("- No obvious low-cardinality categorical fields found.")
            lines.append("")
            continue
        lines.append("| Field | Unique | Non-null | Ratio | Top Values |")
        lines.append("|---|---:|---:|---:|---|")
        for f in ds["categorical_fields"]:
            top_str = ", ".join([f"{k}:{v}" for k, v in f["top_values"][:8]])
            lines.append(
                f"| `{f['field']}` | {f['unique_count']} | {f['nonnull_count']} | {f['nonnull_ratio']:.2%} | {top_str} |"
            )
        lines.append("")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze type/category fields on selected 500 samples across 3 datasets.")
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--data_path_2wiki", type=str, default=runner_2wiki.DEFAULT_2WIKI_LOCAL_PATH)
    parser.add_argument("--data_path_musique", type=str, default=runner_musique.DEFAULT_MUSIQUE_LOCAL_PATH)
    parser.add_argument("--output_json", type=str, default="results/selected500_type_fields_all_datasets.json")
    parser.add_argument("--output_md", type=str, default="results/selected500_type_fields_all_datasets.md")
    args = parser.parse_args()

    hotpot_rows = _load_hotpot_with_fallback()
    rows_2wiki = runner_2wiki.load_2wiki_data(args.data_path_2wiki)
    rows_musique = runner_musique.load_musique_data(args.data_path_musique)

    hotpot_sel = _select_samples(hotpot_rows, args.seed, args.num_samples)
    wiki2_sel = _select_samples(rows_2wiki, args.seed, args.num_samples)
    musique_sel = _select_samples(rows_musique, args.seed, args.num_samples)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": int(args.seed),
        "num_samples": int(args.num_samples),
        "datasets": [
            {"dataset": "hotpotqa", **_collect_categorical_stats(hotpot_sel)},
            {"dataset": "2wiki", **_collect_categorical_stats(wiki2_sel)},
            {"dataset": "musique", **_collect_categorical_stats(musique_sel)},
        ],
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _write_md(args.output_md, payload)

    print(f"[INFO] Wrote JSON: {args.output_json}")
    print(f"[INFO] Wrote MD:   {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
