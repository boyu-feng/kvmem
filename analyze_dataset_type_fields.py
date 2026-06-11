#!/usr/bin/env python3
"""
Inspect dataset files and report potential categorical/type fields.
Useful for checking whether datasets contain explicit class labels.
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_list(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for k in ("data", "examples", "items"):
            v = data.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


def inspect_file(path: str, max_unique: int = 30) -> Dict[str, Any]:
    data = _as_list(_load_json(path))
    n = len(data)
    out: Dict[str, Any] = {"path": path, "rows": n, "categorical_fields": []}
    if not data:
        return out

    counters = defaultdict(Counter)
    nonnull = Counter()
    for row in data:
        for k, v in row.items():
            if v is None:
                continue
            nonnull[k] += 1
            if isinstance(v, str):
                s = v.strip()
                if s:
                    counters[k][s] += 1
            elif isinstance(v, (int, float, bool)):
                counters[k][str(v)] += 1

    for k, c in counters.items():
        uniq = len(c)
        if uniq <= max_unique:
            out["categorical_fields"].append(
                {
                    "field": k,
                    "unique": uniq,
                    "nonnull": int(nonnull[k]),
                    "nonnull_ratio": float(nonnull[k] / n) if n else 0.0,
                    "top_values": c.most_common(10),
                }
            )
    out["categorical_fields"].sort(key=lambda x: (x["unique"], -x["nonnull"]))
    out["sample_keys"] = sorted(data[0].keys())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect categorical/type fields in dataset JSON files.")
    parser.add_argument("paths", nargs="+", help="Dataset json paths")
    parser.add_argument("--output_json", default="results/dataset_type_field_report.json")
    args = parser.parse_args()

    reports = []
    for p in args.paths:
        if not os.path.exists(p):
            reports.append({"path": p, "error": "file_not_found"})
            continue
        reports.append(inspect_file(p))

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"reports": reports}, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved report: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
