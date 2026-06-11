#!/usr/bin/env python3
"""
Build step-wise EM/F1 summary table from experiment result JSONs.

Input JSON format: expects top-level "results" list, and each item contains:
- "num_steps"
- "em"
- "f1"
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Any

import matplotlib.pyplot as plt


def _load_results(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported result format: {path}")


def _step_stats(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bucket = defaultdict(list)
    for r in rows:
        s = r.get("num_steps")
        if s is None:
            continue
        try:
            step = int(s)
        except Exception:
            continue
        em = 1.0 if bool(r.get("em", False)) else 0.0
        try:
            f1 = float(r.get("f1", 0.0))
        except Exception:
            f1 = 0.0
        bucket[step].append((em, f1))

    out = []
    for step in sorted(bucket.keys()):
        vals = bucket[step]
        n = len(vals)
        avg_em = sum(v[0] for v in vals) / n if n else 0.0
        avg_f1 = sum(v[1] for v in vals) / n if n else 0.0
        out.append(
            {
                "step": step,
                "n_samples": n,
                "avg_em": avg_em,
                "avg_f1": avg_f1,
            }
        )
    return out


def _write_md(path: str, method_to_stats: Dict[str, List[Dict[str, Any]]]) -> None:
    lines: List[str] = []
    lines.append("# Step-wise EM/F1")
    lines.append("")
    for method, stats in method_to_stats.items():
        lines.append(f"## {method}")
        lines.append("| Step | N | Avg EM | Avg F1 |")
        lines.append("|---:|---:|---:|---:|")
        for row in stats:
            lines.append(
                f"| {row['step']} | {row['n_samples']} | {row['avg_em']:.2%} | {row['avg_f1']:.4f} |"
            )
        lines.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _plot_lines(
    method_to_stats: Dict[str, List[Dict[str, Any]]],
    output_png_em: str,
    output_png_f1: str,
) -> None:
    # Paper-style figure defaults
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )
    os.makedirs(os.path.dirname(output_png_em), exist_ok=True)
    os.makedirs(os.path.dirname(output_png_f1), exist_ok=True)

    # EM line chart
    plt.figure(figsize=(8, 5))
    for method, stats in method_to_stats.items():
        xs = [row["step"] for row in stats]
        ys = [row["avg_em"] * 100.0 for row in stats]
        if xs:
            plt.plot(xs, ys, marker="o", linewidth=2.0, markersize=4.5, label=method)
    plt.xlabel("Step")
    plt.ylabel("Average EM (%)")
    plt.title("Step-wise Exact Match")
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.7)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(output_png_em, dpi=200)
    plt.close()

    # F1 line chart
    plt.figure(figsize=(8, 5))
    for method, stats in method_to_stats.items():
        xs = [row["step"] for row in stats]
        ys = [row["avg_f1"] for row in stats]
        if xs:
            plt.plot(xs, ys, marker="o", linewidth=2.0, markersize=4.5, label=method)
    plt.xlabel("Step")
    plt.ylabel("Average F1")
    plt.title("Step-wise F1")
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.7)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(output_png_f1, dpi=200)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create step-wise EM/F1 table from result JSON files.")
    parser.add_argument(
        "--result",
        action="append",
        nargs=2,
        metavar=("METHOD", "JSON_PATH"),
        required=True,
        help="Repeatable pair: method_name result_json_path",
    )
    parser.add_argument("--output_md", type=str, default="results/stepwise_em_f1_table.md")
    parser.add_argument("--output_json", type=str, default="results/stepwise_em_f1_table.json")
    parser.add_argument("--output_png_em", type=str, default="results/stepwise_avg_em.png")
    parser.add_argument("--output_png_f1", type=str, default="results/stepwise_avg_f1.png")
    args = parser.parse_args()

    payload = {"methods": {}}
    method_to_stats: Dict[str, List[Dict[str, Any]]] = {}

    for method, path in args.result:
        rows = _load_results(path)
        stats = _step_stats(rows)
        method_to_stats[method] = stats
        payload["methods"][method] = {
            "source": path,
            "total_rows": len(rows),
            "step_stats": stats,
        }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _write_md(args.output_md, method_to_stats)
    _plot_lines(method_to_stats, args.output_png_em, args.output_png_f1)

    print(f"[INFO] Wrote JSON: {args.output_json}")
    print(f"[INFO] Wrote MD:   {args.output_md}")
    print(f"[INFO] Wrote PNG:  {args.output_png_em}")
    print(f"[INFO] Wrote PNG:  {args.output_png_f1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
