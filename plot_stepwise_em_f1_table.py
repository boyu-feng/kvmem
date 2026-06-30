#!/usr/bin/env python3
"""
Re-plot step-wise EM/F1 from saved table JSON (no re-analysis).

Uses grouped bar charts (not line charts) so each step bucket is shown as a
discrete category rather than a continuous curve.

Expected JSON format (from analyze_stepwise_em_f1_table.py):
  {
    "methods": {
      "FullKV": {
        "step_stats": [
          {"step": 1, "n_samples": 120, "avg_em": 0.12, "avg_f1": 0.25},
          ...
        ]
      },
      ...
    }
  }

Usage:
  python plot_stepwise_em_f1_table.py

  python plot_stepwise_em_f1_table.py \\
    --input_json results/stepwise_em_f1_table.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required. Install with: pip install matplotlib"
    ) from exc


DEFAULT_INPUT_JSON = "results/stepwise_em_f1_table.json"
DEFAULT_OUTPUT_PNG = "results/stepwise_em_f1_table_replot.png"
DEFAULT_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]


def _resolve_json_path(path: str) -> str:
    if os.path.isfile(path):
        return path
    if not path.endswith(".json") and os.path.isfile(path + ".json"):
        return path + ".json"
    return path


def load_method_stats(path: str) -> Dict[str, List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON: {path}")

    if isinstance(data.get("methods"), dict):
        out: Dict[str, List[Dict[str, Any]]] = {}
        for method, block in data["methods"].items():
            stats = block.get("step_stats") if isinstance(block, dict) else block
            if not isinstance(stats, list):
                raise ValueError(f"Missing step_stats for method {method!r} in {path}")
            out[str(method)] = stats
        return out

    # Legacy: top-level keys are method names.
    out = {}
    for method, block in data.items():
        if method in ("meta", "generated_at_utc"):
            continue
        if isinstance(block, list):
            out[str(method)] = block
        elif isinstance(block, dict) and isinstance(block.get("step_stats"), list):
            out[str(method)] = block["step_stats"]
    if not out:
        raise ValueError(f"Unsupported step-wise table JSON format: {path}")
    return out


def _collect_steps(method_to_stats: Dict[str, List[Dict[str, Any]]]) -> List[int]:
    steps = set()
    for stats in method_to_stats.values():
        for row in stats:
            try:
                steps.add(int(row["step"]))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(steps)


def _lookup(stats: List[Dict[str, Any]], step: int, field: str) -> float:
    for row in stats:
        try:
            if int(row.get("step", -1)) == step:
                val = row.get(field, 0.0)
                return float(val)
        except (TypeError, ValueError):
            continue
    return float("nan")


def _setup_style(labelsize: float, ticksize: float, legend_size: float) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": ticksize,
            "axes.labelsize": labelsize,
            "xtick.labelsize": ticksize,
            "ytick.labelsize": ticksize,
            "legend.fontsize": legend_size,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_grouped_step_bars(
    method_to_stats: Dict[str, List[Dict[str, Any]]],
    output_png: str,
    *,
    labelsize: float = 18,
    ticksize: float = 14,
    legend_size: float = 12,
) -> None:
    _setup_style(labelsize, ticksize, legend_size)

    steps = _collect_steps(method_to_stats)
    if not steps:
        raise ValueError("No step buckets found in JSON.")

    methods = list(method_to_stats.keys())
    n_methods = len(methods)
    x = list(range(len(steps)))
    total_w = 0.78
    bar_w = total_w / max(1, n_methods)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharex=True)
    panels: List[Tuple[Any, str, str, float]] = [
        (axes[0], "avg_em", "EM (%)", 100.0),
        (axes[1], "avg_f1", "F1", 1.0),
    ]

    for ax, field, ylabel, scale in panels:
        for i, method in enumerate(methods):
            vals = [_lookup(method_to_stats[method], step, field) * scale for step in steps]
            shift = (i - (n_methods - 1) / 2.0) * bar_w
            pos = [xi + shift for xi in x]
            color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            ax.bar(
                pos,
                vals,
                width=bar_w * 0.88,
                label=method,
                color=color,
                edgecolor="white",
                linewidth=0.6,
            )

        ax.set_xticks(x, [str(s) for s in steps], fontsize=ticksize)
        ax.set_xlabel("Number of Steps", fontsize=labelsize)
        ax.set_ylabel(ylabel, fontsize=labelsize)
        ax.tick_params(axis="y", labelsize=ticksize)
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=min(4, max(1, n_methods)),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=legend_size,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    out_dir = os.path.dirname(os.path.abspath(output_png)) or "."
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_png)
    pdf_path = os.path.splitext(output_png)[0] + ".pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[INFO] Saved: {output_png}")
    print(f"[INFO] Saved: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-plot step-wise EM/F1 as grouped bar charts from saved JSON."
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default=DEFAULT_INPUT_JSON,
        help=f"Path to step-wise table JSON (default: {DEFAULT_INPUT_JSON})",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default=DEFAULT_OUTPUT_PNG,
        help=f"Output figure path (default: {DEFAULT_OUTPUT_PNG})",
    )
    parser.add_argument("--labelsize", type=float, default=18, help="Axis label font size.")
    parser.add_argument("--ticksize", type=float, default=14, help="Tick label font size.")
    parser.add_argument("--legend_size", type=float, default=12, help="Legend font size.")
    args = parser.parse_args()

    input_json = _resolve_json_path(os.path.abspath(args.input_json))
    if not os.path.isfile(input_json):
        raise FileNotFoundError(f"JSON not found: {input_json}")

    method_to_stats = load_method_stats(input_json)
    steps = _collect_steps(method_to_stats)
    print(f"[INFO] Loaded {len(method_to_stats)} methods, steps={steps} from {input_json}")

    plot_grouped_step_bars(
        method_to_stats,
        args.output_png,
        labelsize=args.labelsize,
        ticksize=args.ticksize,
        legend_size=args.legend_size,
    )


if __name__ == "__main__":
    main()
