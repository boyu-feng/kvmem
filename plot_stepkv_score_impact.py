#!/usr/bin/env python3
"""
Re-plot StepKV score-impact hit-rate bars from saved JSON (no re-analysis).

Expected JSON format: output of analyze_stepaware_score_impact.py
  results/stepkv_score_impact_proxy.json

Usage:
  python plot_stepkv_score_impact.py \\
    --input_json results/stepkv_score_impact_proxy.json

  python plot_stepkv_score_impact.py \\
    --input_json results/stepkv_score_impact_proxy.json \\
    --output_png results/stepkv_score_impact_proxy_replot.png
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required. Install with: pip install matplotlib"
    ) from exc


DEFAULT_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#FF9DA6", "#9D755D"]
MODES = ["top1", "bottom1", "random1"]
MODE_LABELS = ["Top-1", "Bottom-1", "Random-1"]


def load_metrics(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("metrics"), dict):
        return data["metrics"]
    if isinstance(data, dict):
        return data
    raise ValueError(f"Unsupported score-impact JSON format: {path}")


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


def plot_hit_rate_bars(
    all_metrics: Dict[str, Dict[str, Any]],
    output_png: str,
    *,
    labelsize: float = 16,
    ticksize: float = 13,
    legend_size: float = 11,
) -> None:
    _setup_style(labelsize, ticksize, legend_size)

    labels = list(all_metrics.keys())
    n_groups = len(labels)
    x = list(range(len(MODES)))
    total_w = 0.72
    bar_w = total_w / max(1, n_groups)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    metric_keys = ("answer_hit_rate", "em_true_answer_hit_rate")

    for ax, metric_key in zip(axes, metric_keys):
        for i, label in enumerate(labels):
            agg = all_metrics[label]
            vals = [float(agg[m].get(metric_key, 0.0)) * 100.0 for m in MODES]
            shift = (i - (n_groups - 1) / 2.0) * bar_w
            pos = [v + shift for v in x]
            color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            ax.bar(pos, vals, width=bar_w * 0.86, label=label, color=color, edgecolor="white", linewidth=0.6)

        ax.set_xticks(x, MODE_LABELS, fontsize=ticksize)
        ax.set_ylabel("Hit Rate (%)", fontsize=labelsize)
        ax.tick_params(axis="y", labelsize=ticksize)
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
        ax.set_axisbelow(True)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=min(4, max(1, n_groups)),
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
        description="Re-plot StepKV score-impact hit-rate bars from saved JSON."
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default="results/stepkv_score_impact_proxy.json",
        help="Path to stepkv_score_impact_proxy.json",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default=None,
        help="Output figure path (default: same dir as input, *_replot.png)",
    )
    parser.add_argument("--labelsize", type=float, default=16, help="Axis label font size.")
    parser.add_argument("--ticksize", type=float, default=13, help="Tick label font size.")
    parser.add_argument("--legend_size", type=float, default=11, help="Legend font size.")
    args = parser.parse_args()

    input_json = os.path.abspath(args.input_json)
    if not os.path.isfile(input_json):
        raise FileNotFoundError(f"JSON not found: {input_json}")

    all_metrics = load_metrics(input_json)
    if args.output_png:
        output_png = args.output_png
    else:
        output_png = os.path.join(
            os.path.dirname(input_json),
            "stepkv_score_impact_proxy_replot.png",
        )

    plot_hit_rate_bars(
        all_metrics,
        output_png,
        labelsize=args.labelsize,
        ticksize=args.ticksize,
        legend_size=args.legend_size,
    )


if __name__ == "__main__":
    main()
