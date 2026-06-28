#!/usr/bin/env python3
"""
Re-plot StepKV beta sweep results from summary.json (no re-run experiments).

Expected JSON format: output of run_stepkv_stepscore_weight_sweep.py
  results/stepkv_stepscore_weight_sweep/summary.json

Usage:
  python plot_stepkv_beta_sweep.py \\
    --input_json results/stepkv_stepscore_weight_sweep/summary.json

  python plot_stepkv_beta_sweep.py \\
    --input_json results/stepkv_stepscore_weight_sweep/summary.json \\
    --output_png results/stepkv_stepscore_weight_sweep/stepscore_weight_curve_v2.png
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


DEFAULT_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
RATIO_TAGS = [("0.2", "r20"), ("0.5", "r50")]
MARKERS = {"r20": "o", "r50": "s"}


def load_summary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "datasets" not in data:
        raise ValueError(f"Invalid beta sweep summary JSON: {path}")
    return data


def _collect_betas(summary: Dict[str, Any]) -> List[float]:
    if isinstance(summary.get("betas"), list) and summary["betas"]:
        return sorted(float(b) for b in summary["betas"])

    found = set()
    for ds_data in summary.get("datasets", {}).values():
        if not isinstance(ds_data, dict):
            continue
        for rdata in ds_data.values():
            if not isinstance(rdata, dict):
                continue
            for key in (rdata.get("betas") or {}).keys():
                try:
                    found.add(float(key))
                except ValueError:
                    continue
    if not found:
        raise ValueError("No beta values found in summary JSON.")
    return sorted(found)


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


def plot_beta_sweep(
    summary: Dict[str, Any],
    output_png: str,
    labelsize: float = 16,
    ticksize: float = 13,
    legend_size: float = 11,
    show_title: bool = False,
) -> None:
    betas = _collect_betas(summary)
    _setup_style(labelsize, ticksize, legend_size)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    for ax, metric in zip(axes, ("em", "f1")):
        ci = 0
        for ds, ds_data in summary.get("datasets", {}).items():
            if not isinstance(ds_data, dict):
                continue
            for ratio_str, rtag in RATIO_TAGS:
                if rtag not in ds_data:
                    continue
                rblock = ds_data[rtag]
                beta_map = rblock.get("betas") or {}
                ys: List[float] = []
                for b in betas:
                    row = beta_map.get(f"{b:g}")
                    if row is None:
                        ys.append(float("nan"))
                    else:
                        ys.append(float(row.get(metric, float("nan"))))
                ax.plot(
                    betas,
                    ys,
                    marker=MARKERS.get(rtag, "o"),
                    color=DEFAULT_COLORS[ci % len(DEFAULT_COLORS)],
                    linewidth=2.0,
                    markersize=7,
                    label=f"{ds} (ratio={ratio_str})",
                )
                ci += 1

        ax.set_xlabel(r"step-score weight $\beta$", fontsize=labelsize)
        ax.set_ylabel(metric.upper(), fontsize=labelsize)
        ax.tick_params(axis="both", labelsize=ticksize)
        ax.set_xticks(betas)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
        ax.set_axisbelow(True)
        if show_title:
            ax.set_title(f"{metric.upper()} vs $\\beta$", fontsize=labelsize)
        ax.legend(frameon=False, fontsize=legend_size)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)) or ".", exist_ok=True)
    fig.savefig(output_png)
    pdf_path = os.path.splitext(output_png)[0] + ".pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[INFO] Saved: {output_png}")
    print(f"[INFO] Saved: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-plot StepKV beta sweep from summary.json.")
    parser.add_argument(
        "--input_json",
        type=str,
        default="results/stepkv_stepscore_weight_sweep/summary.json",
        help="Path to beta sweep summary.json",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default=None,
        help="Output figure path (default: same dir as input, stepscore_weight_curve_replot.png)",
    )
    parser.add_argument("--labelsize", type=float, default=16, help="Axis label font size.")
    parser.add_argument("--ticksize", type=float, default=13, help="Tick label font size.")
    parser.add_argument("--legend_size", type=float, default=11, help="Legend font size.")
    parser.add_argument("--show_title", action="store_true", help="Show subplot titles.")
    args = parser.parse_args()

    input_json = os.path.abspath(args.input_json)
    if not os.path.isfile(input_json):
        raise FileNotFoundError(f"JSON not found: {input_json}")

    summary = load_summary(input_json)
    if args.output_png:
        output_png = args.output_png
    else:
        output_png = os.path.join(
            os.path.dirname(input_json),
            "stepscore_weight_curve_replot.png",
        )

    plot_beta_sweep(
        summary,
        output_png,
        labelsize=args.labelsize,
        ticksize=args.ticksize,
        legend_size=args.legend_size,
        show_title=args.show_title,
    )


if __name__ == "__main__":
    main()
