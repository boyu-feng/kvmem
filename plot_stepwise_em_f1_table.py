#!/usr/bin/env python3
"""
Re-plot step-wise EM/F1 from saved table JSON (no re-analysis).

Each step bucket is discrete (not a continuous curve). Supported styles:
  heatmap  - methods x steps color grid (default; soft, paper-friendly)
  dot      - marker-only categorical dot plot (no connecting lines)
  facet    - one small panel per method (cross-method comparison is indirect)
  bar      - grouped bars (legacy)

Expected JSON format (from analyze_stepwise_em_f1_table.py):
  {"methods": {"FullKV": {"step_stats": [{"step": 1, "avg_em": ..., "avg_f1": ...}, ...]}}}

Usage:
  python plot_stepwise_em_f1_table.py
  python plot_stepwise_em_f1_table.py --style heatmap --soft_colors
  python plot_stepwise_em_f1_table.py --style dot --compress_y
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required. Install with: pip install matplotlib"
    ) from exc


DEFAULT_INPUT_JSON = "results/stepwise_em_f1_table.json"
DEFAULT_OUTPUT_PNG = "results/stepwise_em_f1_table_replot.png"
DEFAULT_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
MARKERS = ["o", "s", "D", "^", "v", "P"]


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

    out: Dict[str, List[Dict[str, Any]]] = {}
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
                return float(row.get(field, 0.0))
        except (TypeError, ValueError):
            continue
    return float("nan")


def _value_matrix(
    methods: Sequence[str],
    steps: Sequence[int],
    method_to_stats: Dict[str, List[Dict[str, Any]]],
    field: str,
    scale: float,
) -> List[List[float]]:
    return [
        [_lookup(method_to_stats[method], step, field) * scale for step in steps]
        for method in methods
    ]


def _finite_values(matrix: Sequence[Sequence[float]]) -> List[float]:
    out: List[float] = []
    for row in matrix:
        for v in row:
            if isinstance(v, (int, float)) and math.isfinite(v):
                out.append(float(v))
    return out


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _color_limits(
    matrix: Sequence[Sequence[float]],
    *,
    soft: bool,
) -> Tuple[float, float]:
    vals = _finite_values(matrix)
    if not vals:
        return 0.0, 1.0
    if soft:
        return _percentile(vals, 0.08), _percentile(vals, 0.92)
    return min(vals), max(vals)


def _compressed_ylim(
    matrix: Sequence[Sequence[float]],
    *,
    margin_ratio: float = 0.12,
) -> Tuple[float, float]:
    vals = _finite_values(matrix)
    if not vals:
        return 0.0, 1.0
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1e-6)
    pad = span * margin_ratio
    return lo - pad, hi + pad


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


def _save_figure(fig: Any, output_png: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(output_png)) or "."
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_png)
    pdf_path = os.path.splitext(output_png)[0] + ".pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[INFO] Saved: {output_png}")
    print(f"[INFO] Saved: {pdf_path}")


def _panel_defs() -> List[Tuple[str, str, float]]:
    return [
        ("avg_em", "EM (%)", 100.0),
        ("avg_f1", "F1", 1.0),
    ]


def plot_heatmap(
    method_to_stats: Dict[str, List[Dict[str, Any]]],
    output_png: str,
    *,
    labelsize: float = 18,
    ticksize: float = 14,
    legend_size: float = 12,
    soft_colors: bool = True,
    annotate: bool = False,
) -> None:
    _setup_style(labelsize, ticksize, legend_size)
    steps = _collect_steps(method_to_stats)
    methods = list(method_to_stats.keys())
    if not steps:
        raise ValueError("No step buckets found in JSON.")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, max(3.8, 0.55 * len(methods) + 2.2)))
    for ax, (field, ylabel, scale) in zip(axes, _panel_defs()):
        mat = _value_matrix(methods, steps, method_to_stats, field, scale)
        vmin, vmax = _color_limits(mat, soft=soft_colors)
        if math.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad(color="#f0f0f0")
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(steps)), [str(s) for s in steps], fontsize=ticksize)
        ax.set_yticks(range(len(methods)), methods, fontsize=ticksize)
        ax.set_xlabel("Number of Steps", fontsize=labelsize)
        ax.set_ylabel("", fontsize=labelsize)
        ax.tick_params(length=0)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=ticksize - 1)
        cbar.set_label(ylabel, fontsize=labelsize - 1)

        if annotate:
            for i, row in enumerate(mat):
                for j, val in enumerate(row):
                    if not math.isfinite(val):
                        continue
                    txt = f"{val:.1f}" if scale > 1.0 else f"{val:.2f}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=ticksize - 3, color="#222222")

    fig.tight_layout()
    _save_figure(fig, output_png)


def plot_dot(
    method_to_stats: Dict[str, List[Dict[str, Any]]],
    output_png: str,
    *,
    labelsize: float = 18,
    ticksize: float = 14,
    legend_size: float = 12,
    compress_y: bool = True,
) -> None:
    _setup_style(labelsize, ticksize, legend_size)
    steps = _collect_steps(method_to_stats)
    methods = list(method_to_stats.keys())
    if not steps:
        raise ValueError("No step buckets found in JSON.")

    n_methods = len(methods)
    x = list(range(len(steps)))
    jitter_w = min(0.22, 0.72 / max(1, n_methods))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharex=True)
    for ax, (field, ylabel, scale) in zip(axes, _panel_defs()):
        mat = _value_matrix(methods, steps, method_to_stats, field, scale)
        for i, method in enumerate(methods):
            vals = mat[i]
            shift = (i - (n_methods - 1) / 2.0) * jitter_w
            pos = [xi + shift for xi in x]
            color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            ax.scatter(
                pos,
                vals,
                s=78,
                marker=MARKERS[i % len(MARKERS)],
                color=color,
                edgecolors="white",
                linewidths=0.8,
                label=method,
                zorder=3,
            )

        ax.set_xticks(x, [str(s) for s in steps], fontsize=ticksize)
        ax.set_xlabel("Number of Steps", fontsize=labelsize)
        ax.set_ylabel(ylabel, fontsize=labelsize)
        ax.tick_params(axis="y", labelsize=ticksize)
        ax.grid(axis="y", alpha=0.22, linestyle=":", linewidth=0.7)
        ax.set_axisbelow(True)
        if compress_y:
            ax.set_ylim(_compressed_ylim(mat))

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
    _save_figure(fig, output_png)


def plot_facet(
    method_to_stats: Dict[str, List[Dict[str, Any]]],
    output_png: str,
    *,
    labelsize: float = 18,
    ticksize: float = 14,
    legend_size: float = 12,
    compress_y: bool = True,
) -> None:
    _setup_style(labelsize, ticksize, legend_size)
    steps = _collect_steps(method_to_stats)
    methods = list(method_to_stats.keys())
    if not steps:
        raise ValueError("No step buckets found in JSON.")

    n_methods = len(methods)
    fig, axes = plt.subplots(n_methods, 2, figsize=(12, max(2.4, 1.55 * n_methods)), sharex=True)
    if n_methods == 1:
        axes = [axes]

    for i, method in enumerate(methods):
        for j, (field, ylabel, scale) in enumerate(_panel_defs()):
            ax = axes[i][j]
            vals = [_lookup(method_to_stats[method], step, field) * scale for step in steps]
            x = list(range(len(steps)))
            color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            ax.scatter(x, vals, s=70, color=color, marker=MARKERS[i % len(MARKERS)], zorder=3)
            ax.grid(axis="y", alpha=0.18, linestyle=":", linewidth=0.7)
            ax.set_axisbelow(True)
            if compress_y:
                ax.set_ylim(_compressed_ylim([vals]))
            if j == 0:
                ax.set_ylabel(method, fontsize=labelsize - 1)
            else:
                ax.set_ylabel(ylabel, fontsize=labelsize - 1)
            if i == n_methods - 1:
                ax.set_xticks(x, [str(s) for s in steps], fontsize=ticksize)
                ax.set_xlabel("Number of Steps", fontsize=labelsize)
            else:
                ax.set_xticks(x, [str(s) for s in steps], fontsize=ticksize)
                ax.tick_params(labelbottom=True)

    fig.tight_layout()
    _save_figure(fig, output_png)


def plot_grouped_step_bars(
    method_to_stats: Dict[str, List[Dict[str, Any]]],
    output_png: str,
    *,
    labelsize: float = 18,
    ticksize: float = 14,
    legend_size: float = 12,
    compress_y: bool = True,
) -> None:
    _setup_style(labelsize, ticksize, legend_size)
    steps = _collect_steps(method_to_stats)
    methods = list(method_to_stats.keys())
    if not steps:
        raise ValueError("No step buckets found in JSON.")

    n_methods = len(methods)
    x = list(range(len(steps)))
    total_w = 0.78
    bar_w = total_w / max(1, n_methods)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharex=True)
    for ax, (field, ylabel, scale) in zip(axes, _panel_defs()):
        mat = _value_matrix(methods, steps, method_to_stats, field, scale)
        for i, method in enumerate(methods):
            vals = mat[i]
            shift = (i - (n_methods - 1) / 2.0) * bar_w
            pos = [xi + shift for xi in x]
            color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            ax.bar(pos, vals, width=bar_w * 0.88, label=method, color=color, edgecolor="white", linewidth=0.6)

        ax.set_xticks(x, [str(s) for s in steps], fontsize=ticksize)
        ax.set_xlabel("Number of Steps", fontsize=labelsize)
        ax.set_ylabel(ylabel, fontsize=labelsize)
        ax.tick_params(axis="y", labelsize=ticksize)
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
        ax.set_axisbelow(True)
        if compress_y:
            ax.set_ylim(_compressed_ylim(mat))

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
    _save_figure(fig, output_png)


def plot_stepwise(
    method_to_stats: Dict[str, List[Dict[str, Any]]],
    output_png: str,
    *,
    style: str = "heatmap",
    labelsize: float = 18,
    ticksize: float = 14,
    legend_size: float = 12,
    soft_colors: bool = True,
    compress_y: bool = True,
    annotate: bool = False,
) -> None:
    if style == "heatmap":
        plot_heatmap(
            method_to_stats,
            output_png,
            labelsize=labelsize,
            ticksize=ticksize,
            legend_size=legend_size,
            soft_colors=soft_colors,
            annotate=annotate,
        )
    elif style == "dot":
        plot_dot(
            method_to_stats,
            output_png,
            labelsize=labelsize,
            ticksize=ticksize,
            legend_size=legend_size,
            compress_y=compress_y,
        )
    elif style == "facet":
        plot_facet(
            method_to_stats,
            output_png,
            labelsize=labelsize,
            ticksize=ticksize,
            legend_size=legend_size,
            compress_y=compress_y,
        )
    elif style == "bar":
        plot_grouped_step_bars(
            method_to_stats,
            output_png,
            labelsize=labelsize,
            ticksize=ticksize,
            legend_size=legend_size,
            compress_y=compress_y,
        )
    else:
        raise ValueError(f"Unsupported style: {style}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-plot step-wise EM/F1 from saved JSON."
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
    parser.add_argument(
        "--style",
        choices=["heatmap", "dot", "facet", "bar"],
        default="heatmap",
        help="heatmap (default): soft grid; dot: discrete markers; facet: per-method panels; bar: grouped bars",
    )
    parser.add_argument(
        "--no_soft_colors",
        action="store_true",
        help="For heatmap: use full min-max color range instead of compressed range.",
    )
    parser.add_argument(
        "--no_compress_y",
        action="store_true",
        help="For dot/facet/bar: include zero in y-axis instead of zooming to data range.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="For heatmap: write numeric values in each cell.",
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
    print(f"[INFO] Plot style: {args.style}")

    plot_stepwise(
        method_to_stats,
        args.output_png,
        style=args.style,
        labelsize=args.labelsize,
        ticksize=args.ticksize,
        legend_size=args.legend_size,
        soft_colors=not args.no_soft_colors,
        compress_y=not args.no_compress_y,
        annotate=args.annotate,
    )


if __name__ == "__main__":
    main()
