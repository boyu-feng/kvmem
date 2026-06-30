#!/usr/bin/env python3
"""
Re-plot StepKV score-impact hit-rate bars from saved JSON (no re-analysis).

Supported JSON formats (from analyze_stepaware_score_impact.py):
1) Combined multi-group:
     {"metrics": {"r20": {"top1": {...}, ...}, "r50": {...}}}
2) Single-group (legacy):
     {"metrics": {"top1": {...}, "bottom1": {...}, "random1": {...}}}

Usage:
  python plot_stepkv_score_impact.py

  python plot_stepkv_score_impact.py \\
    --input_json results/stepkv_score_impact_proxy_r20_r50.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required. Install with: pip install matplotlib"
    ) from exc


DEFAULT_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#FF9DA6", "#9D755D"]
MODES = ["top1", "bottom1", "random1"]
MODE_LABELS = ["Top-1", "Bottom-1", "Random-1"]
DEFAULT_INPUT_JSON = "results/stepkv_score_impact_proxy_r20_r50.json"
DEFAULT_OUTPUT_PNG = "results/stepkv_score_impact_proxy_r20_r50_replot.png"


def _is_mode_agg(obj: Any) -> bool:
    return isinstance(obj, dict) and "answer_hit_rate" in obj


def _is_group_agg(obj: Any) -> bool:
    return isinstance(obj, dict) and all(isinstance(obj.get(m), dict) for m in MODES)


def _guess_group_from_path(path: str) -> str:
    base = os.path.basename(path).lower()
    if "r20_r50" in base or "r20-r50" in base:
        return "combined"
    if re.search(r"[_\-/]r20(?:[_\-.]|$)", base) and "r50" not in base:
        return "r20"
    if re.search(r"[_\-/]r50(?:[_\-.]|$)", base) and "r20" not in base:
        return "r50"
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.endswith("_r20"):
        return "r20"
    if stem.endswith("_r50"):
        return "r50"
    return stem or "default"


def normalize_metrics_block(raw: Any, default_group: str) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected object JSON, got {type(raw).__name__}")

    metrics = raw.get("metrics", raw)
    if not isinstance(metrics, dict):
        raise ValueError("JSON must contain a metrics object.")

    if _is_group_agg(metrics):
        return {default_group: metrics}

    grouped: Dict[str, Dict[str, Any]] = {}
    for key, value in metrics.items():
        if _is_group_agg(value):
            grouped[str(key)] = value
    if grouped:
        return grouped

    raise ValueError(
        "Unsupported metrics format. Expected either "
        '{"metrics": {"top1": ..., "bottom1": ..., "random1": ...}} '
        'or {"metrics": {"r20": {"top1": ...}, "r50": {...}}}.'
    )


def load_metrics_from_file(path: str, group: str | None = None) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    label = group or _guess_group_from_path(path)
    return normalize_metrics_block(data, default_group=label)


def merge_metric_groups(items: List[Tuple[str, str]]) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for group, path in items:
        block = load_metrics_from_file(path, group=group)
        for g, agg in block.items():
            if g in merged:
                raise ValueError(f"Duplicate group {g!r} from {path}")
            merged[g] = agg
    return merged


def discover_default_inputs(results_dir: str = "results") -> List[Tuple[str, str]]:
    for name in (
        "stepkv_score_impact_proxy_r20_r50.json",
        "stepkv_score_impact_proxy.json",
    ):
        path = os.path.join(results_dir, name)
        if os.path.isfile(path):
            return [("combined", path)]

    found: List[Tuple[str, str]] = []
    for group in ("r20", "r50"):
        path = os.path.join(results_dir, f"stepkv_score_impact_proxy_{group}.json")
        if os.path.isfile(path):
            found.append((group, path))
    if found:
        return found

    matches = sorted(glob.glob(os.path.join(results_dir, "stepkv_score_impact_proxy*.json")))
    return [(_guess_group_from_path(p), p) for p in matches]


def _resolve_json_path(path: str) -> str:
    if os.path.isfile(path):
        return path
    if not path.endswith(".json") and os.path.isfile(path + ".json"):
        return path + ".json"
    return path


def _sorted_group_labels(all_metrics: Dict[str, Dict[str, Any]]) -> List[str]:
    order = {"r20": 0, "r50": 1}
    return sorted(all_metrics.keys(), key=lambda k: (order.get(k, 99), k))


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
    labelsize: float = 18,
    ticksize: float = 14,
    legend_size: float = 12,
) -> None:
    _setup_style(labelsize, ticksize, legend_size)

    labels = _sorted_group_labels(all_metrics)
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
        nargs="*",
        default=None,
        help="One or more score-impact JSON files. Group name is inferred from filename "
             "(e.g. *_r20.json -> r20).",
    )
    parser.add_argument(
        "--result",
        action="append",
        nargs=2,
        metavar=("GROUP", "JSON_PATH"),
        help="Explicit group/json pair. Repeat for r20 + r50.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to auto-discover JSON when no input is given.",
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

    items: List[Tuple[str, str]] = []
    if args.result:
        items.extend((group, os.path.abspath(path)) for group, path in args.result)
    elif args.input_json:
        for path in args.input_json:
            abs_path = _resolve_json_path(os.path.abspath(path))
            items.append((_guess_group_from_path(abs_path), abs_path))
    else:
        default_path = _resolve_json_path(os.path.abspath(DEFAULT_INPUT_JSON))
        if os.path.isfile(default_path):
            items = [("combined", default_path)]
        else:
            items = [(g, os.path.abspath(p)) for g, p in discover_default_inputs(args.results_dir)]

    if not items:
        raise FileNotFoundError(
            f"No score-impact JSON found. Expected: {DEFAULT_INPUT_JSON}"
        )

    missing = [p for _, p in items if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError("JSON not found: " + ", ".join(missing))

    all_metrics = merge_metric_groups(items)
    print("[INFO] Loaded: " + ", ".join(f"{g} <- {p}" for g, p in items))
    print("[INFO] Plot groups: " + ", ".join(all_metrics.keys()))

    if args.output_png:
        output_png = args.output_png
    else:
        output_png = DEFAULT_OUTPUT_PNG

    plot_hit_rate_bars(
        all_metrics,
        output_png,
        labelsize=args.labelsize,
        ticksize=args.ticksize,
        legend_size=args.legend_size,
    )


if __name__ == "__main__":
    main()
