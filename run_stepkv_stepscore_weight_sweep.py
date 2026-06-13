#!/usr/bin/env python3
"""
Sweep the step-score weight hyperparameter of StepKV (step_aware_h2o) and
measure final-answer accuracy (EM / F1).

The swept knob is `step_aware_beta`: the weight of the step-level utility score
in the pruning combined score (combined = alpha * HH + beta * StepScore).
By default alpha is coupled as alpha = 1 - beta, so larger beta means pruning
is driven more by step scores and less by token-level Heavy-Hitter scores.

For each dataset x cache_ratio in {0.2, 0.5} x beta in {1.0, 0.8, 0.6, 0.4, 0.2}:
    run step_aware_h2o and record EM / F1 / wall time.

Outputs (under --output_root):
- per-run raw result json + checkpoint
- aggregate summary json
- markdown table
- line plots (EM and F1 vs beta, one line per dataset-ratio)

Usage (run from the repo root, e.g. /root/autodl-tmp/kvmem):
    python run_stepkv_stepscore_weight_sweep.py
    python run_stepkv_stepscore_weight_sweep.py --datasets hotpotqa 2wiki musique
    python run_stepkv_stepscore_weight_sweep.py --betas 1 0.8 0.6 0.4 0.2 --no_couple_alpha
"""

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import matplotlib.pyplot as plt

import run_all_wiki_experiments_v2 as base
import run_all_2wiki_experiments_v2 as runner_2wiki
import run_all_musique_experiments_v2 as runner_musique


def _save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _prepare_dataset(name: str, num_samples: int, seed: int):
    base.NUM_SAMPLES = int(num_samples)
    base.RANDOM_SEED = int(seed)
    if name == "hotpotqa":
        val_data = base.load_hotpotqa_data()
    elif name == "2wiki":
        val_data = runner_2wiki.load_2wiki_data(runner_2wiki.DEFAULT_2WIKI_LOCAL_PATH)
    elif name == "musique":
        val_data = runner_musique.load_musique_data(runner_musique.DEFAULT_MUSIQUE_LOCAL_PATH)
    else:
        raise ValueError(name)
    selected = base.select_samples(val_data)
    from retrievers.WikiBM25Retriever import WikiBM25Retriever

    retriever = WikiBM25Retriever(index_dir=base.WIKI_INDEX_DIR, load_corpus=True)
    return selected, retriever


def _run_one(selected_samples, retriever, out_json: str, ckpt_json: str,
             ratio: float, beta: float, couple_alpha: bool) -> Dict[str, float]:
    alpha = (1.0 - float(beta)) if couple_alpha else None
    kv_override: Dict[str, Any] = {
        "cache_ratio": float(ratio),
        "attn_mode": "piggyback",
        "observation_window": 0,
        "step_poolwise_prune": True,
        "step_aware_beta": float(beta),
    }
    if alpha is not None:
        kv_override["step_aware_alpha"] = float(alpha)

    em, f1, t = base.run_react_kv_experiment(
        val_data=None,
        selected_samples=selected_samples,
        retriever=retriever,
        pruning_mode="step_aware_h2o",
        output_path=out_json,
        checkpoint_path=ckpt_json,
        kv_config_override=kv_override,
    )
    return {
        "em": float(em),
        "f1": float(f1),
        "time_s": float(t),
        "beta": float(beta),
        "alpha": float(alpha) if alpha is not None else None,
    }


def _plot(summary: Dict[str, Any], betas: List[float], output_png: str) -> None:
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )
    xs = sorted(betas)
    ratio_tags = [("0.2", "r20"), ("0.5", "r50")]
    markers = {"r20": "o", "r50": "s"}
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax_i, metric in enumerate(["em", "f1"]):
        ax = axes[ax_i]
        ci = 0
        for ds, ds_data in summary["datasets"].items():
            for ratio_str, rtag in ratio_tags:
                if rtag not in ds_data:
                    continue
                ys = []
                for b in xs:
                    key = f"{b:g}"
                    row = ds_data[rtag]["betas"].get(key)
                    ys.append(row[metric] if row is not None else float("nan"))
                ax.plot(
                    xs, ys,
                    marker=markers.get(rtag, "o"),
                    color=colors[ci % len(colors)],
                    linewidth=1.8,
                    markersize=6,
                    label=f"{ds} (ratio={ratio_str})",
                )
                ci += 1
        ax.set_xlabel(r"step-score weight $\beta$")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} vs step-score weight")
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(xs)
        ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()


def _write_md(summary: Dict[str, Any], betas: List[float], output_md: str) -> None:
    lines = [
        "# StepKV step-score weight sweep",
        "",
        f"- generated_at_utc: {summary['generated_at_utc']}",
        f"- seed: {summary['seed']}",
        f"- num_samples: {summary['num_samples']}",
        f"- couple_alpha (alpha = 1 - beta): {summary['couple_alpha']}",
        f"- betas: {summary['betas']}",
        "",
        "| Dataset | Ratio | beta | alpha | EM | F1 | Time(s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for ds, ds_data in summary["datasets"].items():
        for rtag, rdata in ds_data.items():
            ratio_str = rdata.get("cache_ratio", rtag)
            for b in sorted(betas):
                row = rdata["betas"].get(f"{b:g}")
                if row is None:
                    continue
                a = "" if row.get("alpha") is None else f"{row['alpha']:.2f}"
                lines.append(
                    f"| {ds} | {ratio_str} | {b:g} | {a} | "
                    f"{row['em']:.2f} | {row['f1']:.2f} | {row['time_s']:.1f} |"
                )
    os.makedirs(os.path.dirname(output_md), exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep StepKV step-score weight (beta) and measure EM/F1.")
    parser.add_argument("--output_root", default="results/stepkv_stepscore_weight_sweep", type=str)
    parser.add_argument("--num_samples", default=500, type=int)
    parser.add_argument("--seed", default=233, type=int)
    parser.add_argument("--model_path", default="/root/autodl-tmp/hf_cache/models/Meta-Llama-3.1-8B-Instruct", type=str)
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa"],
                        choices=["hotpotqa", "2wiki", "musique"])
    parser.add_argument("--betas", nargs="+", type=float, default=[1.0, 0.8, 0.6, 0.4, 0.2])
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.2, 0.5])
    parser.add_argument("--no_couple_alpha", action="store_true",
                        help="If set, keep alpha at its default and only vary beta. "
                             "Default behavior couples alpha = 1 - beta.")
    args = parser.parse_args()

    base.MODEL_PATH = args.model_path
    couple_alpha = not args.no_couple_alpha

    summary: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": int(args.seed),
        "num_samples": int(args.num_samples),
        "couple_alpha": bool(couple_alpha),
        "betas": [float(b) for b in args.betas],
        "ratios": [float(r) for r in args.ratios],
        "datasets": {},
    }

    for ds in args.datasets:
        selected, retriever = _prepare_dataset(ds, args.num_samples, args.seed)
        summary["datasets"][ds] = {}
        for ratio in args.ratios:
            rtag = f"r{int(round(ratio * 100))}"
            ratio_block: Dict[str, Any] = {"cache_ratio": float(ratio), "betas": {}}
            for beta in args.betas:
                run_dir = os.path.join(args.output_root, ds, rtag, f"beta_{beta:g}")
                os.makedirs(run_dir, exist_ok=True)
                out_json = os.path.join(run_dir, "stepaware.json")
                ckpt_json = os.path.join(run_dir, "stepaware_checkpoint.json")
                res = _run_one(selected, retriever, out_json, ckpt_json,
                               ratio=ratio, beta=beta, couple_alpha=couple_alpha)
                ratio_block["betas"][f"{beta:g}"] = res
            summary["datasets"][ds][rtag] = ratio_block
            # Persist incrementally so partial progress is not lost.
            _save_json(os.path.join(args.output_root, "summary.json"), summary)

    out_json = os.path.join(args.output_root, "summary.json")
    out_md = os.path.join(args.output_root, "summary.md")
    out_png = os.path.join(args.output_root, "stepscore_weight_curve.png")
    _save_json(out_json, summary)
    _write_md(summary, args.betas, out_md)
    _plot(summary, args.betas, out_png)
    print(f"[INFO] Saved summary json: {out_json}")
    print(f"[INFO] Saved summary md:   {out_md}")
    print(f"[INFO] Saved figure png:  {out_png}")


if __name__ == "__main__":
    main()
