#!/usr/bin/env python3
"""
Run StepKV online drop-step ablation on HotpotQA / 2Wiki / MuSiQue.

For each dataset and ratio in {0.2, 0.5}:
1) baseline step_aware_h2o
2) derive per-sample top/bottom/random step id from baseline step_scores
3) force-drop that step's full token span and re-run

Outputs:
- per-run raw result json/checkpoint
- aggregate summary json
- markdown table
- figure png
"""

import argparse
import json
import os
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

import run_all_wiki_experiments_v2 as base
import run_all_2wiki_experiments_v2 as runner_2wiki
import run_all_musique_experiments_v2 as runner_musique


def _save_json(path: str, data: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_drop_map(result_json: str, mode: str, seed: int) -> Dict[str, List[int]]:
    data = _load_json(result_json)
    rows = data.get("results", []) if isinstance(data, dict) else []
    rnd = random.Random(seed)
    out: Dict[str, List[int]] = {}
    for r in rows:
        sid = str(r.get("id", ""))
        sc = r.get("step_scores", {}) or r.get("debug_payload", {}).get("step_scores", {})
        pairs: List[Tuple[int, float]] = []
        for k, v in sc.items():
            try:
                pairs.append((int(k), float(v)))
            except Exception:
                continue
        if not sid or not pairs:
            continue
        if mode == "top1":
            pick = max(pairs, key=lambda x: x[1])[0]
        elif mode == "bottom1":
            pick = min(pairs, key=lambda x: x[1])[0]
        elif mode == "random1":
            pick = rnd.choice([x[0] for x in pairs])
        else:
            raise ValueError(mode)
        out[sid] = [int(pick)]
    return out


def _run_stepaware(selected_samples, retriever, out_json: str, ckpt_json: str, ratio: float,
                   drop_map=None, metrics_dataset="hotpotqa"):
    kv_override = {
        "cache_ratio": float(ratio),
        "attn_mode": "piggyback",
        "observation_window": 0,
        "step_poolwise_prune": True,
        "step_force_drop_map": drop_map or {},
    }
    em, f1, t = base.run_react_kv_experiment(
        val_data=None,
        selected_samples=selected_samples,
        retriever=retriever,
        pruning_mode="step_aware_h2o",
        output_path=out_json,
        checkpoint_path=ckpt_json,
        kv_config_override=kv_override,
        metrics_dataset=metrics_dataset,
    )
    return {"em": em, "f1": f1, "time_s": t}


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


def _plot(summary: Dict[str, Any], output_png: str):
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
    datasets = ["hotpotqa", "2wiki", "musique"]
    ratios = ["r20", "r50"]
    modes = ["top1", "bottom1", "random1"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax_i, metric in enumerate(["em", "f1"]):
        ax = axes[ax_i]
        x_labels = []
        vals = []
        for ds in datasets:
            for rr in ratios:
                for mm in modes:
                    row = summary["datasets"][ds][rr]["drops"][mm]["delta"][metric]
                    vals.append(row)
                    x_labels.append(f"{ds}-{rr}-{mm}")
        ax.bar(range(len(vals)), vals, width=0.72, color="#4C78A8")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(f"Delta {metric.upper()}")
        ax.set_title(f"Drop-step impact on {metric.upper()}")
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(x_labels, rotation=75, ha="right")
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()


def _write_md(summary: Dict[str, Any], output_md: str):
    lines = [
        "# StepKV Drop-step Ablation (All Datasets)",
        "",
        f"- generated_at_utc: {summary['generated_at_utc']}",
        f"- seed: {summary['seed']}",
        f"- num_samples: {summary['num_samples']}",
        "",
        "| Dataset | Ratio | Mode | Baseline EM | Drop EM | Delta EM | Baseline F1 | Drop F1 | Delta F1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ds, ds_data in summary["datasets"].items():
        for ratio, rdata in ds_data.items():
            b = rdata["baseline"]
            for mode, d in rdata["drops"].items():
                dr = d["drop"]
                de = d["delta"]
                lines.append(
                    f"| {ds} | {ratio} | {mode} | {b['em']:.2f} | {dr['em']:.2f} | {de['em']:.2f} | {b['f1']:.2f} | {dr['f1']:.2f} | {de['f1']:.2f} |"
                )
    os.makedirs(os.path.dirname(output_md), exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run StepKV drop-step ablation on 3 datasets.")
    parser.add_argument("--output_root", default="results/stepkv_drop_ablation_all", type=str)
    parser.add_argument("--num_samples", default=500, type=int)
    parser.add_argument("--seed", default=233, type=int)
    parser.add_argument("--model_path", default="/root/autodl-tmp/hf_cache/models/Meta-Llama-3.1-8B-Instruct", type=str)
    args = parser.parse_args()

    base.MODEL_PATH = args.model_path

    summary: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": int(args.seed),
        "num_samples": int(args.num_samples),
        "datasets": {},
    }

    for ds in ("hotpotqa", "2wiki", "musique"):
        selected, retriever = _prepare_dataset(ds, args.num_samples, args.seed)
        summary["datasets"][ds] = {}
        for ratio, rtag in ((0.2, "r20"), (0.5, "r50")):
            base_dir = os.path.join(args.output_root, ds, rtag, "baseline")
            os.makedirs(base_dir, exist_ok=True)
            base_json = os.path.join(base_dir, "stepaware_baseline.json")
            base_ckpt = os.path.join(base_dir, "stepaware_baseline_checkpoint.json")
            baseline = _run_stepaware(selected, retriever, base_json, base_ckpt, ratio, drop_map=None,
                                      metrics_dataset=ds)

            ratio_block = {"baseline": baseline, "drops": {}}
            for mode in ("top1", "bottom1", "random1"):
                drop_map = _extract_drop_map(base_json, mode=mode, seed=args.seed)
                ddir = os.path.join(args.output_root, ds, rtag, f"drop_{mode}")
                os.makedirs(ddir, exist_ok=True)
                djson = os.path.join(ddir, "stepaware_drop.json")
                dckpt = os.path.join(ddir, "stepaware_drop_checkpoint.json")
                dropped = _run_stepaware(selected, retriever, djson, dckpt, ratio, drop_map=drop_map,
                                         metrics_dataset=ds)
                ratio_block["drops"][mode] = {
                    "drop": dropped,
                    "delta": {
                        "em": dropped["em"] - baseline["em"],
                        "f1": dropped["f1"] - baseline["f1"],
                    },
                    "drop_map_size": len(drop_map),
                    "drop_map_path": os.path.join(ddir, "drop_map.json"),
                }
                _save_json(os.path.join(ddir, "drop_map.json"), drop_map)
            summary["datasets"][ds][rtag] = ratio_block

    out_json = os.path.join(args.output_root, "summary.json")
    out_md = os.path.join(args.output_root, "summary.md")
    out_png = os.path.join(args.output_root, "delta_bar.png")
    _save_json(out_json, summary)
    _write_md(summary, out_md)
    _plot(summary, out_png)
    print(f"[INFO] Saved summary json: {out_json}")
    print(f"[INFO] Saved summary md:   {out_md}")
    print(f"[INFO] Saved figure png:  {out_png}")


if __name__ == "__main__":
    main()
