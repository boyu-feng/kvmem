#!/usr/bin/env python3
"""
Analyze which decode tokens are discarded under different KV pruning methods.

Two analyses:
1) method_compare: H2O vs TOVA vs StepKV (step_aware_h2o) on the same sample.
2) beta_sweep: StepKV with different step-score weights (step_aware_beta).

Outputs JSON with saved discarded token IDs and PNG figures.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_all_wiki_experiments_v2 as base
from models.QwenLLMWithKVCache import QwenLLMWithKVCache
from retrievers.WikiBM25Retriever import WikiBM25Retriever
from token_tracker import TokenTracker


METHOD_LABELS = {
    "h2o": "H2O",
    "tova": "TOVA",
    "step_aware_h2o": "StepKV",
}


def _build_kv_config(
    pruning_mode: str,
    cache_ratio: float = 0.5,
    beta: Optional[float] = None,
    alpha: Optional[float] = None,
) -> Dict[str, Any]:
    obs_window_default = 0 if pruning_mode in ("step_aware_h2o", "step_inter", "tova") else 32
    attn_mode_default = "piggyback" if pruning_mode in ("step_aware_h2o", "step_inter") else "scoring_forward"
    step_poolwise_default = pruning_mode in ("step_aware_h2o", "step_inter")
    cfg: Dict[str, Any] = {
        "pruning_mode": pruning_mode,
        "prune_every_n": 1,
        "cache_ratio": float(cache_ratio),
        "protect_prompt": True,
        "pool_window": 4,
        "max_trajectory_tokens": 1024,
        "sink_size": 4,
        "observation_window": obs_window_default,
        "num_score_layers": 3,
        "attn_mode": attn_mode_default,
        "step_anchor_keep_last_obs": 1,
        "step_aware_alpha": 0.8,
        "step_aware_beta": 0.8,
        "step_aware_min_keep": 12,
        "step_aware_min_keep_ratio": 0.30,
        "step_aware_bonus": 0.0,
        "step_poolwise_prune": step_poolwise_default,
        "step_reward_weight": 0.85,
        "step_citation_weight": 0.15,
        "prompt_prefill_keep_ratio": 1.0,
    }
    if pruning_mode in ("step_aware_h2o", "step_inter") and beta is not None:
        cfg["step_aware_beta"] = float(beta)
        if alpha is not None:
            cfg["step_aware_alpha"] = float(alpha)
    return cfg


def _owner_step(global_id: int, step_token_ranges: Dict[str, Any]) -> int:
    gid = int(global_id)
    for sid_str, rng in sorted(step_token_ranges.items(), key=lambda kv: int(kv[0])):
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            continue
        s, e = int(rng[0]), int(rng[1])
        if s <= gid <= e:
            return int(sid_str)
    return -1


def extract_discarded_tokens(debug_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Collect all discarded decode token global IDs from debug payload."""
    prompt_token_count = int(debug_payload.get("prompt_token_count", 0) or 0)
    step_token_ranges = debug_payload.get("step_token_ranges", {}) or {}
    token_tracker = debug_payload.get("token_tracker", {}) or {}
    step_pruning_events = token_tracker.get("step_pruning_events", {}) or {}

    all_global: Set[int] = set()
    by_prune_step: Dict[str, List[int]] = {}
    by_owner_step: Dict[str, List[int]] = {}
    records: List[Dict[str, Any]] = []

    for prune_step_str, dropped_ids in sorted(step_pruning_events.items(), key=lambda kv: str(kv[0])):
        uniq = sorted(set(int(x) for x in (dropped_ids or [])))
        decode_ids = [gid for gid in uniq if gid >= prompt_token_count]
        if not decode_ids:
            continue
        by_prune_step[str(prune_step_str)] = decode_ids
        all_global.update(decode_ids)
        for gid in decode_ids:
            owner = _owner_step(gid, step_token_ranges)
            owner_key = str(owner)
            by_owner_step.setdefault(owner_key, []).append(gid)
            records.append(
                {
                    "global_id": int(gid),
                    "decode_index": int(gid - prompt_token_count),
                    "prune_step": str(prune_step_str),
                    "owner_step": int(owner),
                }
            )

    decode_sorted = sorted(int(gid - prompt_token_count) for gid in all_global)
    boundaries = []
    for sid_str, rng in sorted(step_token_ranges.items(), key=lambda kv: int(kv[0])):
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            continue
        end_shifted = int(rng[1]) - prompt_token_count
        if end_shifted >= 0:
            boundaries.append({"step": int(sid_str), "x": float(end_shifted) + 0.5})

    return {
        "prompt_token_count": prompt_token_count,
        "discarded_global_ids": sorted(all_global),
        "discarded_decode_indices": decode_sorted,
        "discarded_count": len(all_global),
        "by_prune_step": by_prune_step,
        "by_owner_step": {k: sorted(set(v)) for k, v in by_owner_step.items()},
        "records": records,
        "step_token_ranges": step_token_ranges,
        "step_boundaries": boundaries,
        "step_scores": debug_payload.get("step_scores", {}),
    }


def _prepare_dataset(args) -> Tuple[List[Tuple[int, Dict[str, Any]]], Any]:
    base.NUM_SAMPLES = int(args.num_samples)
    base.RANDOM_SEED = int(args.seed)
    base.MAX_STEPS = base.parse_max_steps(args.max_steps)
    base.BM25_TOP_K = int(args.bm25_top_k)
    base.WIKI_INDEX_DIR = args.wiki_index_dir
    base.MODEL_PATH = args.model_path

    retriever = None
    if args.dataset == "hotpotqa":
        val_data = base.load_hotpotqa_data()
        retriever = WikiBM25Retriever(index_dir=args.wiki_index_dir, load_corpus=True)
    elif args.dataset == "2wiki":
        import run_all_2wiki_experiments_v2 as runner_2wiki

        val_data = runner_2wiki.load_2wiki_data(args.data_path or runner_2wiki.DEFAULT_2WIKI_LOCAL_PATH)
        retriever = WikiBM25Retriever(index_dir=args.wiki_index_dir, load_corpus=True)
    elif args.dataset == "musique":
        import run_all_musique_experiments_v2 as runner_musique

        val_data = runner_musique.load_musique_data(args.data_path or runner_musique.DEFAULT_MUSIQUE_LOCAL_PATH)
        retriever = WikiBM25Retriever(index_dir=args.wiki_index_dir, load_corpus=True)
    elif args.dataset == "browsecomp":
        import run_all_browsecomp_experiments_v2 as runner_bc

        val_data = runner_bc.load_browsecomp_data(
            local_path=args.data_path,
            hf_dataset_name=args.hf_dataset_name,
            hf_split=args.hf_split,
            canary=args.canary,
        )
        if args.retriever_backend == "browsecomp_bm25":
            from retrievers.BrowseCompBM25Retriever import BrowseCompBM25Retriever

            retriever = BrowseCompBM25Retriever(index_dir=args.browsecomp_index_dir, load_corpus=True)
        else:
            from retrievers.WebSearchRetriever import WebSearchRetriever

            retriever = WebSearchRetriever(timeout_sec=int(args.web_timeout_sec))
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    selected = base.select_samples(val_data)
    return selected, retriever


def _pick_sample(selected: List[Tuple[int, Dict[str, Any]]], args) -> Tuple[int, int, Dict[str, Any]]:
    if args.sample_id:
        for pos, (orig_idx, sample) in enumerate(selected):
            if str(sample.get("id", "")) == str(args.sample_id):
                return pos, orig_idx, sample
        raise ValueError(f"sample_id not found in selected set: {args.sample_id}")
    pos = int(args.sample_pos)
    if pos < 0 or pos >= len(selected):
        raise IndexError(f"sample_pos out of range: {pos} (total {len(selected)})")
    orig_idx, sample = selected[pos]
    return pos, orig_idx, sample


def _run_one_sample(
    sample: Dict[str, Any],
    retriever,
    pruning_mode: str,
    cache_ratio: float,
    max_steps,
    beta: Optional[float] = None,
    alpha: Optional[float] = None,
) -> Dict[str, Any]:
    alpha_val = alpha
    if beta is not None and alpha_val is None:
        alpha_val = 1.0 - float(beta)
    kv_config = _build_kv_config(pruning_mode, cache_ratio=cache_ratio, beta=beta, alpha=alpha_val)
    token_tracker = TokenTracker()
    llm = QwenLLMWithKVCache(base.MODEL_PATH, kv_config, token_tracker=token_tracker)
    try:
        pred_answer, trajectory_log, step_timings, debug_payload = base._run_react_kv_episode(
            sample["question"],
            llm,
            retriever,
            pruning_mode=pruning_mode,
            max_steps=max_steps,
            return_debug=True,
        )
    finally:
        del llm

    discarded = extract_discarded_tokens(debug_payload if isinstance(debug_payload, dict) else {})
    return {
        "pruning_mode": pruning_mode,
        "kv_config": kv_config,
        "predicted_answer": pred_answer,
        "trajectory": trajectory_log,
        "step_timings": step_timings,
        "debug_payload": debug_payload,
        "discarded": discarded,
    }


def _overlap_summary(method_to_ids: Dict[str, Set[int]]) -> Dict[str, Any]:
    keys = list(method_to_ids.keys())
    summary: Dict[str, Any] = {"methods": keys, "counts": {k: len(method_to_ids[k]) for k in keys}}
    if len(keys) >= 2:
        all_sets = [method_to_ids[k] for k in keys]
        summary["intersection_all"] = sorted(set.intersection(*all_sets)) if all_sets else []
        summary["union_all"] = sorted(set.union(*all_sets)) if all_sets else []
    if len(keys) == 3:
        a, b, c = keys
        sa, sb, sc = method_to_ids[a], method_to_ids[b], method_to_ids[c]
        summary["pairwise"] = {
            f"only_{a}": sorted(sa - sb - sc),
            f"only_{b}": sorted(sb - sa - sc),
            f"only_{c}": sorted(sc - sa - sb),
            f"{a}_and_{b}_not_{c}": sorted((sa & sb) - sc),
            f"{a}_and_{c}_not_{b}": sorted((sa & sc) - sb),
            f"{b}_and_{c}_not_{a}": sorted((sb & sc) - sa),
            "all_three": sorted(sa & sb & sc),
        }
        summary["pairwise_counts"] = {k: len(v) for k, v in summary["pairwise"].items()}
    return summary


def _plot_method_compare(
    method_runs: Dict[str, Dict[str, Any]],
    overlap: Dict[str, Any],
    output_png: str,
    sample_meta: Dict[str, Any],
) -> None:
    methods = list(method_runs.keys())
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.2], hspace=0.28)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0])

    cmap = {"h2o": "#4C78A8", "tova": "#F58518", "step_aware_h2o": "#54A24B"}
    y_positions = {m: i for i, m in enumerate(methods)}

    boundaries = []
    for m in methods:
        boundaries = method_runs[m]["discarded"].get("step_boundaries", []) or boundaries
        xs = method_runs[m]["discarded"].get("discarded_decode_indices", [])
        ys = [y_positions[m]] * len(xs)
        ax_main.scatter(
            xs,
            ys,
            s=18,
            alpha=0.85,
            c=cmap.get(m, "#333333"),
            label=METHOD_LABELS.get(m, m),
        )

    for bd in boundaries:
        ax_main.axvline(float(bd["x"]), linestyle="--", linewidth=0.9, color="gray", alpha=0.55)

    ax_main.set_yticks([y_positions[m] for m in methods])
    ax_main.set_yticklabels([METHOD_LABELS.get(m, m) for m in methods])
    ax_main.set_xlabel("Decode Token Index (prompt excluded)")
    ax_main.set_title(
        f"Discarded Tokens by Method | sample={sample_meta.get('sample_id')} pos={sample_meta.get('sample_pos')}"
    )
    ax_main.grid(True, axis="x", alpha=0.25)
    ax_main.legend(loc="upper right")

    if "pairwise_counts" in overlap:
        labels = []
        values = []
        for key, cnt in overlap["pairwise_counts"].items():
            labels.append(key.replace("_", "\n"))
            values.append(cnt)
        ax_bar.bar(range(len(values)), values, color="#72B7B2")
        ax_bar.set_xticks(range(len(labels)))
        ax_bar.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax_bar.set_ylabel("# Discarded Tokens")
        ax_bar.set_title("Overlap Categories (decode token indices)")
        ax_bar.grid(True, axis="y", alpha=0.25)
    else:
        ax_bar.axis("off")

    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_beta_sweep(
    beta_runs: Dict[str, Dict[str, Any]],
    output_png: str,
    sample_meta: Dict[str, Any],
) -> None:
    betas = sorted(beta_runs.keys(), key=lambda x: float(x), reverse=True)
    fig, ax = plt.subplots(figsize=(16, max(4, 0.8 * len(betas) + 2)))
    cmap = plt.get_cmap("viridis")

    boundaries = []
    for i, beta in enumerate(betas):
        run = beta_runs[beta]
        discarded = run["discarded"]
        boundaries = discarded.get("step_boundaries", []) or boundaries
        xs = discarded.get("discarded_decode_indices", [])
        ys = [i] * len(xs)
        color = cmap(i / max(1, len(betas) - 1))
        ax.scatter(xs, ys, s=18, alpha=0.85, c=[color], label=f"beta={beta}")

    for bd in boundaries:
        ax.axvline(float(bd["x"]), linestyle="--", linewidth=0.9, color="gray", alpha=0.55)

    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels([f"beta={b}" for b in betas])
    ax.set_xlabel("Decode Token Index (prompt excluded)")
    ax.set_title(
        f"StepKV Discarded Tokens vs step-score weight | sample={sample_meta.get('sample_id')}"
    )
    ax.grid(True, axis="x", alpha=0.25)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_method_compare(args, selected, retriever, sample_pos, orig_idx, sample) -> Dict[str, Any]:
    methods = ["h2o", "tova", "step_aware_h2o"]
    method_runs: Dict[str, Dict[str, Any]] = {}
    method_to_ids: Dict[str, Set[int]] = {}

    for method in methods:
        print(f"[INFO] Running method_compare: {method} ...")
        run = _run_one_sample(
            sample,
            retriever,
            pruning_mode=method,
            cache_ratio=float(args.cache_ratio),
            max_steps=base.MAX_STEPS,
        )
        method_runs[method] = run
        method_to_ids[method] = set(run["discarded"].get("discarded_decode_indices", []))
        print(
            f"[INFO] {method}: discarded={len(method_to_ids[method])} decode tokens"
        )

    overlap = _overlap_summary(method_to_ids)
    return {
        "analysis": "method_compare",
        "methods": methods,
        "method_runs": {
            m: {
                "predicted_answer": method_runs[m]["predicted_answer"],
                "kv_config": method_runs[m]["kv_config"],
                "discarded": method_runs[m]["discarded"],
            }
            for m in methods
        },
        "overlap": overlap,
    }


def run_beta_sweep(args, selected, retriever, sample_pos, orig_idx, sample) -> Dict[str, Any]:
    beta_runs: Dict[str, Dict[str, Any]] = {}
    beta_to_ids: Dict[str, Set[int]] = {}

    for beta in args.betas:
        beta_str = f"{float(beta):g}"
        alpha = (1.0 - float(beta)) if args.couple_alpha else None
        print(f"[INFO] Running beta_sweep: beta={beta_str} alpha={alpha} ...")
        run = _run_one_sample(
            sample,
            retriever,
            pruning_mode="step_aware_h2o",
            cache_ratio=float(args.cache_ratio),
            max_steps=base.MAX_STEPS,
            beta=float(beta),
            alpha=alpha,
        )
        beta_runs[beta_str] = run
        beta_to_ids[beta_str] = set(run["discarded"].get("discarded_decode_indices", []))
        print(f"[INFO] beta={beta_str}: discarded={len(beta_to_ids[beta_str])} decode tokens")

    overlap = _overlap_summary(beta_to_ids)
    return {
        "analysis": "beta_sweep",
        "betas": [f"{float(b):g}" for b in args.betas],
        "couple_alpha": bool(args.couple_alpha),
        "beta_runs": {
            b: {
                "predicted_answer": beta_runs[b]["predicted_answer"],
                "kv_config": beta_runs[b]["kv_config"],
                "discarded": beta_runs[b]["discarded"],
            }
            for b in beta_runs
        },
        "overlap": overlap,
    }


def _save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _save_tokens_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze discarded tokens for H2O/TOVA/StepKV.")
    parser.add_argument(
        "--analysis",
        choices=["method_compare", "beta_sweep", "both"],
        default="both",
    )
    parser.add_argument("--dataset", choices=["hotpotqa", "2wiki", "musique", "browsecomp"], default="hotpotqa")
    parser.add_argument("--sample_pos", type=int, default=0)
    parser.add_argument("--sample_id", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--max_steps", type=str, default="7")
    parser.add_argument("--cache_ratio", type=float, default=0.5)
    parser.add_argument("--model_path", type=str, default=base.MODEL_PATH)
    parser.add_argument("--wiki_index_dir", type=str, default=base.WIKI_INDEX_DIR)
    parser.add_argument("--bm25_top_k", type=int, default=5)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--retriever_backend", type=str, default="wiki", choices=["wiki", "browsecomp_bm25", "web"])
    parser.add_argument("--browsecomp_index_dir", type=str, default="data/browsecomp_index")
    parser.add_argument("--hf_dataset_name", type=str, default="Tevatron/browsecomp-plus")
    parser.add_argument("--hf_split", type=str, default=None)
    parser.add_argument("--canary", type=str, default="")
    parser.add_argument("--web_timeout_sec", type=int, default=12)
    parser.add_argument("--betas", type=float, nargs="+", default=[1.0, 0.8, 0.6, 0.4, 0.2])
    parser.add_argument("--couple_alpha", action="store_true", help="Set step_aware_alpha = 1 - beta.")
    parser.add_argument("--output_dir", type=str, default="results/stepkv_discarded_token_analysis")
    args = parser.parse_args()

    if args.canary == "":
        import run_all_browsecomp_experiments_v2 as runner_bc

        args.canary = runner_bc.DEFAULT_BROWSECOMP_PLUS_CANARY

    os.makedirs(args.output_dir, exist_ok=True)
    selected, retriever = _prepare_dataset(args)
    sample_pos, orig_idx, sample = _pick_sample(selected, args)

    meta = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": args.dataset,
        "sample_pos": int(sample_pos),
        "orig_idx": int(orig_idx),
        "sample_id": str(sample.get("id", "")),
        "question": sample.get("question", ""),
        "gold_answer": sample.get("answer", ""),
        "seed": int(args.seed),
        "num_samples": int(args.num_samples),
        "max_steps": base.format_max_steps(base.parse_max_steps(args.max_steps)),
        "cache_ratio": float(args.cache_ratio),
        "model_path": args.model_path,
    }

    outputs: Dict[str, Any] = {"meta": meta, "analyses": {}}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{args.dataset}_sample{sample_pos}_{stamp}"

    if args.analysis in ("method_compare", "both"):
        cmp_payload = run_method_compare(args, selected, retriever, sample_pos, orig_idx, sample)
        outputs["analyses"]["method_compare"] = cmp_payload
        cmp_json = os.path.join(args.output_dir, f"{prefix}_method_compare.json")
        cmp_png = os.path.join(args.output_dir, f"{prefix}_method_compare.png")
        _save_json(cmp_json, cmp_payload)
        _plot_method_compare(
            cmp_payload["method_runs"],
            cmp_payload["overlap"],
            cmp_png,
            meta,
        )
        token_rows = []
        for method, run in cmp_payload["method_runs"].items():
            for rec in run["discarded"].get("records", []):
                token_rows.append({"method": method, **rec})
        _save_tokens_jsonl(
            os.path.join(args.output_dir, f"{prefix}_method_compare_tokens.jsonl"),
            token_rows,
        )
        print(f"[DONE] method_compare JSON: {cmp_json}")
        print(f"[DONE] method_compare figure: {cmp_png}")

    if args.analysis in ("beta_sweep", "both"):
        beta_payload = run_beta_sweep(args, selected, retriever, sample_pos, orig_idx, sample)
        outputs["analyses"]["beta_sweep"] = beta_payload
        beta_json = os.path.join(args.output_dir, f"{prefix}_beta_sweep.json")
        beta_png = os.path.join(args.output_dir, f"{prefix}_beta_sweep.png")
        _save_json(beta_json, beta_payload)
        _plot_beta_sweep(beta_payload["beta_runs"], beta_png, meta)
        token_rows = []
        for beta, run in beta_payload["beta_runs"].items():
            for rec in run["discarded"].get("records", []):
                token_rows.append({"beta": beta, **rec})
        _save_tokens_jsonl(
            os.path.join(args.output_dir, f"{prefix}_beta_sweep_tokens.jsonl"),
            token_rows,
        )
        print(f"[DONE] beta_sweep JSON: {beta_json}")
        print(f"[DONE] beta_sweep figure: {beta_png}")

    summary_json = os.path.join(args.output_dir, f"{prefix}_summary.json")
    _save_json(summary_json, outputs)
    print(f"[DONE] summary: {summary_json}")


if __name__ == "__main__":
    main()
