#!/usr/bin/env python3
"""
Analyze which decode tokens are discarded under different KV pruning methods.

Analyses:
1) method_compare: H2O vs TOVA vs StepKV on the same sample.
2) beta_sweep: StepKV with different step-score weights.
3) token_score_heatmap: select sample(s), re-run H2O/TOVA/StepKV, plot score heatmap.

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
import numpy as np

import run_all_wiki_experiments_v2 as base
from models.QwenLLMWithKVCache import QwenLLMWithKVCache
from models.model_paths import resolve_local_model_path
from retrievers.WikiBM25Retriever import WikiBM25Retriever
from token_tracker import TokenTracker


METHOD_LABELS = {
    "h2o": "H2O",
    "tova": "TOVA",
    "step_aware_h2o": "StepKV",
}

SCORE_METHODS = ["h2o", "tova", "step_aware_h2o"]


def _max_decode_len(debug_payload: Dict[str, Any]) -> int:
    prompt_token_count = int(debug_payload.get("prompt_token_count", 0) or 0)
    max_end = prompt_token_count
    for rng in (debug_payload.get("step_token_ranges", {}) or {}).values():
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            max_end = max(max_end, int(rng[1]))
    return max(0, max_end - prompt_token_count)


def _latest_token_score_snapshot(debug_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    history = debug_payload.get("pruning_history", []) or []
    for ev in reversed(history):
        if isinstance(ev, dict) and isinstance(ev.get("token_score_snapshot"), dict):
            return ev["token_score_snapshot"]
    return None


def extract_decode_token_scores(debug_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Map the latest pruning score snapshot onto decode-token indices (prompt excluded)."""
    prompt_token_count = int(debug_payload.get("prompt_token_count", 0) or 0)
    decode_len = _max_decode_len(debug_payload)
    snap = _latest_token_score_snapshot(debug_payload)
    out = {
        "decode_len": int(decode_len),
        "prompt_token_count": int(prompt_token_count),
        "scores": [None] * decode_len,
        "hh_scores": [None] * decode_len,
        "step_scores": [None] * decode_len,
        "combined_scores": [None] * decode_len,
        "has_snapshot": bool(snap),
    }
    if not snap:
        return out

    prune_start = int(snap.get("prune_start", 0))
    display_scores = snap.get("display_scores") or snap.get("combined_scores") or snap.get("hh_scores") or []
    hh_scores = snap.get("hh_scores") or []
    step_scores = snap.get("step_scores") or []
    combined_scores = snap.get("combined_scores") or display_scores

    def _fill(target: List[Any], values: List[Any]) -> None:
        for i, val in enumerate(values):
            decode_idx = int(prune_start - prompt_token_count + i)
            if 0 <= decode_idx < decode_len:
                target[decode_idx] = float(val)

    _fill(out["scores"], display_scores)
    if hh_scores:
        _fill(out["hh_scores"], hh_scores)
    if step_scores:
        _fill(out["step_scores"], step_scores)
    if combined_scores:
        _fill(out["combined_scores"], combined_scores)
    return out



def _step_boundaries_from_debug(debug_payload: Dict[str, Any]) -> List[float]:
    prompt_token_count = int(debug_payload.get("prompt_token_count", 0) or 0)
    boundaries = []
    for sid_str, rng in sorted((debug_payload.get("step_token_ranges", {}) or {}).items(), key=lambda kv: int(kv[0])):
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            continue
        end_shifted = int(rng[1]) - prompt_token_count
        if end_shifted >= 0:
            boundaries.append(float(end_shifted) + 0.5)
    return boundaries


def _max_plotted_decode_index(runs: Dict[str, Dict[str, Any]]) -> int:
    """Last decode-token index that has score / discard / step-boundary data."""
    max_idx = -1
    for run in runs.values():
        debug_payload = run.get("debug_payload", {}) or {}
        if debug_payload:
            score_info = extract_decode_token_scores(debug_payload)
            for idx, val in enumerate(score_info.get("scores") or []):
                if val is not None:
                    max_idx = max(max_idx, int(idx))
        discarded = run.get("discarded", {}) or {}
        for idx in discarded.get("discarded_decode_indices", []) or []:
            max_idx = max(max_idx, int(idx))
        for bd in discarded.get("step_boundaries", []) or []:
            max_idx = max(max_idx, int(float(bd.get("x", -1))))
    return max(max_idx, 0)


def _heatmap_figsize(plot_len: int) -> float:
    return float(min(12.0, max(4.0, plot_len * 0.08)))


def _scored_decode_len(score_info: Dict[str, Any]) -> int:
    scores = score_info.get("scores") or []
    last = -1
    for idx, val in enumerate(scores):
        if val is not None:
            last = idx
    return int(last + 1)


def _heatmap_plot_len(method_runs: Dict[str, Dict[str, Any]], max_plot_tokens: Optional[int] = None) -> int:
    """Use the scored-token span, not the full decode trajectory length."""
    plot_len = 0
    for run in method_runs.values():
        debug_payload = run.get("debug_payload", {}) or {}
        score_info = extract_decode_token_scores(debug_payload)
        snap = _latest_token_score_snapshot(debug_payload) or {}
        hh = snap.get("hh_scores") or snap.get("display_scores") or []
        plot_len = max(plot_len, len(hh), _scored_decode_len(score_info))
    if max_plot_tokens is not None and int(max_plot_tokens) > 0:
        plot_len = min(plot_len, int(max_plot_tokens))
    return int(plot_len)


def _extract_attention_row(
    snapshot: Dict[str, Any],
    prompt_len: int,
    n: int,
) -> Tuple[np.ndarray, str]:
    """Get per-decode-token importance/attention row of length n."""
    prune_start = int(snapshot.get("prune_start", prompt_len))
    hh = snapshot.get("display_scores") or snapshot.get("combined_scores") or snapshot.get("hh_scores") or []
    row = np.zeros(n, dtype=float)
    if hh:
        for j in range(min(n, len(hh))):
            if hh[j] is not None:
                row[j] = float(hh[j])

    attn_raw = snapshot.get("attention_matrix")
    if attn_raw:
        attn = np.asarray(attn_raw, dtype=float)
        if attn.ndim == 2 and attn.size > 0:
            q_len, kv_len = attn.shape
            query_base = int(snapshot.get("query_base", kv_len - q_len))
            best_row = None
            best_fill = -1
            for q_rel in range(q_len):
                abs_q = query_base + q_rel
                if abs_q < prune_start:
                    continue
                vals = []
                for j in range(n):
                    abs_k = prune_start + j
                    if abs_k < kv_len:
                        vals.append(float(attn[q_rel, abs_k]))
                    else:
                        vals.append(0.0)
                fill = sum(1 for v in vals if v > 0)
                if fill > best_fill:
                    best_fill = fill
                    best_row = np.asarray(vals, dtype=float)
            if best_row is not None and best_fill > 0:
                return best_row, "attention"

    if np.any(row > 0):
        return row, "score"
    return row, "empty"


def _build_decode_attention_square(
    snapshot: Dict[str, Any],
    prompt_len: int,
    plot_len: int,
    fallback_scores: Optional[List[Any]] = None,
) -> Tuple[np.ndarray, str]:
    """
    Build a causal decode square matrix.

    Each row i shows the key-importance pattern for tokens 0..i.
    This matches common KV/attention visualizations and stays dense/contrasty.
    """
    n = int(plot_len)
    if n <= 0:
        return np.zeros((0, 0), dtype=float), "empty"

    attn_row, source = _extract_attention_row(snapshot, prompt_len, n)
    if source == "empty" and fallback_scores:
        for j, val in enumerate(fallback_scores[:n]):
            if val is not None:
                attn_row[j] = float(val)
        if np.any(attn_row > 0):
            source = "score"

    if not np.any(attn_row > 0):
        return np.zeros((n, n), dtype=float), "empty"

    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1):
            mat[i, j] = attn_row[j]
    return mat, source


def _enhance_attention_contrast(mat: np.ndarray) -> np.ndarray:
    """Log-scale + percentile stretch so differences are visible."""
    out = mat.copy()
    pos = out[np.isfinite(out) & (out > 0)]
    if pos.size == 0:
        return out
    out = np.log1p(out)
    pos = out[out > 0]
    if pos.size == 0:
        return out
    p95 = float(np.percentile(pos, 95))
    if p95 <= 0:
        p95 = float(pos.max())
    if p95 > 0:
        out = np.clip(out / p95, 0.0, 1.0)
    return out


def _build_kv_config(
    pruning_mode: str,
    cache_ratio: float = 0.5,
    beta: Optional[float] = None,
    alpha: Optional[float] = None,
    attention_viz: bool = False,
) -> Dict[str, Any]:
    obs_window_default = 0 if pruning_mode in ("step_aware_h2o", "step_inter", "tova") else 32
    attn_mode_default = "piggyback" if pruning_mode in ("step_aware_h2o", "step_inter") or attention_viz else "scoring_forward"
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


def _resolve_target_samples(
    selected: List[Tuple[int, Dict[str, Any]]],
    args,
) -> List[Tuple[int, int, Dict[str, Any]]]:
    if getattr(args, "sample_positions", None):
        out: List[Tuple[int, int, Dict[str, Any]]] = []
        for pos in args.sample_positions:
            pos = int(pos)
            if pos < 0 or pos >= len(selected):
                raise IndexError(f"sample_pos out of range: {pos} (total {len(selected)})")
            orig_idx, sample = selected[pos]
            out.append((pos, orig_idx, sample))
        return out
    pos, orig_idx, sample = _pick_sample(selected, args)
    return [(pos, orig_idx, sample)]


def _count_scored_tokens(score_info: Dict[str, Any]) -> int:
    return sum(1 for val in (score_info.get("scores") or []) if val is not None)


def _method_runs_have_scores(method_runs: Dict[str, Dict[str, Any]]) -> bool:
    for method in SCORE_METHODS:
        debug_payload = method_runs.get(method, {}).get("debug_payload", {}) or {}
        score_info = extract_decode_token_scores(debug_payload)
        if score_info.get("has_snapshot") and _count_scored_tokens(score_info) > 0:
            return True
    return False


def rerun_token_scores_for_sample(
    sample: Dict[str, Any],
    retriever,
    args,
    sample_pos: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Re-run H2O / TOVA / StepKV on one sample and collect fresh token scores."""
    pos_label = f"pos={sample_pos} " if sample_pos is not None else ""
    print(
        f"[INFO] token_score heatmap: re-running sample {pos_label}"
        f"id={sample.get('id', '')} ..."
    )
    method_runs: Dict[str, Dict[str, Any]] = {}
    for method in SCORE_METHODS:
        print(f"[INFO] token_score heatmap rerun -> {METHOD_LABELS.get(method, method)} ...")
        run = _run_one_sample(
            sample,
            retriever,
            pruning_mode=method,
            cache_ratio=float(args.cache_ratio),
            max_steps=base.MAX_STEPS,
            attention_viz=True,
        )
        score_info = extract_decode_token_scores(run.get("debug_payload", {}) or {})
        scored = _count_scored_tokens(score_info)
        decode_len = int(score_info.get("decode_len", 0) or 0)
        print(
            f"[INFO]   {method}: scored_tokens={scored}/{decode_len} "
            f"has_snapshot={bool(score_info.get('has_snapshot'))}"
        )
        method_runs[method] = run
    return method_runs


def rerun_token_scores_with_sample_search(
    args,
    selected: List[Tuple[int, Dict[str, Any]]],
    retriever,
    start_pos: int,
    sample: Dict[str, Any],
    orig_idx: int,
) -> Tuple[int, int, Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Re-run from start_pos; optionally scan forward until score snapshots exist."""
    tries = int(getattr(args, "max_auto_tries", 1))
    if not getattr(args, "auto_find_nonempty_scores", False):
        tries = 1

    chosen_pos = None
    chosen_orig_idx = None
    chosen_sample = None
    method_runs: Dict[str, Dict[str, Any]] = {}

    for off in range(max(1, tries)):
        pos = int(start_pos + off)
        if pos >= len(selected):
            break
        orig_idx, sample = selected[pos]
        print(f"[INFO] token_score heatmap candidate sample_pos={pos}, orig_idx={orig_idx}, id={sample.get('id', '')}")
        method_runs = rerun_token_scores_for_sample(
            sample,
            retriever,
            args,
            sample_pos=pos,
        )
        chosen_pos = pos
        chosen_orig_idx = orig_idx
        chosen_sample = sample
        if _method_runs_have_scores(method_runs):
            print(f"[INFO] token_score heatmap: using sample_pos={pos} with non-empty scores")
            break

    if chosen_sample is None:
        raise RuntimeError("No valid sample could be executed for token score heatmap.")

    if not _method_runs_have_scores(method_runs):
        raise RuntimeError(
            "Re-run finished but no token_score_snapshot was found. "
            "Ensure kv_cache/pruning_strategy.py exports token_score_snapshot and the episode triggers pruning."
        )

    return int(chosen_pos), int(chosen_orig_idx), chosen_sample, method_runs


def run_token_score_heatmap_analysis(
    args,
    selected: List[Tuple[int, Dict[str, Any]]],
    retriever,
    sample_pos: int,
    orig_idx: int,
    sample: Dict[str, Any],
) -> Dict[str, Any]:
    """Select sample(s) and re-run inference to build token score heatmap data."""
    pos, orig_idx, sample, method_runs = rerun_token_scores_with_sample_search(
        args,
        selected,
        retriever,
        start_pos=int(sample_pos),
        sample=sample,
        orig_idx=int(orig_idx),
    )
    token_score_matrix = {
        method: extract_decode_token_scores(method_runs[method].get("debug_payload", {}) or {})
        for method in SCORE_METHODS
    }
    return {
        "analysis": "token_score_heatmap",
        "methods": SCORE_METHODS,
        "sample_pos": int(pos),
        "orig_idx": int(orig_idx),
        "sample_id": str(sample.get("id", "")),
        "question": sample.get("question", ""),
        "gold_answer": sample.get("answer", ""),
        "method_runs_full": method_runs,
        "token_score_matrix": token_score_matrix,
        "method_runs": {
            m: {
                "predicted_answer": method_runs[m]["predicted_answer"],
                "kv_config": method_runs[m]["kv_config"],
                "token_scores": token_score_matrix[m],
            }
            for m in SCORE_METHODS
        },
    }


def _run_one_sample(
    sample: Dict[str, Any],
    retriever,
    pruning_mode: str,
    cache_ratio: float,
    max_steps,
    beta: Optional[float] = None,
    alpha: Optional[float] = None,
    attention_viz: bool = False,
) -> Dict[str, Any]:
    alpha_val = alpha
    if beta is not None and alpha_val is None:
        alpha_val = 1.0 - float(beta)
    kv_config = _build_kv_config(
        pruning_mode,
        cache_ratio=cache_ratio,
        beta=beta,
        alpha=alpha_val,
        attention_viz=attention_viz,
    )
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
) -> None:
    methods = list(method_runs.keys())
    plot_x_max = _max_plotted_decode_index(method_runs)
    fig = plt.figure(figsize=(_heatmap_figsize(plot_x_max + 1), 10))
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
        if float(bd["x"]) <= plot_x_max:
            ax_main.axvline(float(bd["x"]), linestyle="--", linewidth=0.9, color="gray", alpha=0.55)

    ax_main.set_yticks([y_positions[m] for m in methods])
    ax_main.set_yticklabels([METHOD_LABELS.get(m, m) for m in methods])
    ax_main.set_xlabel("Decode Token Index")
    ax_main.set_xlim(-0.5, plot_x_max + 0.5)
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
        ax_bar.grid(True, axis="y", alpha=0.25)
    else:
        ax_bar.axis("off")

    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_token_score_heatmap(
    method_runs: Dict[str, Dict[str, Any]],
    output_png: str,
    max_plot_tokens: Optional[int] = None,
) -> Dict[str, str]:
    """Plot one square attention-style heatmap PNG per method."""
    methods = list(method_runs.keys())
    base, ext = os.path.splitext(output_png)
    if not ext:
        ext = ".png"
    output_paths: Dict[str, str] = {}

    for method in methods:
        debug_payload = method_runs[method].get("debug_payload", {}) or {}
        score_info = extract_decode_token_scores(debug_payload)
        prompt_len = int(score_info.get("prompt_token_count", debug_payload.get("prompt_token_count", 0)) or 0)
        snap = _latest_token_score_snapshot(debug_payload) or {}
        plot_len = max(
            len(snap.get("hh_scores") or snap.get("display_scores") or []),
            _scored_decode_len(score_info),
        )
        if max_plot_tokens is not None and int(max_plot_tokens) > 0:
            plot_len = min(plot_len, int(max_plot_tokens))
        if plot_len <= 0:
            continue

        prune_start = int(snap.get("prune_start", prompt_len))
        decode_boundaries = []
        for x in _step_boundaries_from_debug(debug_payload):
            shifted = float(x) - float(prune_start - prompt_len)
            if 0 <= shifted <= plot_len:
                decode_boundaries.append(shifted)

        mat, source = _build_decode_attention_square(
            snap,
            prompt_len=prompt_len,
            plot_len=plot_len,
            fallback_scores=score_info.get("scores") or [],
        )
        mat = _enhance_attention_contrast(mat)
        if source == "empty" or not np.any(mat > 0):
            continue

        fig_side = _heatmap_figsize(plot_len)
        fig, ax = plt.subplots(figsize=(fig_side, fig_side))
        im = ax.imshow(
            mat,
            cmap="Reds",
            interpolation="nearest",
            aspect="equal",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_xlabel("Key Token Index")
        ax.set_ylabel("Query Token Index")
        ax.set_xlim(-0.5, plot_len - 0.5)
        ax.set_ylim(-0.5, plot_len - 0.5)
        for x in decode_boundaries:
            ax.axvline(x, linestyle="--", linewidth=0.6, color="black", alpha=0.35)
            ax.axhline(x, linestyle="--", linewidth=0.6, color="black", alpha=0.35)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        method_suffix = method.replace("step_aware_h2o", "stepkv")
        method_path = f"{base}_{method_suffix}{ext}"
        fig.savefig(method_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        output_paths[method] = method_path

    return output_paths


def _plot_beta_sweep(
    beta_runs: Dict[str, Dict[str, Any]],
    output_png: str,
) -> None:
    betas = sorted(beta_runs.keys(), key=lambda x: float(x), reverse=True)
    plot_x_max = _max_plotted_decode_index(beta_runs)
    fig, ax = plt.subplots(figsize=(_heatmap_figsize(plot_x_max + 1), max(4, 0.8 * len(betas) + 2)))
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
        if float(bd["x"]) <= plot_x_max:
            ax.axvline(float(bd["x"]), linestyle="--", linewidth=0.9, color="gray", alpha=0.55)

    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels([f"beta={b}" for b in betas])
    ax.set_xlabel("Decode Token Index")
    ax.set_xlim(-0.5, plot_x_max + 0.5)
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
    token_score_matrix = {}
    for method in methods:
        debug_payload = method_runs[method].get("debug_payload", {}) or {}
        token_score_matrix[method] = extract_decode_token_scores(debug_payload)
    return {
        "analysis": "method_compare",
        "methods": methods,
        "method_runs_full": method_runs,
        "method_runs": {
            m: {
                "predicted_answer": method_runs[m]["predicted_answer"],
                "kv_config": method_runs[m]["kv_config"],
                "discarded": method_runs[m]["discarded"],
                "token_scores": token_score_matrix[m],
            }
            for m in methods
        },
        "token_score_matrix": token_score_matrix,
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
        choices=["method_compare", "beta_sweep", "token_score_heatmap", "both"],
        default="both",
        help="both = method_compare + beta_sweep + token_score_heatmap",
    )
    parser.add_argument("--dataset", choices=["hotpotqa", "2wiki", "musique", "browsecomp"], default="hotpotqa")
    parser.add_argument("--sample_pos", type=int, default=0, help="Primary sample position in selected set.")
    parser.add_argument(
        "--sample_positions",
        type=int,
        nargs="+",
        default=None,
        help="Optional multiple sample positions for token_score_heatmap.",
    )
    parser.add_argument("--sample_id", type=str, default="")
    parser.add_argument(
        "--auto_find_nonempty_scores",
        action="store_true",
        help="For heatmap: scan forward from sample_pos until score snapshots exist.",
    )
    parser.add_argument(
        "--max_auto_tries",
        type=int,
        default=20,
        help="Max samples to try when --auto_find_nonempty_scores is set.",
    )
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--max_steps", type=str, default="7")
    parser.add_argument("--cache_ratio", type=float, default=0.5)
    parser.add_argument(
        "--model_path",
        type=str,
        default="auto",
        help="Local model dir, or 'auto' to use KVMEM_MODEL_PATH / hf_cache/models/Qwen2.5-7B-Instruct.",
    )
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
    parser.add_argument(
        "--heatmap_max_tokens",
        type=int,
        default=0,
        help="Optional hard cap on heatmap x-axis length (0 = auto truncate at last scored token).",
    )
    args = parser.parse_args()

    if args.canary == "":
        import run_all_browsecomp_experiments_v2 as runner_bc

        args.canary = runner_bc.DEFAULT_BROWSECOMP_PLUS_CANARY

    os.makedirs(args.output_dir, exist_ok=True)
    args.model_path = resolve_local_model_path(args.model_path)
    base.MODEL_PATH = args.model_path
    print(f"[INFO] Analysis model (local): {base.MODEL_PATH}")
    selected, retriever = _prepare_dataset(args)
    target_samples = _resolve_target_samples(selected, args)
    sample_pos, orig_idx, sample = target_samples[0]

    meta = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": args.dataset,
        "sample_pos": int(sample_pos),
        "sample_positions": [int(p) for p, _, _ in target_samples],
        "orig_idx": int(orig_idx),
        "sample_id": str(sample.get("id", "")),
        "question": sample.get("question", ""),
        "gold_answer": sample.get("answer", ""),
        "seed": int(args.seed),
        "num_samples": int(args.num_samples),
        "max_steps": base.format_max_steps(base.parse_max_steps(args.max_steps)),
        "cache_ratio": float(args.cache_ratio),
        "model_path": args.model_path,
        "auto_find_nonempty_scores": bool(args.auto_find_nonempty_scores),
    }

    outputs: Dict[str, Any] = {"meta": meta, "analyses": {}}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{args.dataset}_sample{sample_pos}_{stamp}"

    if args.analysis in ("method_compare", "both"):
        cmp_payload = run_method_compare(args, selected, retriever, sample_pos, orig_idx, sample)
        cmp_payload_for_json = dict(cmp_payload)
        cmp_payload_for_json.pop("method_runs_full", None)
        outputs["analyses"]["method_compare"] = cmp_payload_for_json
        cmp_json = os.path.join(args.output_dir, f"{prefix}_method_compare.json")
        cmp_png = os.path.join(args.output_dir, f"{prefix}_method_compare.png")
        _save_json(cmp_json, cmp_payload_for_json)
        _plot_method_compare(
            cmp_payload["method_runs_full"],
            cmp_payload["overlap"],
            cmp_png,
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

    if args.analysis in ("token_score_heatmap", "both"):
        heatmap_outputs: List[Dict[str, Any]] = []
        for pos, oidx, smp in target_samples:
            heatmap_payload = run_token_score_heatmap_analysis(
                args, selected, retriever, pos, oidx, smp
            )
            heatmap_payload_for_json = dict(heatmap_payload)
            heatmap_payload_for_json.pop("method_runs_full", None)
            heatmap_outputs.append(heatmap_payload_for_json)

            used_pos = int(heatmap_payload["sample_pos"])
            sample_prefix = f"{args.dataset}_sample{used_pos}_{stamp}"
            heatmap_json = os.path.join(args.output_dir, f"{sample_prefix}_token_score_heatmap.json")
            heatmap_png_base = os.path.join(args.output_dir, f"{sample_prefix}_token_score_heatmap.png")
            _save_json(heatmap_json, heatmap_payload_for_json)
            heatmap_paths = _plot_token_score_heatmap(
                heatmap_payload["method_runs_full"],
                heatmap_png_base,
                max_plot_tokens=(int(args.heatmap_max_tokens) if int(args.heatmap_max_tokens) > 0 else None),
            )
            heatmap_payload_for_json["plot_token_len"] = _heatmap_plot_len(
                heatmap_payload["method_runs_full"],
                max_plot_tokens=(int(args.heatmap_max_tokens) if int(args.heatmap_max_tokens) > 0 else None),
            )
            heatmap_payload_for_json["figure_paths"] = heatmap_paths
            _save_json(heatmap_json, heatmap_payload_for_json)
            print(f"[DONE] token_score heatmap JSON: {heatmap_json}")
            for method, path in heatmap_paths.items():
                print(f"[DONE] token_score heatmap ({METHOD_LABELS.get(method, method)}): {path}")

        outputs["analyses"]["token_score_heatmap"] = (
            heatmap_outputs[0] if len(heatmap_outputs) == 1 else heatmap_outputs
        )

    if args.analysis in ("beta_sweep", "both"):
        beta_payload = run_beta_sweep(args, selected, retriever, sample_pos, orig_idx, sample)
        outputs["analyses"]["beta_sweep"] = beta_payload
        beta_json = os.path.join(args.output_dir, f"{prefix}_beta_sweep.json")
        beta_png = os.path.join(args.output_dir, f"{prefix}_beta_sweep.png")
        _save_json(beta_json, beta_payload)
        _plot_beta_sweep(beta_payload["beta_runs"], beta_png)
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
