#!/usr/bin/env python3
"""
Online intervention ablation for StepKV on HotpotQA.

Protocol:
1) Baseline run (step_aware_h2o) with cache_ratio in {0.2, 0.5}.
2) Read per-sample final step scores.
3) Re-run with force-drop of Top-1 scored step (drop entire step span tokens).
4) Report EM/F1 drop.
"""

import argparse
import json
import os
import random
from datetime import datetime, timezone
from typing import Any, Dict, List

import run_all_wiki_experiments_v2 as base


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _extract_drop_map(result_json: str, mode: str = "top1", seed: int = 233) -> Dict[str, List[int]]:
    data = _load_json(result_json)
    rows = data.get("results", []) if isinstance(data, dict) else []
    rnd = random.Random(seed)
    out: Dict[str, List[int]] = {}
    for r in rows:
        sid = str(r.get("id", ""))
        sc = r.get("step_scores", {}) or r.get("debug_payload", {}).get("step_scores", {})
        pairs = []
        for k, v in sc.items():
            try:
                pairs.append((int(k), float(v)))
            except Exception:
                continue
        if not pairs or not sid:
            continue
        if mode == "top1":
            pick = max(pairs, key=lambda x: x[1])[0]
        elif mode == "bottom1":
            pick = min(pairs, key=lambda x: x[1])[0]
        elif mode == "random1":
            pick = rnd.choice([x[0] for x in pairs])
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        out[sid] = [int(pick)]
    return out


def _run_once(
    selected_samples,
    retriever,
    out_json: str,
    ckpt_json: str,
    cache_ratio: float,
    drop_map: Dict[str, List[int]] = None,
):
    kv_override = {
        "cache_ratio": float(cache_ratio),
        "attn_mode": "piggyback",
        "observation_window": 0,
        "step_poolwise_prune": True,
        "step_force_drop_map": drop_map or {},
    }
    return base.run_react_kv_experiment(
        val_data=None,
        selected_samples=selected_samples,
        retriever=retriever,
        pruning_mode="step_aware_h2o",
        output_path=out_json,
        checkpoint_path=ckpt_json,
        kv_config_override=kv_override,
        metrics_dataset="hotpotqa",
    )


def main():
    parser = argparse.ArgumentParser(description="Run StepKV drop-step online ablation on HotpotQA.")
    parser.add_argument("--output_root", default="results/stepkv_drop_ablation_hotpot", type=str)
    parser.add_argument("--num_samples", default=500, type=int)
    parser.add_argument("--seed", default=233, type=int)
    parser.add_argument("--drop_mode", default="top1", choices=["top1", "bottom1", "random1"])
    parser.add_argument("--model_path", default="/root/autodl-tmp/hf_cache/models/Meta-Llama-3.1-8B-Instruct", type=str)
    args = parser.parse_args()

    base.NUM_SAMPLES = int(args.num_samples)
    base.RANDOM_SEED = int(args.seed)
    base.MODEL_PATH = args.model_path

    val_data = base.load_hotpotqa_data()
    selected_samples = base.select_samples(val_data)
    from retrievers.WikiBM25Retriever import WikiBM25Retriever
    retriever = WikiBM25Retriever(index_dir=base.WIKI_INDEX_DIR, load_corpus=True)

    summary: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": int(args.seed),
        "num_samples": int(args.num_samples),
        "drop_mode": args.drop_mode,
        "ratios": {},
    }

    for ratio in (0.2, 0.5):
        tag = f"r{int(ratio*100)}"
        base_dir = os.path.join(args.output_root, tag, "baseline")
        drop_dir = os.path.join(args.output_root, tag, f"drop_{args.drop_mode}")
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(drop_dir, exist_ok=True)

        base_json = os.path.join(base_dir, "stepaware_baseline.json")
        base_ckpt = os.path.join(base_dir, "stepaware_baseline_checkpoint.json")
        drop_json = os.path.join(drop_dir, "stepaware_drop.json")
        drop_ckpt = os.path.join(drop_dir, "stepaware_drop_checkpoint.json")

        em_b, f1_b, t_b = _run_once(selected_samples, retriever, base_json, base_ckpt, ratio, drop_map=None)
        drop_map = _extract_drop_map(base_json, mode=args.drop_mode, seed=args.seed)
        em_d, f1_d, t_d = _run_once(selected_samples, retriever, drop_json, drop_ckpt, ratio, drop_map=drop_map)

        summary["ratios"][tag] = {
            "cache_ratio": ratio,
            "baseline": {"em": em_b, "f1": f1_b, "time_s": t_b},
            "drop": {"em": em_d, "f1": f1_d, "time_s": t_d},
            "delta": {"em": em_d - em_b, "f1": f1_d - f1_b},
            "drop_map_size": len(drop_map),
        }

    out_summary = os.path.join(args.output_root, "summary.json")
    _save_json(out_summary, summary)
    print(f"[INFO] Ablation summary saved: {out_summary}")


if __name__ == "__main__":
    main()
