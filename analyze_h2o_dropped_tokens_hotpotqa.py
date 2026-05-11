import argparse
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_all_wiki_experiments_v2 as base
from models.QwenLLMWithKVCache import QwenLLMWithKVCache
from retrievers.WikiBM25Retriever import WikiBM25Retriever
from token_tracker import TokenTracker


def _parse_layers(text, model_layers=32):
    raw = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            raw.append(int(part))
        except ValueError:
            continue
    if not raw:
        raw = [0, model_layers // 2, max(0, model_layers - 1)]
    return raw[:3]


def _build_kv_config(args):
    return {
        "pruning_mode": "h2o",
        "prune_every_n": 1,
        "keep_ratio": float(args.keep_ratio),
        "target_cache_ratio": float(args.target_cache_ratio),
        "protect_prompt": bool(args.protect_prompt),
        "pool_window": 4,
        "max_trajectory_tokens": 1024,
        "sink_size": 4,
        "observation_window": int(args.observation_window),
        "num_score_layers": int(args.num_score_layers),
        "attn_mode": "scoring_forward",
    }


def _extract_plot_data(debug_payload):
    prompt_token_count = int(debug_payload.get("prompt_token_count", 0))
    pruning_history = debug_payload.get("pruning_history", []) or []
    step_token_ranges = debug_payload.get("step_token_ranges", {}) or {}

    event_rows = []
    scatter_points = []
    event_id = 0
    for ev in pruning_history:
        if not isinstance(ev, dict):
            continue
        if ev.get("mode") != "h2o":
            continue
        dropped = ev.get("evicted_abs_indices", []) or []
        if not dropped:
            continue
        # Exclude prompt/prefill region from visualization.
        dropped_no_prefill = [int(x) for x in dropped if int(x) >= prompt_token_count]
        if not dropped_no_prefill:
            continue

        event_id += 1
        react_step = ev.get("react_step")
        react_step = int(react_step) if react_step is not None else -1
        shifted = [int(x - prompt_token_count) for x in dropped_no_prefill]

        row = {
            "event_id": int(event_id),
            "react_step": int(react_step),
            "tokens_evicted": int(len(shifted)),
            "evicted_abs_indices_no_prefill": shifted,
            "cache_before": int(ev.get("cache_before", 0)),
            "new_total_len": int(ev.get("new_total_len", 0)),
            "single_token_mode": bool(ev.get("single_token_mode", False)),
        }
        event_rows.append(row)

        for x in shifted:
            scatter_points.append(
                {
                    "event_id": int(event_id),
                    "react_step": int(react_step),
                    "x": int(x),
                }
            )

    step_boundaries = []
    for sid_str, rng in sorted(step_token_ranges.items(), key=lambda kv: int(kv[0])):
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            continue
        sid = int(sid_str)
        end_abs = int(rng[1])
        end_shifted = end_abs - prompt_token_count
        if end_shifted >= 0:
            step_boundaries.append({"step": sid, "x": float(end_shifted) + 0.5})

    return {
        "prompt_token_count": prompt_token_count,
        "events": event_rows,
        "points": scatter_points,
        "step_boundaries": step_boundaries,
    }


def _plot_three_layers(plot_data, layer_ids, output_png):
    points = plot_data["points"]
    boundaries = plot_data["step_boundaries"]
    if not points:
        raise RuntimeError("No dropped-token points found. Try a harder sample or lower keep_ratio.")

    steps = sorted(set(int(p["react_step"]) for p in points))
    cmap = plt.get_cmap("tab10")
    step_to_color = {s: cmap(i % 10) for i, s in enumerate(steps)}

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax_idx, ax in enumerate(axes):
        for step in steps:
            xs = [p["x"] for p in points if int(p["react_step"]) == step]
            ys = [p["event_id"] for p in points if int(p["react_step"]) == step]
            if not xs:
                continue
            label = f"step{step}" if ax_idx == 0 else None
            ax.scatter(xs, ys, s=12, alpha=0.85, c=[step_to_color[step]], label=label)

        for bd in boundaries:
            ax.axvline(float(bd["x"]), linestyle="--", linewidth=1.0, color="gray", alpha=0.7)

        ax.set_ylabel("Prune Event")
        ax.set_title(f"Layer {layer_ids[ax_idx]}")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Key Position Index (No Prefill)")
    axes[0].legend(loc="upper right", ncol=min(6, max(1, len(steps))))
    fig.suptitle("HotpotQA H2O Dropped Tokens (Step Boundaries as Dashed Lines)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze and plot dropped tokens for H2O on HotpotQA.")
    parser.add_argument("--sample_pos", type=int, default=0, help="Position in shuffled selected samples.")
    parser.add_argument("--max_steps", type=int, default=12)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--bm25_top_k", type=int, default=5)
    parser.add_argument("--wiki_index_dir", type=str, default=base.WIKI_INDEX_DIR)
    parser.add_argument("--output_dir", type=str, default="results/h2o_drop_analysis")
    parser.add_argument("--keep_ratio", type=float, default=0.5)
    parser.add_argument("--target_cache_ratio", type=float, default=0.5)
    parser.add_argument("--protect_prompt", action="store_true")
    parser.add_argument("--observation_window", type=int, default=32)
    parser.add_argument("--num_score_layers", type=int, default=3)
    parser.add_argument("--layers", type=str, default="0,13,31")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base.NUM_SAMPLES = int(args.num_samples)
    base.RANDOM_SEED = int(args.seed)
    base.MAX_STEPS = int(args.max_steps)
    base.BM25_TOP_K = int(args.bm25_top_k)
    base.WIKI_INDEX_DIR = args.wiki_index_dir

    if not os.path.exists(args.wiki_index_dir):
        raise FileNotFoundError(f"Wiki index not found: {args.wiki_index_dir}")

    print("[INFO] Loading HotpotQA and selecting samples...")
    val_data = base.load_hotpotqa_data()
    selected_samples = base.select_samples(val_data)
    if args.sample_pos < 0 or args.sample_pos >= len(selected_samples):
        raise IndexError(f"--sample_pos out of range: {args.sample_pos} (total {len(selected_samples)})")

    orig_idx, sample = selected_samples[args.sample_pos]
    print(f"[INFO] Selected sample_pos={args.sample_pos}, orig_idx={orig_idx}, id={sample['id']}")

    retriever = WikiBM25Retriever(index_dir=args.wiki_index_dir, load_corpus=True)
    token_tracker = TokenTracker()
    kv_config = _build_kv_config(args)
    llm = QwenLLMWithKVCache(base.MODEL_PATH, kv_config, token_tracker=token_tracker)

    try:
        pred_answer, trajectory_log, step_timings, debug_payload = base._run_react_kv_episode(
            sample["question"],
            llm,
            retriever,
            pruning_mode="h2o",
            max_steps=int(args.max_steps),
            return_debug=True,
        )
    finally:
        del llm

    layer_ids = _parse_layers(args.layers, model_layers=32)
    if len(layer_ids) < 3:
        while len(layer_ids) < 3:
            layer_ids.append(layer_ids[-1] if layer_ids else 0)

    plot_data = _extract_plot_data(debug_payload)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"hotpot_h2o_sample{args.sample_pos}_{stamp}"
    json_path = os.path.join(args.output_dir, f"{prefix}.json")
    png_path = os.path.join(args.output_dir, f"{prefix}.png")
    svg_path = os.path.join(args.output_dir, f"{prefix}.svg")
    points_jsonl_path = os.path.join(args.output_dir, f"{prefix}_points.jsonl")
    plot_error_path = os.path.join(args.output_dir, f"{prefix}_plot_error.txt")

    output_blob = {
        "meta": {
            "created_at": stamp,
            "sample_pos": int(args.sample_pos),
            "orig_idx": int(orig_idx),
            "sample_id": sample["id"],
            "question": sample["question"],
            "gold_answer": sample["answer"],
            "predicted_answer": pred_answer,
            "max_steps": int(args.max_steps),
            "kv_config": kv_config,
            "layers_for_plot": layer_ids,
        },
        "trajectory": trajectory_log,
        "step_timings": step_timings,
        "plot_data": plot_data,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_blob, f, ensure_ascii=False, indent=2)

    with open(points_jsonl_path, "w", encoding="utf-8") as f:
        for p in plot_data["points"]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    try:
        _plot_three_layers(plot_data, layer_ids, png_path)
        # Also save an SVG for notebook/file-browser preview compatibility.
        _plot_three_layers(plot_data, layer_ids, svg_path)
    except Exception as e:
        with open(plot_error_path, "w", encoding="utf-8") as f:
            f.write(str(e))
        print(f"[WARN] Plot generation failed, see: {plot_error_path}")

    print(f"[DONE] Data saved: {json_path}")
    print(f"[DONE] Point data saved: {points_jsonl_path}")
    print(f"[DONE] Figure saved: {png_path}")
    print(f"[DONE] Figure saved: {svg_path}")


if __name__ == "__main__":
    main()
