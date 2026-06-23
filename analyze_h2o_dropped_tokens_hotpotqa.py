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
        "cache_ratio": float(args.cache_ratio),
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
    token_tracker = debug_payload.get("token_tracker", {}) or {}
    step_pruning_events = token_tracker.get("step_pruning_events", {}) or {}

    event_rows = []
    scatter_points = []
    event_id = 0
    owner_rows = {}

    # Build step span lookup for "token belongs to which step".
    step_ranges = []
    for sid_str, rng in sorted(step_token_ranges.items(), key=lambda kv: int(kv[0])):
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            continue
        sid = int(sid_str)
        s, e = int(rng[0]), int(rng[1])
        if e >= s:
            step_ranges.append((sid, s, e))

    def _owner_step(global_id):
        gid = int(global_id)
        for sid, s, e in step_ranges:
            if s <= gid <= e:
                return sid
        return -1

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

    # Final-view points: use token_tracker global dropped IDs,
    # then classify each dropped token by the step span it belongs to.
    final_points = []
    final_event_rows = []
    final_event_id = 0
    for prune_step_str, dropped_ids in sorted(step_pruning_events.items(), key=lambda kv: int(kv[0])):
        prune_step = int(prune_step_str)
        dropped_ids = sorted(set(int(x) for x in (dropped_ids or [])))
        dropped_ids = [gid for gid in dropped_ids if gid >= prompt_token_count]
        if not dropped_ids:
            continue
        final_event_id += 1
        owner_counts = {}
        shifted_ids = []
        for gid in dropped_ids:
            shifted_x = int(gid - prompt_token_count)
            shifted_ids.append(shifted_x)
            owner = int(_owner_step(gid))
            owner_counts[owner] = int(owner_counts.get(owner, 0) + 1)
            key = str(owner)
            if key not in owner_rows:
                owner_rows[key] = {"owner_step": owner, "dropped_count": 0}
            owner_rows[key]["dropped_count"] = int(owner_rows[key]["dropped_count"] + 1)
            final_points.append(
                {
                    "event_id": int(final_event_id),
                    "prune_step": int(prune_step),
                    "owner_step": int(owner),
                    "x": shifted_x,
                    "global_id": int(gid),
                }
            )
        final_event_rows.append(
            {
                "event_id": int(final_event_id),
                "prune_step": int(prune_step),
                "tokens_evicted": int(len(shifted_ids)),
                "evicted_abs_indices_no_prefill": shifted_ids,
                "owner_step_counts": {str(int(k)): int(v) for k, v in sorted(owner_counts.items())},
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
        "final_events": final_event_rows,
        "final_points": final_points,
        "dropped_by_owner_step": [owner_rows[k] for k in sorted(owner_rows.keys(), key=lambda x: int(x))],
        "step_boundaries": step_boundaries,
    }


def _plot_three_layers(plot_data, layer_ids, output_png):
    # Use final-view points by default: dropped tokens after whole reasoning,
    # classified by owner step spans.
    points = plot_data.get("final_points", []) or plot_data.get("points", [])
    boundaries = plot_data["step_boundaries"]
    step_key = "owner_step" if points and "owner_step" in points[0] else "react_step"
    steps = sorted(set(int(p[step_key]) for p in points))
    cmap = plt.get_cmap("tab10")
    step_to_color = {s: cmap(i % 10) for i, s in enumerate(steps)}

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax_idx, ax in enumerate(axes):
        if points:
            for step in steps:
                xs = [p["x"] for p in points if int(p[step_key]) == step]
                ys = [p["event_id"] for p in points if int(p[step_key]) == step]
                if not xs:
                    continue
                if step_key == "owner_step":
                    label_text = f"owner_step{step}"
                else:
                    label_text = f"step{step}"
                label = label_text if ax_idx == 0 else None
                ax.scatter(xs, ys, s=12, alpha=0.85, c=[step_to_color[step]], label=label)
        else:
            ax.text(
                0.5,
                0.5,
                "No dropped-token points under current config",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="gray",
            )

        for bd in boundaries:
            ax.axvline(float(bd["x"]), linestyle="--", linewidth=1.0, color="gray", alpha=0.7)

        ax.set_ylabel("Prune Event")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Key Position Index (No Prefill)")
    if steps:
        axes[0].legend(loc="upper right", ncol=min(6, max(1, len(steps))))
    plt.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze and plot dropped tokens for H2O on HotpotQA.")
    parser.add_argument("--sample_pos", type=int, default=0, help="Position in shuffled selected samples.")
    parser.add_argument("--auto_find_nonempty", action="store_true", help="Auto scan next samples until non-empty dropped points are found.")
    parser.add_argument("--max_auto_tries", type=int, default=20, help="Max samples to try when --auto_find_nonempty is set.")
    parser.add_argument("--max_steps", type=int, default=12)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--bm25_top_k", type=int, default=5)
    parser.add_argument("--wiki_index_dir", type=str, default=base.WIKI_INDEX_DIR)
    parser.add_argument("--output_dir", type=str, default="results/h2o_drop_analysis")
    parser.add_argument("--cache_ratio", type=float, default=0.5)
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

    retriever = WikiBM25Retriever(index_dir=args.wiki_index_dir, load_corpus=True)
    kv_config = _build_kv_config(args)

    start_pos = int(args.sample_pos)
    tries = int(args.max_auto_tries) if args.auto_find_nonempty else 1
    chosen_pos = None
    chosen_orig_idx = None
    chosen_sample = None
    pred_answer = ""
    trajectory_log = []
    step_timings = []
    debug_payload = {}
    plot_data = {"prompt_token_count": 0, "events": [], "points": [], "step_boundaries": []}

    for off in range(max(1, tries)):
        pos = start_pos + off
        if pos >= len(selected_samples):
            break
        orig_idx, sample = selected_samples[pos]
        print(f"[INFO] Trying sample_pos={pos}, orig_idx={orig_idx}, id={sample['id']}")

        token_tracker = TokenTracker()
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

        plot_data = _extract_plot_data(debug_payload)
        chosen_pos = pos
        chosen_orig_idx = orig_idx
        chosen_sample = sample
        if plot_data["points"]:
            print(f"[INFO] Found non-empty dropped-token points at sample_pos={pos}")
            break

    if chosen_sample is None:
        raise RuntimeError("No valid sample could be executed.")

    layer_ids = _parse_layers(args.layers, model_layers=32)
    if len(layer_ids) < 3:
        while len(layer_ids) < 3:
            layer_ids.append(layer_ids[-1] if layer_ids else 0)
    if not plot_data["points"]:
        print(
            "[WARN] No dropped-token points found in selected sample range. "
            "Output files will still be generated."
        )

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
            "sample_pos": int(chosen_pos),
            "requested_sample_pos": int(args.sample_pos),
            "orig_idx": int(chosen_orig_idx),
            "sample_id": chosen_sample["id"],
            "question": chosen_sample["question"],
            "gold_answer": chosen_sample["answer"],
            "predicted_answer": pred_answer,
            "max_steps": int(args.max_steps),
            "kv_config": kv_config,
            "layers_for_plot": layer_ids,
            "auto_find_nonempty": bool(args.auto_find_nonempty),
            "max_auto_tries": int(args.max_auto_tries),
            "nonempty_points_found": bool(len(plot_data["points"]) > 0),
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
