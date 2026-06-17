#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timezone


def main() -> int:
    parser = argparse.ArgumentParser(description="Record EM/F1/time metrics from result JSON.")
    parser.add_argument("--result_json", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--method", required=True, type=str)
    parser.add_argument("--output_file", required=True, type=str)
    parser.add_argument("--cache_ratio", type=str, default="")
    args = parser.parse_args()

    if not os.path.exists(args.result_json):
        print(f"[WARN] Result file not found, skip metrics record: {args.result_json}")
        return 0

    with open(args.result_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        f"# Experiment Metrics ({args.dataset})",
        "",
        f"- recorded_at_utc: {ts}",
        f"- dataset: {args.dataset}",
        f"- method: {args.method}",
    ]
    if args.cache_ratio:
        lines.append(f"- cache_ratio: {args.cache_ratio}")

    lines.extend(
        [
            f"- n_samples: {summary.get('n_samples', 'N/A')}",
            f"- EM: {summary.get('exact_match', 'N/A')}",
            f"- F1: {summary.get('f1_score', 'N/A')}",
            f"- total_time_seconds: {summary.get('total_time_seconds', 'N/A')}",
            f"- avg_time_per_sample: {summary.get('avg_time_per_sample', 'N/A')}",
            f"- avg_peak_memory_mb: {summary.get('avg_peak_memory_mb', 'N/A')}",
            f"- max_peak_memory_mb: {summary.get('max_peak_memory_mb', 'N/A')}",
            "",
            "## Full Summary JSON",
            "```json",
            json.dumps(summary, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[INFO] Metrics recorded: {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
