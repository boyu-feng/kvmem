#!/usr/bin/env python3
"""
Analyze label coverage on the same sampled subsets used by experiments.

Datasets:
- HotpotQA
- 2Wiki
- MuSiQue

Sampling logic:
- Reuse base.select_samples with the same seed/num_samples as experiments.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

import run_all_wiki_experiments_v2 as base
import run_all_2wiki_experiments_v2 as runner_2wiki
import run_all_musique_experiments_v2 as runner_musique


def _nonempty_text(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, (list, tuple, set)):
        return any(_nonempty_text(x) for x in v)
    if isinstance(v, dict):
        return any(_nonempty_text(x) for x in v.values())
    return bool(str(v).strip())


def _ratio(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return numer / denom


def _count_hotpot_labels(samples: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    has_answer = 0
    has_supporting = 0
    total = 0
    for item in samples:
        total += 1
        if _nonempty_text(item.get("answer")):
            has_answer += 1
        if _nonempty_text(item.get("supporting_facts")):
            has_supporting += 1
    return {
        "total": total,
        "has_answer": has_answer,
        "has_supporting_facts": has_supporting,
    }


def _count_answer_labels(samples: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    has_answer = 0
    total = 0
    for item in samples:
        total += 1
        if _nonempty_text(item.get("answer")):
            has_answer += 1
    return {
        "total": total,
        "has_answer": has_answer,
    }


def _build_dataset_summary(
    dataset_name: str,
    full_counts: Dict[str, int],
    selected_counts: Dict[str, int],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "dataset": dataset_name,
        "full": {
            **full_counts,
            "answer_ratio": _ratio(full_counts.get("has_answer", 0), full_counts.get("total", 0)),
        },
        "selected": {
            **selected_counts,
            "answer_ratio": _ratio(selected_counts.get("has_answer", 0), selected_counts.get("total", 0)),
        },
    }
    if "has_supporting_facts" in full_counts:
        out["full"]["supporting_facts_ratio"] = _ratio(
            full_counts.get("has_supporting_facts", 0), full_counts.get("total", 0)
        )
    if "has_supporting_facts" in selected_counts:
        out["selected"]["supporting_facts_ratio"] = _ratio(
            selected_counts.get("has_supporting_facts", 0), selected_counts.get("total", 0)
        )
    return out


def _write_markdown(path: str, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Label Coverage Report")
    lines.append("")
    lines.append(f"- generated_at_utc: {payload['generated_at_utc']}")
    lines.append(f"- seed: {payload['seed']}")
    lines.append(f"- num_samples: {payload['num_samples']}")
    lines.append("")
    lines.append("| Dataset | Split | Total | Has Answer | Answer Ratio | Has Supporting Facts | Supporting Ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for ds in payload["datasets"]:
        for split in ("full", "selected"):
            row = ds[split]
            lines.append(
                "| {dataset} | {split} | {total} | {has_answer} | {answer_ratio:.2%} | {has_sf} | {sf_ratio} |".format(
                    dataset=ds["dataset"],
                    split=split,
                    total=row.get("total", 0),
                    has_answer=row.get("has_answer", 0),
                    answer_ratio=row.get("answer_ratio", 0.0),
                    has_sf=row.get("has_supporting_facts", "N/A"),
                    sf_ratio=(
                        "{:.2%}".format(row.get("supporting_facts_ratio", 0.0))
                        if "supporting_facts_ratio" in row
                        else "N/A"
                    ),
                )
            )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze label coverage for selected samples.")
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output_json", type=str, default="results/label_coverage_selected_samples.json")
    parser.add_argument("--output_md", type=str, default="results/label_coverage_selected_samples.md")
    parser.add_argument("--data_path_2wiki", type=str, default=runner_2wiki.DEFAULT_2WIKI_LOCAL_PATH)
    parser.add_argument("--data_path_musique", type=str, default=runner_musique.DEFAULT_MUSIQUE_LOCAL_PATH)
    args = parser.parse_args()

    base.RANDOM_SEED = int(args.seed)
    base.NUM_SAMPLES = int(args.num_samples)

    hotpot_data = base.load_hotpotqa_data()
    hotpot_selected = [x[1] for x in base.select_samples(hotpot_data)]

    data_2wiki = runner_2wiki.load_2wiki_data(args.data_path_2wiki)
    selected_2wiki = [x[1] for x in base.select_samples(data_2wiki)]

    data_musique = runner_musique.load_musique_data(args.data_path_musique)
    selected_musique = [x[1] for x in base.select_samples(data_musique)]

    hotpot_summary = _build_dataset_summary(
        "hotpotqa",
        _count_hotpot_labels(hotpot_data),
        _count_hotpot_labels(hotpot_selected),
    )
    wiki2_summary = _build_dataset_summary(
        "2wiki",
        _count_answer_labels(data_2wiki),
        _count_answer_labels(selected_2wiki),
    )
    musique_summary = _build_dataset_summary(
        "musique",
        _count_answer_labels(data_musique),
        _count_answer_labels(selected_musique),
    )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": int(args.seed),
        "num_samples": int(args.num_samples),
        "datasets": [hotpot_summary, wiki2_summary, musique_summary],
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _write_markdown(args.output_md, payload)

    print(f"[INFO] Wrote JSON report: {args.output_json}")
    print(f"[INFO] Wrote Markdown report: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
