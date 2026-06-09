"""
Standalone dataset sampler.

Usage example:
  python sample_dataset_only.py --dataset hotpotqa --num_samples 500 \
    --sampling_mode stratified --category_field type \
    --output_dir results/sampling
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import run_all_wiki_experiments_v2 as base
import run_all_2wiki_experiments_v2 as wiki2
import run_all_musique_experiments_v2 as musique
import run_all_browsecomp_experiments_v2 as browsecomp


DEFAULT_CATEGORY_FIELDS = {
    "hotpotqa": "type",
    "2wiki": "type",
    "musique": "question_type",
    "browsecomp": "domain",
}


def _load_dataset(dataset: str, data_path: Optional[str]) -> List[Dict[str, Any]]:
    if dataset == "hotpotqa":
        if data_path:
            with open(data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        local_path = os.path.join("data", "dev.json")
        if os.path.exists(local_path):
            with open(local_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return base.load_hotpotqa_data()
    if dataset == "2wiki":
        local_path = data_path or wiki2.DEFAULT_2WIKI_LOCAL_PATH
        if local_path and os.path.exists(local_path):
            return wiki2.load_2wiki_data(local_path)
        raise RuntimeError(
            "2Wiki local file not found. Please provide --data_path (e.g. data/2wiki/dev.json)."
        )
    if dataset == "musique":
        local_path = data_path or musique.DEFAULT_MUSIQUE_LOCAL_PATH
        if local_path and os.path.exists(local_path):
            return musique.load_musique_data(local_path)
        raise RuntimeError(
            "Musique local file not found. Please provide --data_path (e.g. data/musique/dev.json)."
        )
    if dataset == "browsecomp":
        local_path = data_path or browsecomp.DEFAULT_BROWSECOMP_LOCAL_PATH
        if local_path and os.path.exists(local_path):
            return browsecomp.load_browsecomp_data(local_path)
        raise RuntimeError(
            "BrowseComp local file not found. Please provide --data_path."
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample datasets without running experiments")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["hotpotqa", "2wiki", "musique", "browsecomp"],
                        help="Which dataset to sample")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Optional local data path for non-hotpotqa datasets")
    parser.add_argument("--output_dir", type=str, default="results/sampling",
                        help="Directory to save sampling outputs")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of samples to select")
    parser.add_argument("--seed", type=int, default=233,
                        help="Random seed for sampling")
    parser.add_argument("--sampling_mode", type=str, default="stratified",
                        choices=["random", "stratified"],
                        help="Sampling mode: random or stratified")
    parser.add_argument("--category_field", type=str, default=None,
                        help="Category field for stratified sampling")
    parser.add_argument("--selection_output", type=str, default=None,
                        help="Path to save selected samples JSON")
    parser.add_argument("--selection_stats_output", type=str, default=None,
                        help="Path to save category counts JSON")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    category_field = args.category_field or DEFAULT_CATEGORY_FIELDS.get(args.dataset, "type")
    selection_output = args.selection_output or os.path.join(
        args.output_dir, f"selected_samples_{args.dataset}.json"
    )
    stats_output = args.selection_stats_output or os.path.join(
        args.output_dir, f"selected_counts_{args.dataset}.json"
    )

    val_data = _load_dataset(args.dataset, args.data_path)

    base.select_samples(
        val_data,
        num_samples=args.num_samples,
        seed=args.seed,
        sampling_mode=args.sampling_mode,
        category_field=category_field,
        selection_output_path=selection_output,
        stats_output_path=stats_output,
    )


if __name__ == "__main__":
    main()
