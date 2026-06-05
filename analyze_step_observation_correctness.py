import argparse
import json
import re
from typing import Any, Dict, List, Tuple

import run_all_wiki_experiments_v2 as hotpot_base
import run_all_2wiki_experiments_v2 as wiki2_base


def _normalize(text: str) -> str:
    s = (text or "").lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _token_f1(pred: str, gold: str) -> float:
    p = _normalize(pred).split()
    g = _normalize(gold).split()
    if not p or not g:
        return float(p == g)
    from collections import Counter

    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap <= 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2.0 * precision * recall / (precision + recall)


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _collect_strings(obj: Any, out: List[str]) -> None:
    if obj is None:
        return
    if isinstance(obj, str):
        s = obj.strip()
        if s:
            out.append(s)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_strings(v, out)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_strings(v, out)


def _extract_hotpot_supporting_sentences(item: Dict[str, Any]) -> List[str]:
    facts: List[str] = []
    context = item.get("context")
    supporting = item.get("supporting_facts")

    # Hotpot canonical format:
    # context: [[title, [sent0, sent1, ...]], ...]
    # supporting_facts: {"title": [...], "sent_id": [...]} or [[title, sent_id], ...]
    if isinstance(context, list) and supporting is not None:
        title_to_sents: Dict[str, List[str]] = {}
        for row in context:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            title = row[0]
            sents = row[1]
            if isinstance(title, str) and isinstance(sents, list):
                title_to_sents[title] = [str(x) for x in sents]

        pairs: List[Tuple[str, int]] = []
        if isinstance(supporting, dict):
            titles = supporting.get("title", [])
            sent_ids = supporting.get("sent_id", [])
            if isinstance(titles, list) and isinstance(sent_ids, list):
                for t, sid in zip(titles, sent_ids):
                    if isinstance(t, str):
                        pairs.append((t, _safe_int(sid, -1)))
        elif isinstance(supporting, list):
            for row in supporting:
                if isinstance(row, (list, tuple)) and len(row) >= 2 and isinstance(row[0], str):
                    pairs.append((row[0], _safe_int(row[1], -1)))

        for title, sid in pairs:
            if sid < 0:
                continue
            sents = title_to_sents.get(title)
            if not sents:
                continue
            if sid < len(sents):
                sent = (sents[sid] or "").strip()
                if sent:
                    facts.append(sent)

    return facts


def _extract_gold_facts(item: Dict[str, Any], dataset: str) -> List[str]:
    facts: List[str] = []

    if dataset == "hotpot":
        facts.extend(_extract_hotpot_supporting_sentences(item))

    # Generic fallbacks for 2Wiki and other formats.
    for key in (
        "supporting_facts",
        "supporting_sentences",
        "supporting_paragraphs",
        "evidence",
        "evidences",
        "facts",
        "gold_facts",
    ):
        _collect_strings(item.get(key), facts)

    # Deduplicate while preserving order.
    seen = set()
    dedup: List[str] = []
    for f in facts:
        nf = _normalize(f)
        if not nf or nf in seen:
            continue
        seen.add(nf)
        dedup.append(f.strip())
    return dedup


def _load_dataset_rows(dataset: str, data_path: str = "") -> List[Dict[str, Any]]:
    if dataset == "hotpot":
        rows = hotpot_base.load_hotpotqa_data()
        return [dict(rows[i]) for i in range(len(rows))]
    if dataset == "2wiki":
        rows = wiki2_base.load_2wiki_data(data_path if data_path else None)
        return [dict(x) for x in rows]
    raise ValueError(f"Unsupported dataset: {dataset}")


def _parse_result_spec(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"Invalid --result '{spec}', expected label=path")
    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Invalid --result '{spec}', expected label=path")
    return label, path


def _compute_metrics(
    result_json_path: str,
    dataset_rows: List[Dict[str, Any]],
    dataset: str,
    threshold: float,
) -> Dict[str, Any]:
    with open(result_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    results = payload.get("results", []) or []

    step_totals: Dict[int, int] = {}
    step_correct: Dict[int, int] = {}
    n_samples_with_obs = 0
    n_samples_useful = 0
    total_obs = 0
    total_correct = 0
    missing_gold = 0

    for row in results:
        idx = _safe_int(row.get("index"), -1)
        if idx < 0 or idx >= len(dataset_rows):
            continue
        gold_item = dataset_rows[idx]
        gold_facts = _extract_gold_facts(gold_item, dataset)
        if not gold_facts:
            missing_gold += 1

        trajectory = row.get("trajectory", []) or []
        sample_has_obs = False
        sample_has_correct = False

        for step in trajectory:
            obs = (step.get("observation", "") or "").strip()
            step_id = _safe_int(step.get("step"), -1)
            if step_id <= 0 or not obs:
                continue
            sample_has_obs = True
            total_obs += 1
            step_totals[step_id] = step_totals.get(step_id, 0) + 1

            best = 0.0
            for fact in gold_facts:
                best = max(best, _token_f1(obs, fact))
            is_correct = best >= threshold
            if is_correct:
                total_correct += 1
                step_correct[step_id] = step_correct.get(step_id, 0) + 1
                sample_has_correct = True

        if sample_has_obs:
            n_samples_with_obs += 1
            if sample_has_correct:
                n_samples_useful += 1

    soc_at_step = {
        str(k): (step_correct.get(k, 0) / v if v > 0 else 0.0)
        for k, v in sorted(step_totals.items())
    }
    soc_avg = (total_correct / total_obs) if total_obs > 0 else 0.0
    useful_obs_rate = (n_samples_useful / n_samples_with_obs) if n_samples_with_obs > 0 else 0.0

    return {
        "n_results": len(results),
        "n_samples_with_observation": n_samples_with_obs,
        "n_samples_useful_observation": n_samples_useful,
        "n_missing_gold_facts": missing_gold,
        "total_observations": total_obs,
        "total_correct_observations": total_correct,
        "soc_at_step": soc_at_step,
        "soc_avg": soc_avg,
        "useful_obs_rate": useful_obs_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate intermediate Observation correctness (SOC@step, SOC-Avg, Useful-Obs Rate)."
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["hotpot", "2wiki"])
    parser.add_argument("--data_path", type=str, default="", help="Optional local data path (mainly for 2wiki).")
    parser.add_argument(
        "--result",
        type=str,
        action="append",
        required=True,
        help="Result spec in form label=path_to_result_json. Repeat for multiple methods.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Observation-vs-fact token-F1 threshold.")
    parser.add_argument("--output_json", type=str, default="", help="Optional output json path.")
    args = parser.parse_args()

    dataset_rows = _load_dataset_rows(args.dataset, args.data_path)
    compare: Dict[str, Any] = {}
    for spec in args.result:
        label, path = _parse_result_spec(spec)
        compare[label] = _compute_metrics(path, dataset_rows, args.dataset, float(args.threshold))

    print("\n=== Step Observation Correctness ===")
    print(f"dataset={args.dataset} threshold={float(args.threshold):.3f}")
    print("| method | SOC-Avg | Useful-Obs Rate | Obs Correct / Total |")
    print("|---|---:|---:|---:|")
    for label, m in compare.items():
        print(
            f"| {label} | {m['soc_avg']:.4f} | {m['useful_obs_rate']:.4f} | "
            f"{m['total_correct_observations']}/{m['total_observations']} |"
        )
    print("\nSOC@step details:")
    for label, m in compare.items():
        detail = ", ".join([f"step{s}:{v:.3f}" for s, v in m["soc_at_step"].items()]) or "N/A"
        print(f"- {label}: {detail}")

    if args.output_json:
        out = {
            "dataset": args.dataset,
            "threshold": float(args.threshold),
            "methods": compare,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nSaved: {args.output_json}")


if __name__ == "__main__":
    main()
