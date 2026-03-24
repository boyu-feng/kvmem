"""
ReAct Framework - HotpotQA Evaluation with Full Wikipedia Corpus

This script runs the ReAct agent on HotpotQA using a large pre-built Wikipedia
BM25 index (~5.2M articles from TIGER-Lab/LongRAG) as the external retrieval
corpus, following the approach used in SearchR1 and similar papers.

Unlike the distractor setting (which uses per-sample context paragraphs),
this open-domain setting requires the agent to retrieve relevant information
from the full Wikipedia corpus.

Prerequisites:
    1. Build the BM25 index first:
       python build_wiki_index.py
    2. Run this script:
       python react_wiki.py --num_samples 10 --output_file react_wiki_test_10.json

Usage:
    python react_wiki.py [--num_samples N] [--start_idx S] [--end_idx E]
                         [--output_file OUTPUT] [--index_dir INDEX_DIR]
"""

import json
import os
import re
import time
import random
import argparse

from models.QwenLLM import QwenLLM
from retrievers.WikiBM25Retriever import WikiBM25Retriever
from data.data_utils import load_hotpotqa_data
from eval.eval_utils import exact_match, f1_score


# ==================== Configuration ====================

MODEL_PATH = "/apdcephfs_szgm/share_303492287/AAA_models/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "/apdcephfs_szgm/share_303492287/ryanylsun/Projects/ReAct/results"
DEFAULT_INDEX_DIR = "/apdcephfs_szgm/share_303492287/ryanylsun/Projects/ReAct/data/wiki_index"

MAX_STEPS = 7
BM25_TOP_K = 3
CHECKPOINT_INTERVAL = 50


# ==================== ReAct Agent ====================
REACT_PROMPT_TEMPLATE = """You are a multi-hop question answering agent. You must answer the following question by reasoning step-by-step using the ReAct framework.

You have access to the following actions:
1. Search[query] - Search for relevant documents in Wikipedia using a query string. Returns top matching paragraphs.
2. Lookup[title] - Look up a specific Wikipedia article by its exact title. Returns the article content.
3. Finish[answer] - Return the final answer to the question.

Follow this format strictly:
Thought: <your reasoning about what to do next>
Action: <one of Search[query], Lookup[title], or Finish[answer]>

Important rules:
- You MUST output exactly one Thought and one Action per step.
- After receiving an Observation, continue with the next Thought and Action.
- Use Search to find information and Lookup to get details of a specific article.
- When you have enough information, use Finish[answer] where 'answer' is replaced by the actual short answer (e.g., Finish[yes], Finish[Paris], Finish[42]). Do NOT write Finish[answer] literally.
- Do NOT repeat the same action with the same argument.

Question: {question}
{trajectory}"""


def parse_action(text):
    """Parse the LLM output to extract Thought and Action."""
    thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|\Z)", text, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else text.strip()

    action_match = re.search(r"Action:\s*(Search|Lookup|Finish)\[(.*?)\]", text, re.DOTALL)
    if action_match:
        action_type = action_match.group(1)
        action_arg = action_match.group(2).strip()
        return thought, action_type, action_arg

    finish_match = re.search(r"Finish\[(.*?)\]", text, re.DOTALL)
    if finish_match:
        return thought, "Finish", finish_match.group(1).strip()

    return thought, None, None


def run_react_episode(question, llm, retriever, max_steps=MAX_STEPS, bm25_top_k=BM25_TOP_K):
    """Run one ReAct episode for a single question using the wiki corpus retriever."""
    trajectory = ""
    trajectory_log = []

    for step in range(1, max_steps + 1):
        prompt = REACT_PROMPT_TEMPLATE.format(
            question=question, trajectory=trajectory
        )
        response = llm.generate(prompt, max_new_tokens=256)
        thought, action_type, action_arg = parse_action(response)

        step_log = {
            "step": step,
            "thought": thought,
            "action_type": action_type,
            "action_arg": action_arg,
        }

        if action_type is None:
            trajectory += f"\nThought: {thought}\nAction: Finish[unknown]\n"
            step_log["observation"] = "Invalid action format. Forcing finish."
            trajectory_log.append(step_log)
            return "unknown", trajectory_log

        trajectory += f"\nThought: {thought}\nAction: {action_type}[{action_arg}]\n"

        if action_type == "Finish":
            step_log["observation"] = f"Final answer: {action_arg}"
            trajectory_log.append(step_log)
            return action_arg, trajectory_log

        if action_type == "Search":
            results = retriever.search(action_arg, top_k=bm25_top_k)
            if results:
                obs_parts = []
                for title, text, score in results:
                    truncated = text[:500] + "..." if len(text) > 500 else text
                    obs_parts.append(f"[{title}] (score={score:.2f}): {truncated}")
                observation = "\n".join(obs_parts)
            else:
                observation = "No results found."
        elif action_type == "Lookup":
            text = retriever.lookup(action_arg)
            if text:
                truncated = text[:800] + "..." if len(text) > 800 else text
                observation = f"[{action_arg}]: {truncated}"
            else:
                observation = f"No article found with title '{action_arg}'."

        step_log["observation"] = observation
        trajectory_log.append(step_log)
        trajectory += f"Observation: {observation}\n"

    return "unknown", trajectory_log


def select_samples(val_data, args):
    """
    Select samples based on args.
    Supports three modes:
      1. Random sampling: --num_samples N [--random_seed S]
      2. Range-based: --start_idx / --end_idx
      3. Full set: default (all samples)
    Returns a list of (original_index, sample) tuples.
    """
    total = len(val_data)

    if args.num_samples is not None and args.num_samples > 0:
        n = min(args.num_samples, total)
        rng = random.Random(args.random_seed)
        indices = sorted(rng.sample(range(total), n))
        print(f"[INFO] Random sampling {n} samples (seed={args.random_seed}) from {total} total.")
        return [(idx, val_data[idx]) for idx in indices]
    else:
        end_idx = total if args.end_idx == -1 else min(args.end_idx, total)
        start_idx = args.start_idx
        print(f"[INFO] Evaluating samples [{start_idx}, {end_idx}), total = {end_idx - start_idx}")
        return [(idx, val_data[idx]) for idx in range(start_idx, end_idx)]


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser(
        description="ReAct on HotpotQA with full Wikipedia corpus retrieval"
    )
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index in validation set (range mode)")
    parser.add_argument("--end_idx", type=int, default=-1,
                        help="End index, -1 for all (range mode)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to randomly select (random mode)")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--output_file", type=str, default="react_wiki_results.json",
                        help="Output filename")
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS,
                        help="Max ReAct steps per question")
    parser.add_argument("--bm25_top_k", type=int, default=BM25_TOP_K,
                        help="BM25 top-k retrieval results")
    parser.add_argument("--checkpoint_interval", type=int, default=CHECKPOINT_INTERVAL,
                        help="Save checkpoint every N samples")
    parser.add_argument("--index_dir", type=str, default=DEFAULT_INDEX_DIR,
                        help="Path to the pre-built BM25 wiki index directory")
    parser.add_argument("--no_load_corpus", action="store_true",
                        help="Don't load corpus texts (faster startup, but no text in results)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, args.output_file)
    checkpoint_path = os.path.join(
        OUTPUT_DIR, args.output_file.replace(".json", "_checkpoint.json")
    )

    # 1. Load evaluation data (HotpotQA distractor validation set)
    val_data = load_hotpotqa_data()

    # 2. Select samples
    selected_samples = select_samples(val_data, args)
    total_samples = len(selected_samples)

    # 3. Load the Wiki BM25 retriever (global corpus)
    load_corpus = not args.no_load_corpus
    retriever = WikiBM25Retriever(index_dir=args.index_dir, load_corpus=load_corpus)

    # 4. Load LLM
    llm = QwenLLM(MODEL_PATH)

    # 5. Load existing checkpoint if available
    results = []
    completed_ids = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            results = json.load(f)
        completed_ids = {r["id"] for r in results}
        print(f"[INFO] Resumed from checkpoint with {len(results)} completed samples.")

    # 6. Run ReAct on each sample
    em_scores = []
    f1_scores = []

    # Restore metrics from checkpoint
    for r in results:
        em_scores.append(r["em"])
        f1_scores.append(r["f1"])

    start_time = time.time()

    for orig_idx, sample in selected_samples:
        sample_id = sample["id"]

        if sample_id in completed_ids:
            continue

        question = sample["question"]
        gold_answer = sample["answer"]

        # Run ReAct with the global wiki corpus retriever
        pred_answer, trajectory_log = run_react_episode(
            question, llm, retriever,
            max_steps=args.max_steps,
            bm25_top_k=args.bm25_top_k,
        )

        # Evaluate
        em = exact_match(pred_answer, gold_answer)
        f1 = f1_score(pred_answer, gold_answer)
        em_scores.append(em)
        f1_scores.append(f1)

        result = {
            "id": sample_id,
            "index": orig_idx,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": pred_answer,
            "em": em,
            "f1": f1,
            "num_steps": len(trajectory_log),
            "trajectory": trajectory_log,
        }
        results.append(result)
        completed_ids.add(sample_id)

        # Progress logging
        done = len(em_scores)
        elapsed = time.time() - start_time
        avg_time = elapsed / done if done > 0 else 0
        eta = avg_time * (total_samples - done)
        running_em = sum(em_scores) / len(em_scores) * 100
        running_f1 = sum(f1_scores) / len(f1_scores) * 100

        print(
            f"[{done}/{total_samples}] idx={orig_idx} | "
            f"EM={em} F1={f1:.4f} | "
            f"Running EM={running_em:.1f}% F1={running_f1:.1f}% | "
            f"Elapsed={elapsed:.0f}s ETA={eta:.0f}s | "
            f"Pred='{pred_answer[:50]}' Gold='{gold_answer}'"
        )

        # Save checkpoint periodically
        if done % args.checkpoint_interval == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(results, f, ensure_ascii=False)
            print(f"[CHECKPOINT] Saved {done} results to {checkpoint_path}")

    # 7. Save final results
    final_em = sum(em_scores) / len(em_scores) * 100 if em_scores else 0
    final_f1 = sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0
    total_time = time.time() - start_time

    sampling_info = (
        f"random({args.num_samples}, seed={args.random_seed})"
        if args.num_samples else f"range([{args.start_idx}, {args.end_idx}))"
    )

    summary = {
        "method": "ReAct-Wiki",
        "model": MODEL_PATH,
        "retriever": "WikiBM25 (bm25s, TIGER-Lab/LongRAG hotpot_qa_wiki)",
        "index_dir": args.index_dir,
        "sampling": sampling_info,
        "total_samples": len(em_scores),
        "exact_match": final_em,
        "f1_score": final_f1,
        "total_time_seconds": total_time,
        "avg_time_per_sample": total_time / len(em_scores) if em_scores else 0,
    }

    output_data = {
        "summary": summary,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"ReAct-Wiki Evaluation Complete")
    print(f"{'='*60}")
    print(f"Retriever:     WikiBM25 (5.2M Wikipedia articles)")
    print(f"Sampling:      {sampling_info}")
    print(f"Total Samples: {len(em_scores)}")
    print(f"Exact Match:   {final_em:.2f}%")
    print(f"F1 Score:      {final_f1:.2f}%")
    print(f"Total Time:    {total_time:.1f}s ({total_time/3600:.1f}h)")
    if em_scores:
        print(f"Avg Time/Sample: {total_time/len(em_scores):.1f}s")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
