import os
import json
import time
import random
import gc
import torch
from typing import List, Tuple, Dict, Any, Optional
from datasets import load_dataset

# Configuration (kept as defaults; can be overridden by function args)
NUM_SAMPLES = 500
RANDOM_SEED = 42
MAX_STEPS = 10
BM25_TOP_K = 5
CHECKPOINT_INTERVAL = 50
MODEL_PATH = "Qwen2.5-7B-Instruct"
DATA_CACHE_DIR = os.path.expanduser("~/.cache/huggingface")

def load_hotpotqa_data(cache_dir: Optional[str] = None):
    """Load HotpotQA distractor validation set."""
    cache_dir = cache_dir or DATA_CACHE_DIR
    print("[INFO] Loading HotpotQA distractor validation set...")
    ds = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        cache_dir=cache_dir,
    )
    val_data = ds["validation"]
    print(f"[INFO] Loaded {len(val_data)} validation examples.")
    return val_data

def select_samples(val_data, num_samples: int = NUM_SAMPLES, seed: int = RANDOM_SEED):
    """
    Select samples matching original ReAct paper approach:
    Shuffle all indices with given seed, take first num_samples.
    """
    total = len(val_data)
    idxs = list(range(total))
    random.Random(seed).shuffle(idxs)
    selected_idxs = idxs[:num_samples]
    print(f"[INFO] Selected {len(selected_idxs)} samples (seed={seed}) from {total} total.")
    return [(idx, val_data[idx]) for idx in selected_idxs]

# --- Utility evaluation / parsing helpers ---
def extract_short_answer(raw_response: str) -> str:
    """Return a short predicted answer from model raw response (simple heuristic)."""
    if not raw_response:
        return ""
    # Take first non-empty line, strip trailing punctuation/quotes
    for line in raw_response.splitlines():
        s = line.strip()
        if s:
            # remove common prefixes like "Answer:" if present
            s = s.split("Answer:", 1)[-1].strip()
            return s.strip(' "\'`.')
    # fallback: whole response trimmed
    return raw_response.strip()

def _tokenize(s: str) -> List[str]:
    return s.lower().strip().split()

def exact_match(pred: str, gold: str) -> int:
    return 1 if pred.strip().lower() == gold.strip().lower() else 0

def f1_score(pred: str, gold: str) -> float:
    p_tokens = _tokenize(pred)
    g_tokens = _tokenize(gold)
    if not p_tokens or not g_tokens:
        return 0.0
    common = 0
    g_counts = {}
    for t in g_tokens:
        g_counts[t] = g_counts.get(t, 0) + 1
    for t in p_tokens:
        if g_counts.get(t, 0) > 0:
            common += 1
            g_counts[t] -= 1
    if common == 0:
        return 0.0
    precision = common / len(p_tokens)
    recall = common / len(g_tokens)
    return 2 * precision * recall / (precision + recall)

# --- Main runner (no kv compression) ---
def run_no_kv(selected_samples: List[Tuple[int, Dict[str, Any]]],
              output_path: str,
              checkpoint_path: str,
              model_path: str = MODEL_PATH,
              checkpoint_interval: int = CHECKPOINT_INTERVAL) -> Tuple[float, float, float]:
    """
    Single Model: direct QA without retrieval / without KV-compression.
    Returns (final_em, final_f1, total_time_seconds)
    """
    from models.QwenLLM import QwenLLM  # import here to avoid import-time dependency if unused

    PROMPT = """Answer the following question with a short and concise answer (just a few words or a short phrase). Do NOT explain your reasoning.

Question: {question}
Answer:"""

    llm = QwenLLM(model_path)

    results = []
    completed_ids = set()
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                results = json.load(f)
            completed_ids = {r["id"] for r in results}
            print(f"[INFO] Resumed from checkpoint with {len(results)} completed samples.")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint {checkpoint_path}: {e}")

    em_scores = [r.get("em", 0) for r in results]
    f1_scores = [r.get("f1", 0) for r in results]
    start_time = time.time()
    total_samples = len(selected_samples)

    for orig_idx, sample in selected_samples:
        sample_id = sample.get("id", f"idx_{orig_idx}")
        if sample_id in completed_ids:
            continue

        question = sample.get("question", "")
        gold_answer = sample.get("answer", "")

        prompt = PROMPT.format(question=question)
        raw_response = llm.generate(prompt, max_new_tokens=128)
        pred_answer = extract_short_answer(raw_response)

        em = exact_match(pred_answer, gold_answer)
        f1 = f1_score(pred_answer, gold_answer)
        em_scores.append(em)
        f1_scores.append(f1)

        result = {
            "id": sample_id, "index": orig_idx, "question": question,
            "gold_answer": gold_answer, "predicted_answer": pred_answer,
            "raw_response": raw_response, "em": em, "f1": f1,
        }
        results.append(result)
        completed_ids.add(sample_id)

        done = len(em_scores)
        elapsed = time.time() - start_time
        running_em = sum(em_scores) / len(em_scores) * 100
        running_f1 = sum(f1_scores) / len(f1_scores) * 100
        eta = (elapsed / done) * (total_samples - done) if done > 0 else 0
        print(f"[{done}/{total_samples}] EM={em} F1={f1:.4f} | Running EM={running_em:.1f}% F1={running_f1:.1f}% | ETA={eta:.0f}s | Pred='{pred_answer[:40]}' Gold='{gold_answer}'")

        if done % checkpoint_interval == 0:
            try:
                with open(checkpoint_path, "w") as f:
                    json.dump(results, f, ensure_ascii=False)
            except Exception as e:
                print(f"[WARN] Failed to write checkpoint: {e}")

    total_time = time.time() - start_time
    final_em = sum(em_scores) / len(em_scores) * 100 if em_scores else 0.0
    final_f1 = sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0.0

    output_data = {
        "summary": {
            "method": "Single Model (Direct QA - no kv compression)",
            "model": model_path,
            "total_samples": len(em_scores),
            "exact_match": final_em,
            "f1_score": final_f1,
            "total_time_seconds": total_time,
            "avg_time_per_sample": total_time / len(em_scores) if em_scores else None,
        },
        "results": results,
    }
    try:
        with open(output_path, "w") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to write output file {output_path}: {e}")

    print(f"\n{'='*60}\nSingle Model Complete: EM={final_em:.2f}%, F1={final_f1:.2f}%, Time={total_time:.1f}s\n{'='*60}")

    try:
        del llm
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

    return final_em, final_f1, total_time

def create_llm(model_path: str = MODEL_PATH):
	"""Create and return a QwenLLM instance (helper for other modules)."""
	from models.QwenLLM import QwenLLM
	return QwenLLM(model_path)