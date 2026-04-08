"""
Unified Experiment Runner: All 6 Methods on HotpotQA with Full Wikipedia Corpus

Methods:
1. Single Model (direct QA, no retrieval)
2. RAG (Retrieve-then-Read with Wikipedia BM25)
3. ReAct (multi-step reasoning with Wikipedia BM25)
4. ReAct-KV (none) - ReAct with KV cache reuse, no pruning
5. ReAct-KV (h2o) - ReAct with KV cache + H2O pruning
6. ReAct-KV (snapkv) - ReAct with KV cache + SnapKV pruning
7. Ours - ReAct with KV cache + our proposed pruning method

Uses:
- HotpotQA distractor validation set (7405 samples), 500 random samples (seed=233, matching original paper)
- Full Wikipedia corpus retrieval via BM25 (TIGER-Lab/LongRAG ~5.2M articles)
- Qwen2.5-7B-Instruct as the base LLM

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_all_wiki_experiments.py --experiment single
    CUDA_VISIBLE_DEVICES=0 python run_all_wiki_experiments.py --experiment rag
    CUDA_VISIBLE_DEVICES=0 python run_all_wiki_experiments.py --experiment react
    CUDA_VISIBLE_DEVICES=0 python run_all_wiki_experiments.py --experiment react_kv_none
    CUDA_VISIBLE_DEVICES=0 python run_all_wiki_experiments.py --experiment react_kv_h2o
    CUDA_VISIBLE_DEVICES=0 python run_all_wiki_experiments.py --experiment react_kv_snapkv
    CUDA_VISIBLE_DEVICES=0 python run_all_wiki_experiments.py --experiment collect  # generate summary
"""

import json
import os
import re
import time
import random
import argparse
import gc

# ==================== Configuration ====================
MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "results/wiki_0318_v3"
WIKI_INDEX_DIR = "data/wiki_index"
DATA_CACHE_DIR = "data/hotpotqa"
SUMMARY_PATH = "final_summary_0406_v3.md"

# Match original ReAct paper: seed=233, first 500 from shuffled 7405 dev samples
NUM_SAMPLES = 500
RANDOM_SEED = 233

MAX_STEPS = 7
BM25_TOP_K = 5
CHECKPOINT_INTERVAL = 50


# ==================== Evaluation Metrics ====================
def normalize_answer(s):
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    from collections import Counter
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ==================== Data Loading ====================
def load_hotpotqa_data():
    """Load HotpotQA distractor validation set."""
    from datasets import load_dataset
    print("[INFO] Loading HotpotQA distractor validation set...")
    ds = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        cache_dir=DATA_CACHE_DIR,
    )
    val_data = ds["validation"]
    print(f"[INFO] Loaded {len(val_data)} validation examples.")
    return val_data


def select_samples(val_data):
    """
    Select 500 samples matching original ReAct paper approach:
    Shuffle all indices with seed 233, take first 500.
    """
    total = len(val_data)
    idxs = list(range(total))
    random.Random(RANDOM_SEED).shuffle(idxs)
    selected_idxs = idxs[:NUM_SAMPLES]
    print(f"[INFO] Selected {NUM_SAMPLES} samples (seed={RANDOM_SEED}) from {total} total.")
    return [(idx, val_data[idx]) for idx in selected_idxs]


# ==================== ReAct Prompt ====================
REACT_PROMPT_TEMPLATE = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
{examples}
Question: {question}
{trajectory}"""

# Few-shot examples from original ReAct paper (webthink_simple6)
REACT_EXAMPLES = """
Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
Thought 2: It does not mention the eastern sector. So I need to look up eastern sector.
Action 2: Lookup[eastern sector]
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
Thought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.
Action 3: Search[High Plains]
Observation 3: High Plains refers to one of two distinct land regions:
Thought 4: I need to instead search High Plains (United States).
Action 4: Search[High Plains (United States)]
Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]
Thought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action 5: Finish[1,800 to 7,000 ft]
Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 
Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action 3: Finish[Richard Nixon]
Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Thought 1: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.
Action 1: Search[Adam Clayton Powell]
Observation 1: Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].
Thought 2: To find the documentary, I can search Adam Clayton Powell (film).
Action 2: Search[Adam Clayton Powell (film)]
Observation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.\nThe film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.
Thought 3: Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.
Action 3: Finish[The Saimaa Gesture]
Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought 1: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.
Action 1: Search[Nicholas Ray]
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Thought 2: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.
Action 2: Search[Elia Kazan]
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Thought 3: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Action 3: Finish[director, screenwriter, actor]
Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century. 
Thought 2: Arthur's Magazine was started in 1844. I need to search First for Women next.
Action 2: Search[First for Women]
Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989. 
Thought 3: First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.
Action 3: Finish[Arthur's Magazine]
Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.
Action 1: Search[Pavel Urysohn]
Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 – August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.
Thought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.
Action 2: Search[Leonid Levin]
Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist. 
Thought 3: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work. 
Action 3: Finish[yes]
"""


# ==================== ReAct-KV Prompt (for incremental generation) ====================
REACT_KV_INITIAL_PROMPT = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
{examples}
Question: {question}
"""


def parse_action_original(text, step_num):
    """
    Parse action in original ReAct format with step numbers.
    Handles: Thought N: ... Action N: ...
    """
    # Try to parse with step numbers
    thought_match = re.search(
        rf"Thought\s*{step_num}:\s*(.*?)(?=\nAction\s*{step_num}:|\Z)",
        text, re.DOTALL
    )
    if thought_match:
        thought = thought_match.group(1).strip()
    else:
        # Fallback: try without step number
        thought_match = re.search(r"Thought[^:]*:\s*(.*?)(?=\nAction|\Z)", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else text.strip()

    # Parse action
    action_match = re.search(
        rf"Action\s*{step_num}:\s*(search|lookup|finish)\[(.*?)\]",
        text, re.DOTALL | re.IGNORECASE
    )
    if not action_match:
        action_match = re.search(
            r"Action[^:]*:\s*(search|lookup|finish)\[(.*?)\]",
            text, re.DOTALL | re.IGNORECASE
        )
    if action_match:
        action_type = action_match.group(1).lower()
        action_arg = action_match.group(2).strip()
        return thought, action_type, action_arg

    # Fallback: try to find finish anywhere
    finish_match = re.search(r"[Ff]inish\[(.*?)\]", text)
    if finish_match:
        return thought, "finish", finish_match.group(1).strip()

    return thought, None, None


def _get_page_obs(text, max_sentences=5):
    """Extract first N sentences from article text, mimicking original WikiEnv get_page_obs."""
    sentences = []
    paragraphs = text.split("\n")
    for p in paragraphs:
        p = p.strip()
        if p:
            sentences.extend(s.strip() + '.' for s in p.split('. ') if s.strip())
    obs = ' '.join(sentences[:max_sentences])
    return obs if obs else text[:1000]


def execute_action(action_type, action_arg, retriever, lookup_state):
    """
    Execute an action using the WikiBM25Retriever, mimicking original WikiEnv behavior.
    
    Key difference from v1: Search first tries exact title match (like original Wikipedia API),
    then falls back to BM25. When BM25 doesn't find the exact entity, returns "similar entities"
    list so the agent can refine its search (matching original ReAct paper behavior).
    
    Returns (observation_str, updated_lookup_state)
    """
    if action_type == "search":
        # Step 1: Try exact title match first (this is what the original Wikipedia API does)
        exact_text = retriever.lookup(action_arg)
        if exact_text:
            obs = _get_page_obs(exact_text)
            # Store full page text for Lookup action
            lookup_state["page"] = exact_text
            lookup_state["lookup_keyword"] = None
            lookup_state["lookup_list"] = None
            lookup_state["lookup_cnt"] = 0
            return obs, lookup_state

        # Step 2: Exact match failed, try BM25 search
        results = retriever.search(action_arg, top_k=BM25_TOP_K)
        if results:
            # Check if the top BM25 result's title closely matches the search query
            top_title, top_text, top_score = results[0]
            if top_text and top_title.lower().strip() == action_arg.lower().strip():
                # BM25 found an exact title match
                obs = _get_page_obs(top_text)
                lookup_state["page"] = top_text
                lookup_state["lookup_keyword"] = None
                lookup_state["lookup_list"] = None
                lookup_state["lookup_cnt"] = 0
                return obs, lookup_state

            # Check if the query is a substring of top title or vice versa
            if top_text and (
                action_arg.lower() in top_title.lower() or
                top_title.lower() in action_arg.lower()
            ):
                obs = _get_page_obs(top_text)
                lookup_state["page"] = top_text
                lookup_state["lookup_keyword"] = None
                lookup_state["lookup_list"] = None
                lookup_state["lookup_cnt"] = 0
                return obs, lookup_state

            # BM25 returned results but no good title match → return similar entities
            # This matches the original ReAct paper's "Could not find X. Similar: [...]" pattern
            similar_titles = [r[0] for r in results[:5]]
            obs = f"Could not find [{action_arg}]. Similar: {similar_titles}."
            lookup_state["page"] = None
            return obs, lookup_state
        else:
            obs = f"Could not find [{action_arg}]."
            lookup_state["page"] = None
        return obs, lookup_state

    elif action_type == "lookup":
        if lookup_state.get("page") is None:
            return "No page loaded. Use Search first.", lookup_state
        
        keyword = action_arg
        page = lookup_state["page"]
        
        if lookup_state.get("lookup_keyword") != keyword:
            # Reset lookup for new keyword
            lookup_state["lookup_keyword"] = keyword
            # Find all sentences containing the keyword in the FULL page text
            paragraphs = page.split("\n")
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            sentences = []
            for p in paragraphs:
                sentences.extend(s.strip() + '.' for s in p.split('. ') if s.strip())
            lookup_state["lookup_list"] = [
                s for s in sentences if keyword.lower() in s.lower()
            ]
            lookup_state["lookup_cnt"] = 0
        
        if lookup_state["lookup_cnt"] >= len(lookup_state.get("lookup_list", [])):
            obs = "No more results."
        else:
            cnt = lookup_state["lookup_cnt"]
            total = len(lookup_state["lookup_list"])
            obs = f"(Result {cnt + 1} / {total}) {lookup_state['lookup_list'][cnt]}"
            lookup_state["lookup_cnt"] = cnt + 1
        
        return obs, lookup_state

    elif action_type == "finish":
        return f"Episode finished.", lookup_state
    
    return "Invalid action.", lookup_state


def extract_short_answer(response):
    """Extract short answer from model response."""
    answer = response.split("\n")[0].strip()
    for prefix in ["The answer is ", "Answer: ", "A: "]:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
    answer = answer.rstrip(".")
    return answer


# ==================== Experiment: Single Model ====================
def run_single_experiment(val_data, selected_samples, output_path, checkpoint_path):
    """Single Model: direct QA without retrieval."""
    import torch
    from models.QwenLLM import QwenLLM

    PROMPT = """Answer the following question with a short and concise answer (just a few words or a short phrase). Do NOT explain your reasoning.

Question: {question}
Answer:"""

    llm = QwenLLM(MODEL_PATH)

    results = []
    completed_ids = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            results = json.load(f)
        completed_ids = {r["id"] for r in results}
        print(f"[INFO] Resumed from checkpoint with {len(results)} completed samples.")

    em_scores = [r["em"] for r in results]
    f1_scores = [r["f1"] for r in results]
    start_time = time.time()
    total_samples = len(selected_samples)

    for orig_idx, sample in selected_samples:
        sample_id = sample["id"]
        if sample_id in completed_ids:
            continue

        question = sample["question"]
        gold_answer = sample["answer"]

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

        if done % CHECKPOINT_INTERVAL == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(results, f, ensure_ascii=False)

    total_time = time.time() - start_time
    final_em = sum(em_scores) / len(em_scores) * 100
    final_f1 = sum(f1_scores) / len(f1_scores) * 100

    output_data = {
        "summary": {
            "method": "Single Model (Direct QA)",
            "model": MODEL_PATH,
            "total_samples": len(em_scores),
            "exact_match": final_em,
            "f1_score": final_f1,
            "total_time_seconds": total_time,
            "avg_time_per_sample": total_time / len(em_scores),
        },
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}\nSingle Model Complete: EM={final_em:.2f}%, F1={final_f1:.2f}%, Time={total_time:.1f}s\n{'='*60}")

    del llm
    torch.cuda.empty_cache()
    gc.collect()

    return final_em, final_f1, total_time


# ==================== Experiment: RAG ====================
def run_rag_experiment(val_data, selected_samples, retriever, output_path, checkpoint_path):
    """RAG: Retrieve-then-Read with Wikipedia BM25."""
    import torch
    from models.QwenLLM import QwenLLM

    RAG_PROMPT = """Read the following retrieved passages and answer the question with a short and concise answer (just a few words or a short phrase). Do NOT explain your reasoning.

Retrieved Passages:
{context}

Question: {question}
Answer:"""

    llm = QwenLLM(MODEL_PATH)

    results = []
    completed_ids = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            results = json.load(f)
        completed_ids = {r["id"] for r in results}
        print(f"[INFO] Resumed from checkpoint with {len(results)} completed samples.")

    em_scores = [r["em"] for r in results]
    f1_scores = [r["f1"] for r in results]
    start_time = time.time()
    total_samples = len(selected_samples)

    for orig_idx, sample in selected_samples:
        sample_id = sample["id"]
        if sample_id in completed_ids:
            continue

        question = sample["question"]
        gold_answer = sample["answer"]

        # Retrieve from full Wikipedia
        search_results = retriever.search(question, top_k=BM25_TOP_K)
        context_parts = []
        for title, text, score in search_results:
            truncated = text[:1500] + "..." if len(text) > 1500 else text
            context_parts.append(f"[{title}] (score={score:.2f}): {truncated}")
        context_str = "\n\n".join(context_parts) if context_parts else "No relevant passages found."

        prompt = RAG_PROMPT.format(context=context_str, question=question)
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

        if done % CHECKPOINT_INTERVAL == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(results, f, ensure_ascii=False)

    total_time = time.time() - start_time
    final_em = sum(em_scores) / len(em_scores) * 100
    final_f1 = sum(f1_scores) / len(f1_scores) * 100

    output_data = {
        "summary": {
            "method": "RAG (Retrieve-then-Read, Full Wiki)",
            "model": MODEL_PATH, "bm25_top_k": BM25_TOP_K,
            "total_samples": len(em_scores),
            "exact_match": final_em, "f1_score": final_f1,
            "total_time_seconds": total_time,
            "avg_time_per_sample": total_time / len(em_scores),
        },
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}\nRAG Complete: EM={final_em:.2f}%, F1={final_f1:.2f}%, Time={total_time:.1f}s\n{'='*60}")

    del llm
    torch.cuda.empty_cache()
    gc.collect()

    return final_em, final_f1, total_time


# ==================== Experiment: ReAct ====================
def run_react_experiment(val_data, selected_samples, retriever, output_path, checkpoint_path):
    """ReAct: multi-step reasoning with Wikipedia BM25 retrieval."""
    import torch
    from models.QwenLLM import QwenLLM

    llm = QwenLLM(MODEL_PATH)

    results = []
    completed_ids = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            results = json.load(f)
        completed_ids = {r["id"] for r in results}
        print(f"[INFO] Resumed from checkpoint with {len(results)} completed samples.")

    em_scores = [r["em"] for r in results]
    f1_scores = [r["f1"] for r in results]
    start_time = time.time()
    total_samples = len(selected_samples)

    for orig_idx, sample in selected_samples:
        sample_id = sample["id"]
        if sample_id in completed_ids:
            continue

        question = sample["question"]
        gold_answer = sample["answer"]

        # Run ReAct episode
        pred_answer, trajectory_log, n_steps = _run_react_episode(
            question, llm, retriever
        )

        em = exact_match(pred_answer, gold_answer)
        f1 = f1_score(pred_answer, gold_answer)
        em_scores.append(em)
        f1_scores.append(f1)

        result = {
            "id": sample_id, "index": orig_idx, "question": question,
            "gold_answer": gold_answer, "predicted_answer": pred_answer,
            "em": em, "f1": f1, "num_steps": n_steps,
            "trajectory": trajectory_log,
        }
        results.append(result)
        completed_ids.add(sample_id)

        done = len(em_scores)
        elapsed = time.time() - start_time
        running_em = sum(em_scores) / len(em_scores) * 100
        running_f1 = sum(f1_scores) / len(f1_scores) * 100
        eta = (elapsed / done) * (total_samples - done) if done > 0 else 0
        print(f"[{done}/{total_samples}] EM={em} F1={f1:.4f} | Running EM={running_em:.1f}% F1={running_f1:.1f}% | ETA={eta:.0f}s | Pred='{pred_answer[:40]}' Gold='{gold_answer}'")

        if done % CHECKPOINT_INTERVAL == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(results, f, ensure_ascii=False)

    total_time = time.time() - start_time
    final_em = sum(em_scores) / len(em_scores) * 100
    final_f1 = sum(f1_scores) / len(f1_scores) * 100

    output_data = {
        "summary": {
            "method": "ReAct (Full Wiki)",
            "model": MODEL_PATH, "max_steps": MAX_STEPS,
            "total_samples": len(em_scores),
            "exact_match": final_em, "f1_score": final_f1,
            "total_time_seconds": total_time,
            "avg_time_per_sample": total_time / len(em_scores),
        },
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}\nReAct Complete: EM={final_em:.2f}%, F1={final_f1:.2f}%, Time={total_time:.1f}s\n{'='*60}")

    del llm
    torch.cuda.empty_cache()
    gc.collect()

    return final_em, final_f1, total_time


def _run_react_episode(question, llm, retriever, max_steps=MAX_STEPS):
    """
    Run one ReAct episode following original paper format:
    Thought N: ... / Action N: ... / Observation N: ...
    """
    prompt = REACT_PROMPT_TEMPLATE.format(
        examples=REACT_EXAMPLES, question=question, trajectory=""
    )
    trajectory = ""
    trajectory_log = []
    lookup_state = {"page": None, "lookup_keyword": None, "lookup_list": None, "lookup_cnt": 0}

    for i in range(1, max_steps + 1):
        # Generate thought + action
        response = llm.generate(
            prompt + trajectory + f"Thought {i}:",
            max_new_tokens=256,
        )

        thought, action_type, action_arg = parse_action_original(
            f"Thought {i}:" + response, i
        )

        if action_type is None:
            # Bad parse — try to recover
            # Try a second call for just the action
            action_response = llm.generate(
                prompt + trajectory + f"Thought {i}: {thought}\nAction {i}:",
                max_new_tokens=64,
            )
            _, action_type, action_arg = parse_action_original(
                f"Action {i}:" + action_response, i
            )
            if action_type is None:
                trajectory_log.append({"step": i, "thought": thought, "action": "finish[]"})
                return "", trajectory_log, i

        step_log = {
            "step": i,
            "thought": thought,
            "action_type": action_type,
            "action_arg": action_arg,
        }

        if action_type == "finish":
            trajectory_log.append(step_log)
            return action_arg if action_arg else "", trajectory_log, i

        # Execute action
        obs, lookup_state = execute_action(action_type, action_arg, retriever, lookup_state)
        obs = obs.replace('\\n', '')
        step_log["observation"] = obs[:1000]
        trajectory_log.append(step_log)

        # Build trajectory string matching original format
        trajectory += f"Thought {i}: {thought}\nAction {i}: {action_type}[{action_arg}]\nObservation {i}: {obs}\n"

    # Exceeded max steps
    return "", trajectory_log, max_steps


# ==================== Experiment: ReAct-KV ====================
def run_react_kv_experiment(val_data, selected_samples, retriever, pruning_mode,
                            output_path, checkpoint_path):
    """ReAct with KV Cache: supports none/h2o/snapkv/ours pruning modes."""
    import torch
    from models.QwenLLMWithKVCache import QwenLLMWithKVCache

    kv_config = {
        "pruning_mode": pruning_mode,
        "prune_every_n": 2,
        "keep_ratio": 0.5,
        "pool_window": 4,
        "max_trajectory_tokens": 1024,
        "sink_size": 4,
        "observation_window": 128,
        "num_score_layers": 3,
        "attn_mode": "scoring_forward",
    }

    llm = QwenLLMWithKVCache(MODEL_PATH, kv_config)

    results = []
    completed_ids = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            results = json.load(f)
        completed_ids = {r["id"] for r in results}
        print(f"[INFO] Resumed from checkpoint with {len(results)} completed samples.")

    em_scores = [r["em"] for r in results]
    f1_scores = [r["f1"] for r in results]
    start_time = time.time()
    total_samples = len(selected_samples)

    for orig_idx, sample in selected_samples:
        sample_id = sample["id"]
        if sample_id in completed_ids:
            continue

        question = sample["question"]
        gold_answer = sample["answer"]

        # Run ReAct-KV episode
        sample_start = time.time()
        pred_answer, trajectory_log, step_timings = _run_react_kv_episode(
            question, llm, retriever
        )
        sample_time = time.time() - sample_start
        llm_stats = llm.get_stats()

        em = exact_match(pred_answer, gold_answer)
        f1 = f1_score(pred_answer, gold_answer)
        em_scores.append(em)
        f1_scores.append(f1)

        result = {
            "id": sample_id, "index": orig_idx, "question": question,
            "gold_answer": gold_answer, "predicted_answer": pred_answer,
            "em": em, "f1": f1, "num_steps": len(trajectory_log),
            "sample_time": sample_time,
            "step_timings": step_timings,
            "llm_stats": {
                "prefill_time": llm_stats.get("prefill_time", 0),
                "decode_time": llm_stats.get("decode_time", 0),
                "scoring_time": llm_stats.get("scoring_time", 0),
                "pruning_time": llm_stats.get("pruning_time", 0),
                "total_prune_count": llm_stats.get("total_prune_count", 0),
                "final_cache_len": llm_stats.get("current_cache_len", 0),
            },
            "trajectory": trajectory_log,
        }
        results.append(result)
        completed_ids.add(sample_id)

        done = len(em_scores)
        elapsed = time.time() - start_time
        running_em = sum(em_scores) / len(em_scores) * 100
        running_f1 = sum(f1_scores) / len(f1_scores) * 100
        eta = (elapsed / done) * (total_samples - done) if done > 0 else 0
        prune_info = f" Prunes={llm_stats.get('total_prune_count', 0)}" if pruning_mode != "none" else ""
        print(f"[{done}/{total_samples}] EM={em} F1={f1:.4f} | Running EM={running_em:.1f}% F1={running_f1:.1f}% | Time={sample_time:.1f}s{prune_info} | ETA={eta:.0f}s | Pred='{pred_answer[:40]}' Gold='{gold_answer}'")

        if done % CHECKPOINT_INTERVAL == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(results, f, ensure_ascii=False)

    total_time = time.time() - start_time
    final_em = sum(em_scores) / len(em_scores) * 100
    final_f1 = sum(f1_scores) / len(f1_scores) * 100

    # Aggregate timing stats
    all_step_timings = []
    for r in results:
        all_step_timings.extend(r.get("step_timings", []))
    timing_stats = {}
    if all_step_timings:
        total_gen_time = sum(t.get("generation_time", 0) for t in all_step_timings)
        cache_lens = [t.get("kv_cache_length", 0) for t in all_step_timings]
        timing_stats = {
            "avg_step_time": total_gen_time / len(all_step_timings),
            "total_generation_time": total_gen_time,
            "total_steps": len(all_step_timings),
            "avg_kv_cache_length": sum(cache_lens) / len(cache_lens) if cache_lens else 0,
            "max_kv_cache_length": max(cache_lens) if cache_lens else 0,
        }

    total_prune_count = sum(
        r.get("llm_stats", {}).get("total_prune_count", 0) for r in results
    )

    output_data = {
        "summary": {
            "method": f"ReAct-KV ({pruning_mode}, Full Wiki)",
            "model": MODEL_PATH, "max_steps": MAX_STEPS,
            "pruning_mode": pruning_mode, "kv_config": kv_config,
            "total_samples": len(em_scores),
            "exact_match": final_em, "f1_score": final_f1,
            "total_time_seconds": total_time,
            "avg_time_per_sample": total_time / len(em_scores),
            "timing_stats": timing_stats,
            "total_prune_count": total_prune_count,
        },
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}\nReAct-KV ({pruning_mode}) Complete: EM={final_em:.2f}%, F1={final_f1:.2f}%, Time={total_time:.1f}s\n{'='*60}")

    del llm
    torch.cuda.empty_cache()
    gc.collect()

    return final_em, final_f1, total_time


def _run_react_kv_episode(question, llm, retriever, max_steps=MAX_STEPS, window_size=128):
    trajectory_log = []
    step_timings = []
    lookup_state = {"page": None, "lookup_keyword": None, "lookup_list": None, "lookup_cnt": 0}
    kv_stop_strings = ["\nObservation", "\nQuestion:"]
    import torch

    def _normalize_kv(kv):
        """
        Normalize KV to:
        ((k,v),(k,v),...)
        """
        if kv is None:
            return None

        normalized = []

        for layer_kv in kv:
            # case 1: already (k,v)
            if isinstance(layer_kv, (tuple, list)) and len(layer_kv) == 2:
                normalized.append((layer_kv[0], layer_kv[1]))

            # case 2: ((k,v),)
            elif isinstance(layer_kv, (tuple, list)) and len(layer_kv) == 1:
                inner = layer_kv[0]
                normalized.append((inner[0], inner[1]))

            else:
                raise ValueError(f"Unexpected KV format: {type(layer_kv)} | {layer_kv}")

        return tuple(normalized)
    
    def _process_kv_flow(prompt_kv, memory_block, new_kv, window_size):
        """
        new_kv: tuple of layer KV
            ((k,v),(k,v),...)
        """
        if new_kv is None:
            return None, memory_block
    
        # recent window
        recent_kv = []
        for n_k, n_v in new_kv:
            r_k = n_k[:, :, -window_size:, :]
            r_v = n_v[:, :, -window_size:, :]
            recent_kv.append((r_k, r_v))
        recent_kv = tuple(recent_kv)
    
        # update memory
        if memory_block is None:
            memory_block = recent_kv
        else:
            fused_memory = []
            for (m_k, m_v), (r_k, r_v) in zip(memory_block, recent_kv):
                f_k = torch.cat([m_k, r_k], dim=2)
                f_v = torch.cat([m_v, r_v], dim=2)
                fused_memory.append((f_k, f_v))
            memory_block = tuple(fused_memory)
    
        return recent_kv, memory_block

    # Step 1: 初始生成
    initial_prompt = REACT_KV_INITIAL_PROMPT.format(
        examples=REACT_EXAMPLES, question=question
    ) + "Thought 1:"
    
    t_step_start = time.time()
    print(f"Generating initial thought and action...time={time.strftime('%H:%M:%S')}")
    response, prompt_kv, generated_kv = llm.generate_first(
        initial_prompt, max_new_tokens=256, stop_strings=kv_stop_strings
    )
    step_time = time.time() - t_step_start
    print(f"Initial generation complete. Time taken: {step_time:.2f}s")
    print(f"Response:{len(response)}")
    print(response)
    print(f"[KV DEBUG] prompt_kv: type={type(prompt_kv)}, len={len(prompt_kv) if prompt_kv is not None else 'None'}")
    if prompt_kv and len(prompt_kv) > 0:
        print(f"[KV DEBUG] prompt_kv[0] shape: K={prompt_kv[0][0].shape}, V={prompt_kv[0][1].shape}")
    print(f"[KV DEBUG] generated_kv: type={type(generated_kv)}, len={len(generated_kv) if generated_kv is not None else 'None'}")
    if generated_kv and len(generated_kv) > 0:
        print(f"[KV DEBUG] generated_kv[0] shape: K={generated_kv[0][0].shape}, V={generated_kv[0][1].shape}")
    print(f"[KV DEBUG] LLM past_key_values type: {type(llm.past_key_values)}")

    thought, action_type, action_arg = parse_action_original("Thought 1:" + response, 1)

    # --- 【初始化】 ---
    memory_block = None
    recent_kv = None  # 初始化 recent_kv

    # 检查生成的 KV 是否有效
    if generated_kv is None or len(generated_kv) == 0:
        print(f"[ERROR] generated_kv is invalid: {generated_kv}")
        return "", trajectory_log, step_timings
    
    try:
        recent_kv, memory_block = _process_kv_flow(None, memory_block, generated_kv, window_size)
        print(f"[DEBUG] After processing: recent_kv type={type(recent_kv)}, memory_block type={type(memory_block)}")
        if recent_kv is not None:
            print(f"[DEBUG] {recent_kv[0]}")
    except Exception as e:
        print(f"Error occurred while processing kv flow: {e}")
        import traceback
        traceback.print_exc()
        # 如果处理失败，创建空的 recent_kv
        if recent_kv is None:
            # 创建一个空的 KV 作为 fallback
            num_layers = llm.model.config.num_hidden_layers
            empty_k = torch.zeros(1, llm.model.config.num_attention_heads, 0, llm.model.config.head_dim).to(llm.device)
            empty_v = torch.zeros(1, llm.model.config.num_attention_heads, 0, llm.model.config.head_dim).to(llm.device)
            recent_kv = (empty_k, empty_v)

    # 添加检查再打印
    if recent_kv is not None:
        print(
        f"Initial KV processed. "
        f"Recent KV length: {recent_kv[0][0].size(2) if recent_kv and recent_kv[0][0] is not None else 0}, "
        f"Memory block updated: {memory_block is not None}"
        )
    else:
        print(f"Initial KV processed. Recent KV is None, Memory block updated: {memory_block is not None}")
        # 如果 recent_kv 仍然是 None，创建一个默认值
        num_layers = llm.model.config.num_hidden_layers
        empty_k = torch.zeros(1, llm.model.config.num_attention_heads, 0, llm.model.config.head_dim).to(llm.device)
        empty_v = torch.zeros(1, llm.model.config.num_attention_heads, 0, llm.model.config.head_dim).to(llm.device)
        recent_kv = (empty_k, empty_v)
    
    step_log = {"step": 1, "thought": thought, "action_type": action_type, "action_arg": action_arg}
    step_timings.append({"step": 1, "generation_time": step_time, "kv_cache_length": llm.get_cache_len()})

    if action_type is None or action_type == "finish":
        return action_arg if action_type == "finish" else "", trajectory_log, step_timings

    obs, lookup_state = execute_action(action_type, action_arg, retriever, lookup_state)
    obs = obs.replace('\\n', '')
    step_log["observation"] = obs[:1000]
    trajectory_log.append(step_log)

    # 后续步骤：增量生成
    for step in range(2, max_steps + 1):
        new_text = f"\nObservation {step - 1}: {obs}\nThought {step}:"
        print("new_text:", new_text[:50])
        t_step_start = time.time()
        
        # 确保 prompt_kv 有效
        if prompt_kv is None or len(prompt_kv) == 0:
            print(f"[ERROR] prompt_kv is invalid at step {step}")
            break
            
        # 推理时使用当前的 memory_block 和精确的 recent_kv
        try:
            response, obs_kv, gen_kv = llm.generate_incremental_with_memory(
                new_text,
                prompt_kv=prompt_kv,
                memory_block=memory_block,
                recent_kv=recent_kv,
                max_new_tokens=256,
                stop_strings=kv_stop_strings
            )
        except Exception as e:
            print(f"Error in generate_incremental_with_memory at step {step}: {e}")
            import traceback
            traceback.print_exc()
            break
            
        step_time = time.time() - t_step_start
        print(f"Step {step} generation complete. Time taken: {step_time:.2f}s | Response length: {len(response)}")
        
        # 检查返回的 KV 是否有效
        if obs_kv is not None and len(obs_kv) > 0:
            print(f"obs_kv has done")
        if gen_kv is not None and len(gen_kv) > 0:
            print(f"gen_kv has done")

        print("type(obs_kv):", type(obs_kv))
        print("len(obs_kv):", len(obs_kv))

        print("type(gen_kv):", type(gen_kv))
        print("len(gen_kv):", len(gen_kv))

        print("obs_kv[0] type:", type(obs_kv[0]))
        print("gen_kv[0] type:", type(gen_kv[0]))
        print("gen_kv[0]:", gen_kv[0])
        obs_kv = _normalize_kv(obs_kv)
        gen_kv = _normalize_kv(gen_kv)

        step_kv = []
        for (o_k, o_v), (g_k, g_v) in zip(obs_kv, gen_kv):
            k = torch.cat([o_k, g_k], dim=2)
            v = torch.cat([o_v, g_v], dim=2)
            step_kv.append((k, v))

        step_kv = tuple(step_kv)

        step_kv = tuple(step_kv)
        # --- 【核心逻辑：先更新 Recent，溢出部分进 Memory】 ---
        try:
            recent_kv, memory_block = _process_kv_flow(
                recent_kv, 
                memory_block, 
                step_kv, 
                window_size
            )
        except Exception as e:
            print(f"Error in _process_kv_flow at step {step}: {e}")
            import traceback
            traceback.print_exc()
            # 继续执行，但可能后续会有问题

        # --- 后续解析与执行 ---
        thought, action_type, action_arg = parse_action_original("Thought " + str(step) + ":" + response, step)
        step_log = {"step": step, "thought": thought, "action_type": action_type, "action_arg": action_arg}
        step_timings.append({
            "step": step, 
            "generation_time": step_time, 
            "kv_cache_length": llm.get_cache_len(),
            "pruned_this_step": llm.kv_manager.last_pruned if llm.kv_manager else False
        })
        print(f"Step {step} thought: {thought[:100]}")
        print(f"Step {step} parsed action: {action_type}[{action_arg}]")

        if action_type == "finish":
            trajectory_log.append(step_log)
            return action_arg if action_arg else "", trajectory_log, step_timings

        obs, lookup_state = execute_action(action_type, action_arg, retriever, lookup_state)
        obs = obs.replace('\\n', '')
        print(f"Step {step} observation: {obs[:200]}...")
        step_log["observation"] = obs[:1000]
        trajectory_log.append(step_log)

    return "", trajectory_log, step_timings


# ==================== Collect Results ====================
def collect_results():
    """Collect all experiment results and generate the final summary markdown."""
    experiments = {
        "single": ("Single Model (Direct QA)", "single_wiki_500_0318.json"),
        "rag": ("RAG (Full Wiki)", "rag_wiki_500_0318.json"),
        "react": ("ReAct (Full Wiki)", "react_wiki_500_0318.json"),
        "react_kv_none": ("ReAct-KV (none)", "react_kv_none_wiki_500_0318.json"),
        "react_kv_h2o": ("ReAct-KV (H2O)", "react_kv_h2o_wiki_500_0318.json"),
        "react_kv_snapkv": ("ReAct-KV (SnapKV)", "react_kv_snapkv_wiki_500_0318.json"),
        "react_kv_h2o_prune": ("ReAct-KV (H2O, Aggressive Prune)", "react_kv_h2o_prune_wiki_500_0318.json"),
        "react_kv_ours": ("ReAct-KV (Ours)", "react_kv_ours_wiki_500_0331.json"),
    }

    results_table = []
    for exp_key, (name, filename) in experiments.items():
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
            summary = data.get("summary", {})
            results_table.append({
                "method": name,
                "em": summary.get("exact_match", 0),
                "f1": summary.get("f1_score", 0),
                "total_time": summary.get("total_time_seconds", 0),
                "avg_time": summary.get("avg_time_per_sample", 0),
                "n_samples": summary.get("total_samples", 0),
                "extra": summary.get("timing_stats", {}),
                "prune_count": summary.get("total_prune_count", 0),
            })
        else:
            print(f"[WARN] Missing result file: {filepath}")
            results_table.append({
                "method": name, "em": -1, "f1": -1,
                "total_time": 0, "avg_time": 0, "n_samples": 0,
                "extra": {}, "prune_count": 0,
            })

    # Generate markdown report
    md = []
    md.append("# HotpotQA Experiment Results — Full Wikipedia Corpus")
    md.append("")
    md.append(f"**Date**: 2026-03-18")
    md.append(f"**Model**: Qwen2.5-7B-Instruct")
    md.append(f"**Dataset**: HotpotQA Dev Validation (500 samples, seed={RANDOM_SEED})")
    md.append(f"**Retrieval Corpus**: TIGER-Lab/LongRAG Wikipedia (~5.2M articles, BM25)")
    md.append(f"**Max ReAct Steps**: {MAX_STEPS}")
    md.append(f"**BM25 Top-K**: {BM25_TOP_K}")
    md.append("")
    md.append("## Key Differences from Previous Experiments")
    md.append("")
    md.append("- **Previous**: Used distractor setting (10 per-sample context paragraphs)")
    md.append("- **Current**: Uses full Wikipedia corpus (~5.2M articles) for open-domain retrieval")
    md.append("- **Original ReAct paper**: Uses live Wikipedia API; we approximate with offline BM25 over full Wikipedia")
    md.append("- **Sampling**: Follows original paper — shuffle with seed=233, take first 500")
    md.append("")
    md.append("## Results Summary")
    md.append("")
    md.append("| Method | EM (%) | F1 (%) | Avg Time/Sample (s) | Total Time |")
    md.append("|--------|--------|--------|---------------------|------------|")

    for r in results_table:
        if r["em"] < 0:
            md.append(f"| {r['method']} | N/A | N/A | N/A | N/A |")
        else:
            hours = r["total_time"] / 3600
            time_str = f"{r['total_time']:.0f}s ({hours:.1f}h)" if hours > 1 else f"{r['total_time']:.0f}s"
            md.append(f"| {r['method']} | {r['em']:.2f} | {r['f1']:.2f} | {r['avg_time']:.1f} | {time_str} |")

    md.append("")
    md.append("## Analysis")
    md.append("")

    # Find ReAct baseline and compare
    react_em = None
    react_f1 = None
    for r in results_table:
        if r["method"] == "ReAct (Full Wiki)" and r["em"] >= 0:
            react_em = r["em"]
            react_f1 = r["f1"]
            break

    if react_em is not None:
        md.append("### Comparison vs ReAct Baseline")
        md.append("")
        md.append("| Method | EM Δ | F1 Δ |")
        md.append("|--------|------|------|")
        for r in results_table:
            if r["em"] >= 0:
                em_diff = r["em"] - react_em
                f1_diff = r["f1"] - react_f1
                md.append(f"| {r['method']} | {em_diff:+.2f} | {f1_diff:+.2f} |")
        md.append("")

    # KV Cache specific analysis
    kv_methods = [r for r in results_table if "KV" in r["method"] and r["em"] >= 0]
    if kv_methods:
        md.append("### KV Cache Analysis")
        md.append("")
        for r in kv_methods:
            extra = r.get("extra", {})
            md.append(f"**{r['method']}**:")
            if extra.get("avg_kv_cache_length"):
                md.append(f"- Avg KV Cache Length: {extra['avg_kv_cache_length']:.0f}")
            if extra.get("max_kv_cache_length"):
                md.append(f"- Max KV Cache Length: {extra['max_kv_cache_length']}")
            if extra.get("avg_step_time"):
                md.append(f"- Avg Step Time: {extra['avg_step_time']:.2f}s")
            if r.get("prune_count", 0) > 0:
                md.append(f"- Total Prune Operations: {r['prune_count']}")
            md.append("")

    md.append("## Methodology Notes")
    md.append("")
    md.append("1. **Single Model**: Direct question → answer, no retrieval")
    md.append("2. **RAG**: Single BM25 retrieval pass → read → answer")
    md.append("3. **ReAct**: Multi-step interleaved reasoning with search/lookup/finish actions")
    md.append("4. **ReAct-KV (none)**: ReAct with KV cache reuse across steps (no pruning)")
    md.append("5. **ReAct-KV (H2O)**: ReAct-KV with Heavy Hitter Oracle pruning (keep_ratio=0.5)")
    md.append("6. **ReAct-KV (SnapKV)**: ReAct-KV with SnapKV attention pooling pruning")
    md.append("")
    md.append("All ReAct variants use the same 6-shot prompt from the original ReAct paper (Yao et al., 2022).")
    md.append("")

    report = "\n".join(md)
    with open(SUMMARY_PATH, "w") as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"Summary report saved to: {SUMMARY_PATH}")
    print(f"{'='*60}")
    print(report)


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser(description="Run all HotpotQA experiments with full Wikipedia corpus")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["single", "rag", "react", "react_kv_none",
                                 "react_kv_h2o", "react_kv_snapkv", "ours", "collect", "all"],
                        help="Which experiment to run")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory (default: wiki_0318_v2)")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    if args.experiment == "collect":
        collect_results()
        return

    # Load data
    val_data = load_hotpotqa_data()
    selected_samples = select_samples(val_data)

    # Experiments that need wiki retriever
    needs_retriever = args.experiment in ["rag", "react", "react_kv_none",
                                           "react_kv_h2o", "react_kv_snapkv", "ours", "all"]

    retriever = None
    if needs_retriever:
        from retrievers.WikiBM25Retriever import WikiBM25Retriever
        if not os.path.exists(os.path.join(WIKI_INDEX_DIR, "titles.json")):
            print(f"[ERROR] Wiki index not found at {WIKI_INDEX_DIR}. "
                  f"Run build_wiki_index.py first.")
            return
        retriever = WikiBM25Retriever(index_dir=WIKI_INDEX_DIR, load_corpus=True)

    if args.experiment == "single" or args.experiment == "all":
        run_single_experiment(
            val_data, selected_samples,
            os.path.join(output_dir, "single_wiki_500_0318.json"),
            os.path.join(output_dir, "single_wiki_500_0318_checkpoint.json"),
        )

    if args.experiment == "rag" or args.experiment == "all":
        run_rag_experiment(
            val_data, selected_samples, retriever,
            os.path.join(output_dir, "rag_wiki_500_0318.json"),
            os.path.join(output_dir, "rag_wiki_500_0318_checkpoint.json"),
        )

    if args.experiment == "react" or args.experiment == "all":
        run_react_experiment(
            val_data, selected_samples, retriever,
            os.path.join(output_dir, "react_wiki_500_0318.json"),
            os.path.join(output_dir, "react_wiki_500_0318_checkpoint.json"),
        )

    if args.experiment == "react_kv_none" or args.experiment == "all":
        run_react_kv_experiment(
            val_data, selected_samples, retriever, "none",
            os.path.join(output_dir, "react_kv_none_wiki_500_0318.json"),
            os.path.join(output_dir, "react_kv_none_wiki_500_0318_checkpoint.json"),
        )

    if args.experiment == "react_kv_h2o" or args.experiment == "all":
        run_react_kv_experiment(
            val_data, selected_samples, retriever, "h2o",
            os.path.join(output_dir, "react_kv_h2o_wiki_500_0318.json"),
            os.path.join(output_dir, "react_kv_h2o_wiki_500_0318_checkpoint.json"),
        )

    if args.experiment == "react_kv_snapkv" or args.experiment == "all":
        run_react_kv_experiment(
            val_data, selected_samples, retriever, "snapkv",
            os.path.join(output_dir, "react_kv_snapkv_wiki_500_0318.json"),
            os.path.join(output_dir, "react_kv_snapkv_wiki_500_0318_checkpoint.json"),
        )

    if args.experiment == "ours" or args.experiment == "all":
        run_react_kv_experiment(
            val_data, selected_samples, retriever,"ours",
            os.path.join(output_dir, "ours_wiki_500_0318.json"),
            os.path.join(output_dir, "ours_wiki_500_0318_checkpoint.json"),
        )

    if args.experiment == "all":
        collect_results()


if __name__ == "__main__":
    main()


# # Run each experiment independently:
# CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT --experiment single         # 1. Direct QA, no retrieval
# CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT --experiment rag            # 2. RAG (BM25 retrieve → read)
# CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT --experiment react          # 3. ReAct (multi-step reasoning)
# CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT --experiment react_kv_none  # 4. ReAct-KV (no pruning)
# CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT --experiment react_kv_h2o   # 5. ReAct-KV (H2O pruning)
# CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT --experiment react_kv_snapkv# 6. ReAct-KV (SnapKV pruning)
# CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT --experiment react_kv_ours  # 7. ReAct-KV (Ours, aggressive pruning)
# CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT --experiment collect        # Generate summary report
# CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT --experiment all            # Run ALL above sequentially 