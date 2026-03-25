import re
import torch
from run_single_model import (
    load_hotpotqa_data, select_samples, run_no_kv, create_llm,
    exact_match, f1_score, MAX_STEPS, BM25_TOP_K, CHECKPOINT_INTERVAL,
    NUM_SAMPLES, RANDOM_SEED, DATA_CACHE_DIR
)
import gc
import os
import argparse
from typing import Optional


MODEL_PATH = "Qwen2.5-7B-Instruct"

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

def run_react_experiment(val_data, selected_samples, retriever, output_path, checkpoint_path):
    """ReAct: multi-step reasoning with Wikipedia BM25 retrieval."""

    # create llm via run_single_model helper to ensure consistent LLM usage
    llm = create_llm(MODEL_PATH)

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

    try:
        del llm
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

    return final_em, final_f1, total_time