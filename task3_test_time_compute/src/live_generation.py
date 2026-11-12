"""
Live generation harness for Qwen2.5-3B-Instruct on 50 MATH evaluation problems.
Compares, under one 3,072-token output allowance per question:
1. Single longer response (3,072 new tokens)
2. Parallel sampling (3 x 1,024 new tokens + self-consistency)
3. Sequential refinement (1,024 tokens + 2,048 tokens critique and revision)
The first parallel response is reused as the refinement's initial response.
Finished questions are saved immediately, so an interrupted run can continue.
"""
import json
import os
import time
from typing import Any, Dict, List, Optional
import torch

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
SYSTEM_MESSAGE = "You are a helpful assistant."
THINK_SUFFIX = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
REVIEW_INSTRUCTION = "Review the solution above. Identify and correct any errors; if it is correct, retain its answer. Give your final answer inside \\boxed{}."
SAMPLING = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.0}
BASE_SEED = 6304

def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract last \\boxed{...} answer using balanced braces. Absent, empty, or unclosed gives None.
    """
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip() or None
    return None

def first_messages(problem: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": f"{problem}\n\n{THINK_SUFFIX}"}]

@torch.no_grad()
def generate_once(model, tokenizer, messages: List[Dict[str, str]], max_new_tokens: int, seed: int) -> Dict[str, Any]:
    """One independently seeded call. Returns the prompt, response, token counts, truncation flag, and time."""
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    start = time.perf_counter()
    output = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, **SAMPLING)[0][enc["input_ids"].shape[1]:]
    elapsed = time.perf_counter() - start
    finished = output[-1].item() in set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list) else [model.generation_config.eos_token_id])
    response = tokenizer.decode(output, skip_special_tokens=True)
    return {"prompt": prompt, "seed": seed, "response": response, "input_tokens": int(enc["input_ids"].shape[1]),
            "output_tokens": int(output.shape[0]), "truncated": not finished, "seconds": elapsed, "extracted": extract_boxed_answer(response)}

def run_question(model, tokenizer, sample_id: str, problem: str, gold: str, question_index: int) -> Dict[str, Any]:
    """All generations for one question: one long response, three parallel, and one revision of the first parallel."""
    base = BASE_SEED + 100 * question_index
    single = generate_once(model, tokenizer, first_messages(problem), 3072, base)
    parallel = [generate_once(model, tokenizer, first_messages(problem), 1024, base + 1 + i) for i in range(3)]
    revision_messages = first_messages(problem) + [{"role": "assistant", "content": parallel[0]["response"]}, {"role": "user", "content": REVIEW_INSTRUCTION}]
    revision = generate_once(model, tokenizer, revision_messages, 2048, base + 10)
    return {"sample_id": sample_id, "gold": gold, "problem": problem, "single": single, "parallel": parallel, "revision": revision}

def run_live_experiment(questions: List[Dict[str, str]], output_path: str, model_name: str = MODEL_NAME, revision: Optional[str] = MODEL_REVISION) -> int:
    """FP16 weights, no quantization, batch size 1. Appends one JSON line per finished question."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            done = {json.loads(line)["sample_id"] for line in f}
    pending = [(i, q) for i, q in enumerate(questions) if q["sample_id"] not in done]
    if not pending:
        return 0

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, torch_dtype=dtype)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    for index, question in pending:
        record = run_question(model, tokenizer, question["sample_id"], question["problem"], question["gold"], index)
        record["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[Live] {len(done) + 1}/{len(questions)} {question['sample_id']}", flush=True)
        done.add(question["sample_id"])
    return len(pending)

def summarize_live(records: List[Dict[str, Any]], equivalent) -> Dict[str, Any]:
    """Per-question table for the three strategies, and revision transitions relative to the shared initial answer."""
    from .aggregation import select_self_consistency
    from .normalization import NO_ANSWER, group_answers, normalize_math_answer
    rows = []
    for record in records:
        gold = normalize_math_answer(record["gold"])
        grade = lambda answer: answer != NO_ANSWER and equivalent(gold, answer)
        answers = [normalize_math_answer(p["extracted"]) for p in record["parallel"]]
        mapping = group_answers(answers, equivalent)
        parallel_answer = select_self_consistency([mapping[a] for a in answers])
        initial = answers[0]
        revised = normalize_math_answer(record["revision"]["extracted"])
        single = normalize_math_answer(record["single"]["extracted"])

        def cost(parts):
            return {"input_tokens": sum(p["input_tokens"] for p in parts), "output_tokens": sum(p["output_tokens"] for p in parts), "seconds": sum(p["seconds"] for p in parts)}

        shared = record["parallel"][0]
        strategies = {
            "single_long": (single, [record["single"]]),
            "parallel": (parallel_answer, record["parallel"]),
            "refinement": (revised, [shared, record["revision"]]),
        }
        for name, (answer, parts) in strategies.items():
            rows.append({"sample_id": record["sample_id"], "strategy": name, "correct": grade(answer), "extracted": answer != NO_ANSWER,
                         "truncated": any(p["truncated"] for p in parts), **cost(parts)})
        rows.append({"sample_id": record["sample_id"], "strategy": "initial_only", "correct": grade(initial), "extracted": initial != NO_ANSWER,
                     "truncated": shared["truncated"], **cost([shared])})
    return rows
