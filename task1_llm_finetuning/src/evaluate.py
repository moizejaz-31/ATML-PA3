"""
Zero-shot benchmark evaluation for Task 1 through lm-evaluation-harness, with complete splits:
1. GSM8K (Test / 1,319): gsm8k_cot, non-thinking chat template, greedy generation, 1,024 cap, flexible-extraction EM.
2. HellaSwag (Validation / 10,042): Stock completion prompt, candidate likelihood, normalized accuracy.
3. BoolQ (Validation / 3,270): Stock completion prompt, candidate likelihood, accuracy.
4. ARC-Easy (Test / 2,376): Stock completion prompt, candidate likelihood, normalized accuracy.

Each condition writes <output_dir>/<condition>/<task>_scores.json and <task>_samples.jsonl.
Finished tasks are skipped on the next call, so an interrupted condition can continue.
A task that fails stops the run: no score is ever filled in.
"""
import json
import os
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from .conditions import CONDITIONS, load_condition_model, verify_merged_logits

GSM8K_TASK = "gsm8k_cot"
MAX_NEW_TOKENS = 1024
SEED = 6304
TASKS = [GSM8K_TASK, "hellaswag", "boolq", "arc_easy"]
EXPECTED_EXAMPLES = {GSM8K_TASK: 1319, "hellaswag": 10042, "boolq": 3270, "arc_easy": 2376}
# task -> (primary metric, harness result keys to report)
TASK_METRICS = {
    GSM8K_TASK: ("exact_match_flexible", {"exact_match_flexible": "exact_match,flexible-extract", "exact_match_strict": "exact_match,strict-match"}),
    "hellaswag": ("acc_norm", {"acc_norm": "acc_norm,none", "acc": "acc,none"}),
    "boolq": ("acc", {"acc": "acc,none"}),
    "arc_easy": ("acc_norm", {"acc_norm": "acc_norm,none", "acc": "acc,none"}),
}

def compact_generation_samples(samples: List[Dict[str, Any]], tokenizer) -> List[Dict[str, Any]]:
    """One row per GSM8K question: output, extracted answers per filter, and truncation flag."""
    rows: Dict[int, Dict[str, Any]] = {}
    for sample in samples:
        row = rows.setdefault(sample["doc_id"], {})
        if not row:
            text = sample["resps"][0][0]
            new_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            row.update({
                "doc_id": sample["doc_id"],
                "target": sample["target"],
                "prompt": sample["arguments"][0][0],
                "output": text,
                "new_tokens": new_tokens,
                "truncated": new_tokens >= MAX_NEW_TOKENS,
            })
        name = sample["filter"].replace("-", "_")
        row[f"extracted_{name}"] = sample["filtered_resps"][0]
        row[f"correct_{name}"] = float(sample["exact_match"])
    return [rows[doc_id] for doc_id in sorted(rows)]

def compact_likelihood_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per multiple-choice question: candidate log-likelihoods and per-example metrics."""
    rows = []
    for sample in samples:
        row = {
            "doc_id": sample["doc_id"],
            "target": sample["target"],
            "loglikelihoods": [float(resp[0]) for resp in sample["filtered_resps"]],
            "acc": float(sample["acc"]),
        }
        if "acc_norm" in sample:
            row["acc_norm"] = float(sample["acc_norm"])
        rows.append(row)
    return sorted(rows, key=lambda r: r["doc_id"])

def evaluate_task(model: nn.Module, tokenizer, task: str, batch_size: int, limit: Optional[int] = None):
    """Run one benchmark. Returns (scores, per-example rows). Any harness failure propagates."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    is_generation = task == GSM8K_TASK
    # The model stays in FP16; only the final log-softmax over the large vocabulary uses FP32
    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, enable_thinking=False, softmax_dtype="float32")
    output = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=[task],
        num_fewshot=0,
        fewshot_as_multiturn=False,
        limit=limit,
        log_samples=True,
        apply_chat_template=is_generation,
        gen_kwargs={"max_gen_toks": MAX_NEW_TOKENS, "do_sample": False} if is_generation else None,
        random_seed=SEED,
        numpy_random_seed=SEED,
        torch_random_seed=SEED,
        fewshot_random_seed=SEED,
        bootstrap_iters=0,
    )

    samples = output["samples"][task]
    if is_generation:
        rows = compact_generation_samples(samples, tokenizer)
    else:
        rows = compact_likelihood_samples(samples)

    if limit is None and len(rows) != EXPECTED_EXAMPLES[task]:
        raise ValueError(f"{task}: expected {EXPECTED_EXAMPLES[task]} examples, evaluated {len(rows)}")

    primary, keys = TASK_METRICS[task]
    harness_results = output["results"][task]
    scores = {name: float(harness_results[key]) * 100.0 for name, key in keys.items()}
    scores.update({"task": task, "primary_metric": primary, "primary": scores[primary], "n": len(rows)})
    if is_generation:
        scores["truncated_rate"] = 100.0 * sum(r["truncated"] for r in rows) / len(rows)
        scores["invalid_flexible_rate"] = 100.0 * sum(r["extracted_flexible_extract"] == "[invalid]" for r in rows) / len(rows)
        scores["mean_new_tokens"] = sum(r["new_tokens"] for r in rows) / len(rows)
    return scores, rows


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer
    from shared.src.seed import set_seed

    parser = argparse.ArgumentParser(description="Evaluate one Task 1 condition on the zero-shot benchmarks")
    parser.add_argument("--condition", type=str, required=True, choices=list(CONDITIONS))
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--checkpoint_dir", type=str, default="./task1_llm_finetuning/checkpoints")
    parser.add_argument("--output_dir", type=str, default="./task1_llm_finetuning/results/eval")
    parser.add_argument("--tasks", type=str, nargs="*", default=TASKS)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Smoke tests only: evaluate fewer examples")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    condition_dir = os.path.join(args.output_dir, args.condition)
    os.makedirs(condition_dir, exist_ok=True)

    print(f"[Evaluate CLI] Device: {device} | Condition: {args.condition}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pending = [t for t in args.tasks if not os.path.exists(os.path.join(condition_dir, f"{t}_scores.json"))]
    if not pending:
        print(f"[Evaluate CLI] {args.condition} already has all requested tasks.")
        raise SystemExit(0)

    merge_check_path = os.path.join(condition_dir, "merge_check.json")
    if not os.path.exists(merge_check_path):
        sample = tokenizer("The capital of France is Paris. 2 + 2 = 4.", return_tensors="pt", add_special_tokens=False)
        check = verify_merged_logits(args.condition, args.checkpoint_dir, args.model_name, sample["input_ids"])
        with open(merge_check_path, "w", encoding="utf-8") as f:
            json.dump(check, f, indent=2)
        print(f"[Evaluate CLI] Merge check: {check}")

    model = load_condition_model(args.condition, args.checkpoint_dir, args.model_name, device)

    for task in pending:
        print(f"[Evaluate CLI] Running {task}...")
        scores, rows = evaluate_task(model, tokenizer, task, args.batch_size, args.limit)
        scores["condition"] = args.condition
        samples_path = os.path.join(condition_dir, f"{task}_samples.jsonl")
        with open(samples_path, "w", encoding="utf-8") as f:
            for row in rows:
                # default=str: harness targets can be NumPy values, and a finished evaluation must not be lost to them
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        # Scores go last: their presence marks the task as finished
        with open(os.path.join(condition_dir, f"{task}_scores.json"), "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2)
        print(f"[Evaluate CLI] {args.condition} {task}: {scores['primary_metric']} = {scores['primary']:.2f}% (n={scores['n']})")
