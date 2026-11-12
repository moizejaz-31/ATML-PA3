"""
Qualitative behavior generation for Task 1.
24 prompts from first_plot_questions.yaml (8 questions x plain, JSON, template), one greedy response
per prompt with thinking disabled and a 512-new-token cap, for each of the 7 conditions (168 responses).
Also writes a blank annotation sheet for the 42-response subset. Annotations are written by hand.
"""
import csv
import json
import os
from typing import Any, Dict, List
import torch
import yaml

from .conditions import CONDITIONS, load_condition_model
from .data import build_prompt

MAX_NEW_TOKENS = 512
ANNOTATION_COLUMNS = ["format_compliance", "coherence_relevance", "repetition", "unnecessary_math_framing", "inappropriate_content", "excerpt", "notes"]

def load_qualitative_prompts(path: str) -> List[Dict[str, Any]]:
    """Flatten the question file into one entry per prompt, keeping the file's question order."""
    with open(path, "r", encoding="utf-8") as f:
        entries = yaml.safe_load(f)

    prompts = []
    plain_order: List[str] = []
    for entry in entries:
        entry_id = entry["id"]
        if entry_id.endswith("_json"):
            question, fmt = entry_id[:-len("_json")], "json"
        elif entry_id.endswith("_template"):
            question, fmt = entry_id[:-len("_template")], "template"
        else:
            question, fmt = entry_id, "plain"
            plain_order.append(question)
        prompts.append({
            "question": question,
            "format": fmt,
            "system": entry.get("system"),
            "user": entry["paraphrases"][0],
        })

    if len(prompts) != 24 or len(plain_order) != 8:
        raise ValueError(f"Expected 24 prompts over 8 questions, found {len(prompts)} over {len(plain_order)}")
    for prompt in prompts:
        prompt["question_index"] = plain_order.index(prompt["question"])
    return prompts

@torch.no_grad()
def generate_responses(model, tokenizer, prompts: List[Dict[str, Any]], condition: str, device: torch.device) -> List[Dict[str, Any]]:
    rows = []
    for prompt in prompts:
        text = build_prompt(tokenizer, prompt["user"], system=prompt["system"])
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0][enc["input_ids"].shape[1]:]
        finished = out[-1].item() == tokenizer.eos_token_id
        rows.append({
            "condition": condition,
            **{k: prompt[k] for k in ["question", "question_index", "format"]},
            "response": tokenizer.decode(out, skip_special_tokens=True),
            "new_tokens": int(out.shape[0]),
            "truncated": not finished,
        })
    return rows

def write_annotation_sheet(path: str, rows: List[Dict[str, Any]]) -> int:
    """Blank sheet for the first and last questions. An existing sheet holds hand annotations, so it is kept."""
    if os.path.exists(path):
        print(f"[Qualitative] Keeping existing annotation sheet {path}")
        return 0
    last_index = max(r["question_index"] for r in rows)
    subset = [r for r in rows if r["question_index"] in (0, last_index)]
    subset.sort(key=lambda r: (r["question_index"], r["format"], list(CONDITIONS).index(r["condition"])))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "format", "condition", "truncated", "response"] + ANNOTATION_COLUMNS)
        writer.writeheader()
        for r in subset:
            writer.writerow({k: r[k] for k in ["question", "format", "condition", "truncated", "response"]})
    return len(subset)


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer
    from shared.src.seed import set_seed

    parser = argparse.ArgumentParser(description="Generate the 168 qualitative responses for Task 1")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--checkpoint_dir", type=str, default="./task1_llm_finetuning/checkpoints")
    parser.add_argument("--output_dir", type=str, default="./task1_llm_finetuning/results/qualitative")
    parser.add_argument("--questions", type=str, default="./task1_llm_finetuning/configs/first_plot_questions.yaml")
    parser.add_argument("--conditions", type=str, nargs="*", default=list(CONDITIONS))
    parser.add_argument("--limit", type=int, default=None, help="Smoke tests only: generate for fewer prompts")
    args = parser.parse_args()

    set_seed(6304)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    prompts = load_qualitative_prompts(args.questions)[:args.limit]
    os.makedirs(args.output_dir, exist_ok=True)

    for condition in args.conditions:
        path = os.path.join(args.output_dir, f"{condition}.jsonl")
        if os.path.exists(path):
            print(f"[Qualitative] {condition} already generated.")
            continue
        print(f"[Qualitative] Generating {len(prompts)} responses for {condition}...")
        model = load_condition_model(condition, args.checkpoint_dir, args.model_name, device)
        rows = generate_responses(model, tokenizer, prompts, condition, device)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(path + ".tmp", path)

    # Group all responses by question and format once every condition exists
    if all(os.path.exists(os.path.join(args.output_dir, f"{c}.jsonl")) for c in CONDITIONS):
        all_rows = []
        for condition in CONDITIONS:
            with open(os.path.join(args.output_dir, f"{condition}.jsonl"), "r", encoding="utf-8") as f:
                all_rows.extend(json.loads(line) for line in f)
        all_rows.sort(key=lambda r: (r["question_index"], r["format"], list(CONDITIONS).index(r["condition"])))
        with open(os.path.join(args.output_dir, "all_responses.json"), "w", encoding="utf-8") as f:
            json.dump(all_rows, f, ensure_ascii=False, indent=1)
        n = write_annotation_sheet(os.path.join(args.output_dir, "annotations_42.csv"), all_rows)
        print(f"[Qualitative] Saved {len(all_rows)} responses; annotation sheet has {n} new rows to annotate by hand.")
