"""
Data loading, preprocessing, GSM8K filtering, and prompt masking for Task 1 (MetaMathQA fine-tuning).
Model: Qwen/Qwen3-0.6B (post-trained, enable_thinking=False).
Seed: 6304.
"""
import re
import json
import os
from typing import List, Dict, Any, Iterable, Optional, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np

METAMATH_DATASET = "meta-math/MetaMathQA"
GSM8K_DATASET = "openai/gsm8k"
HELLASWAG_DATASET = "Rowan/hellaswag"
GSM8K_TEST_SIZE = 1319
HELLASWAG_VAL_SIZE = 10042
END_OF_TURN = "<|im_end|>"

def normalize_text(text: str) -> str:
    """Normalize text for GSM8K overlap filtering (lowercase, strip, single whitespace)."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def build_prompt(tokenizer, question: str, system: Optional[str] = None) -> str:
    """User turn plus the non-thinking generation prompt, which ends in an empty think block."""
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

def tokenize_example(tokenizer, question: str, response: str) -> Tuple[List[int], List[int]]:
    """Tokenize prompt and response separately so that the boundary is exact."""
    prompt_ids = tokenizer.encode(build_prompt(tokenizer, question), add_special_tokens=False)
    response_ids = tokenizer.encode(f"{response}{END_OF_TURN}", add_special_tokens=False)
    return prompt_ids, response_ids

def filter_metamath_examples(
    examples: Iterable[Dict[str, Any]],
    gsm8k_test_questions: Optional[List[str]] = None,
    tokenizer = None,
    max_length: int = 2048,
    target_count: int = 40000,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Select the first 40,000 eligible MetaMathQA examples in source order.
    Exclusion criteria:
      1. Empty query or response.
      2. Query or original_question matches any GSM8K test question after text normalization.
      3. Formatted token length > 2,048 tokens.
    Returns:
      source indices of the eligible examples, filter_statistics
    """
    gsm8k_set = set(normalize_text(q) for q in (gsm8k_test_questions or []))
    eligible = []
    stats = {
        "total_scanned": 0,
        "empty_filtered": 0,
        "gsm8k_overlap_filtered": 0,
        "length_filtered": 0,
        "retained": 0,
    }
    prompt_lengths, response_lengths = [], []

    for source_idx, ex in enumerate(examples):
        stats["total_scanned"] += 1
        query = ex.get("query") or ""
        response = ex.get("response") or ""

        if not query.strip() or not response.strip():
            stats["empty_filtered"] += 1
            continue

        if normalize_text(query) in gsm8k_set or normalize_text(ex.get("original_question") or "") in gsm8k_set:
            stats["gsm8k_overlap_filtered"] += 1
            continue

        if tokenizer is not None:
            prompt_ids, response_ids = tokenize_example(tokenizer, query, response)
            if len(prompt_ids) + len(response_ids) > max_length:
                stats["length_filtered"] += 1
                continue
            prompt_lengths.append(len(prompt_ids))
            response_lengths.append(len(response_ids))

        eligible.append(source_idx)
        stats["retained"] += 1
        if len(eligible) >= target_count:
            break

    if prompt_lengths:
        stats["prompt_lengths"] = prompt_lengths
        stats["response_lengths"] = response_lengths
    return eligible, stats

def format_prompt_and_response(
    tokenizer,
    question: str,
    response: str,
    max_length: int = 2048,
) -> Dict[str, torch.Tensor]:
    """
    Construct input_ids, attention_mask, and labels with assistant-response masking.
    Next-token cross-entropy only on assistant response tokens and the end-of-turn token.
    Prompts and padding tokens are masked with -100. Answers are never truncated.
    """
    prompt_ids, response_ids = tokenize_example(tokenizer, question, response)
    full_ids = prompt_ids + response_ids
    if len(full_ids) > max_length:
        raise ValueError(f"Example of {len(full_ids)} tokens passed the {max_length}-token eligibility filter")

    labels = [-100] * len(prompt_ids) + response_ids

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

class MetaMathDataset(Dataset):
    """PyTorch Dataset for tokenized MetaMathQA examples."""
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        return format_prompt_and_response(
            self.tokenizer,
            question=item["query"],
            response=item["response"],
            max_length=self.max_length,
        )

def collate_fn_pad(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """Pad microbatch to longest sequence in batch."""
    batch_size = len(batch)
    max_len = max(x["input_ids"].shape[0] for x in batch)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        labels[i, :seq_len] = item["labels"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def load_gsm8k_test():
    """Load the complete GSM8K test split. A failure here must stop the run, not disable the overlap filter."""
    from datasets import load_dataset
    gsm_test = load_dataset(GSM8K_DATASET, "main", split="test")
    if len(gsm_test) != GSM8K_TEST_SIZE:
        raise ValueError(f"Expected {GSM8K_TEST_SIZE} GSM8K test questions, found {len(gsm_test)}")
    return gsm_test

def load_hellaswag_validation():
    """Load the complete HellaSwag validation split."""
    from datasets import load_dataset
    hella_val = load_dataset(HELLASWAG_DATASET, split="validation")
    if len(hella_val) != HELLASWAG_VAL_SIZE:
        raise ValueError(f"Expected {HELLASWAG_VAL_SIZE} HellaSwag validation contexts, found {len(hella_val)}")
    return hella_val

def sample_activation_probe_set(
    n_gsm8k: int,
    n_hellaswag: int,
    seed: int = 6304,
    n_samples: int = 32,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Sample 32 GSM8K test questions and 32 HellaSwag validation contexts with seed 6304.
    Saves example IDs to JSON.
    """
    rng = np.random.default_rng(seed)

    gsm8k_indices = rng.choice(n_gsm8k, size=min(n_samples, n_gsm8k), replace=False).tolist()
    hellaswag_indices = rng.choice(n_hellaswag, size=min(n_samples, n_hellaswag), replace=False).tolist()

    probe_manifest = {
        "seed": seed,
        "description": f"Activation probe set of {n_samples} GSM8K test questions and {n_samples} HellaSwag validation contexts",
        "n_gsm8k": len(gsm8k_indices),
        "n_hellaswag": len(hellaswag_indices),
        "gsm8k_indices": gsm8k_indices,
        "hellaswag_indices": hellaswag_indices,
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(probe_manifest, f, indent=2)

    return probe_manifest


def prepare_metamath_dataset(
    dataset_name_or_path: Any = METAMATH_DATASET,
    gsm8k_test_questions: Optional[List[str]] = None,
    tokenizer: Any = None,
    max_length: int = 2048,
    target_count: int = 40000,
    seed: int = 6304,
    ids_path: Optional[str] = None,
) -> Tuple[MetaMathDataset, Dict[str, Any]]:
    """
    Download/load MetaMathQA, apply three-tier exclusion filter, and return MetaMathDataset.
    The selected source indices are saved to ids_path and reused by every later run.
    """
    if isinstance(dataset_name_or_path, list):
        raw_examples = dataset_name_or_path
    else:
        from datasets import load_dataset
        raw_examples = load_dataset(dataset_name_or_path, split="train")

    if ids_path and os.path.exists(ids_path):
        with open(ids_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if saved["target_count"] == target_count and saved["max_length"] == max_length:
            print(f"[Data] Reusing {len(saved['source_indices'])} eligible example IDs from {ids_path}")
            examples = [raw_examples[i] for i in saved["source_indices"]]
            return MetaMathDataset(examples, tokenizer=tokenizer, max_length=max_length), saved["stats"]

    if gsm8k_test_questions is None:
        gsm8k_test_questions = [ex["question"] for ex in load_gsm8k_test()]

    source_indices, stats = filter_metamath_examples(
        examples=raw_examples,
        gsm8k_test_questions=gsm8k_test_questions,
        tokenizer=tokenizer,
        max_length=max_length,
        target_count=target_count,
    )
    if len(source_indices) < target_count:
        raise ValueError(f"Only {len(source_indices)} eligible examples found, {target_count} required")

    prompt_lengths = stats.pop("prompt_lengths", [])
    response_lengths = stats.pop("response_lengths", [])
    if prompt_lengths:
        totals = np.array(prompt_lengths) + np.array(response_lengths)
        stats["total_length_percentiles"] = {str(q): float(np.percentile(totals, q)) for q in [0, 50, 90, 99, 100]}
        stats["prompt_token_mean"] = float(np.mean(prompt_lengths))
        stats["response_token_mean"] = float(np.mean(response_lengths))

    if ids_path:
        os.makedirs(os.path.dirname(ids_path), exist_ok=True)
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump({"target_count": target_count, "max_length": max_length, "stats": stats, "source_indices": source_indices}, f)

    examples = [raw_examples[i] for i in source_indices]
    return MetaMathDataset(examples, tokenizer=tokenizer, max_length=max_length), stats


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer
    from shared.src.seed import set_seed

    parser = argparse.ArgumentParser(description="Prepare MetaMathQA dataset and activation probe set")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--target_count", type=int, default=40000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=6304)
    parser.add_argument("--probe_path", type=str, default="task1_llm_finetuning/splits/probe_set_seed6304.json")
    parser.add_argument("--ids_path", type=str, default="task1_llm_finetuning/splits/metamath_ids_seed6304.json")
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"[Data CLI] Initializing with Seed {args.seed}...")
    print(f"[Data CLI] Loading tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"[Data CLI] Preparing MetaMathQA ({args.target_count} examples, max_len={args.max_length})...")
    ds, stats = prepare_metamath_dataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        target_count=args.target_count,
        seed=args.seed,
        ids_path=args.ids_path,
    )
    print(f"[Data CLI] Dataset preparation complete! Stats: {stats}")

    print(f"[Data CLI] Sampling activation probe set (Seed {args.seed})...")
    sample_activation_probe_set(
        n_gsm8k=len(load_gsm8k_test()),
        n_hellaswag=len(load_hellaswag_validation()),
        seed=args.seed,
        n_samples=32,
        save_path=args.probe_path,
    )
    print(f"[Data CLI] Probe manifest saved to {args.probe_path}")
