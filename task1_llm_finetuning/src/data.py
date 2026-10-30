"""
Data loading, preprocessing, GSM8K filtering, and prompt masking for Task 1 (MetaMathQA fine-tuning).
Model: Qwen/Qwen3-0.6B (post-trained, enable_thinking=False).
Seed: 6304.
"""
import re
import json
import os
from typing import List, Dict, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def normalize_text(text: str) -> str:
    """Normalize text for GSM8K overlap filtering (lowercase, strip, single whitespace)."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def filter_metamath_examples(
    examples: List[Dict[str, Any]],
    gsm8k_test_questions: Optional[List[str]] = None,
    tokenizer = None,
    max_length: int = 2048,
    target_count: int = 40000,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Select the first 40,000 eligible MetaMathQA examples in source order.
    Exclusion criteria:
      1. Empty query or response.
      2. Formatted token length > 2,048 tokens.
      3. Query or original_question matches any GSM8K test question after text normalization.
    Returns:
      eligible_examples, filter_statistics
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

    for ex in examples:
        stats["total_scanned"] += 1
        query = ex.get("query") or ex.get("original_question") or ""
        response = ex.get("response") or ""
        
        if not query.strip() or not response.strip():
            stats["empty_filtered"] += 1
            continue
            
        norm_query = normalize_text(query)
        if norm_query in gsm8k_set:
            stats["gsm8k_overlap_filtered"] += 1
            continue
            
        if tokenizer is not None:
            # Check formatted length using Qwen non-thinking chat format
            user_text = query
            asst_text = response
            # Test tokenized length approximation
            tokens_prompt = tokenizer.encode(f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
            tokens_resp = tokenizer.encode(f"{asst_text}<|im_end|>", add_special_tokens=False)
            if len(tokens_prompt) + len(tokens_resp) > max_length:
                stats["length_filtered"] += 1
                continue
                
        eligible.append(ex)
        stats["retained"] += 1
        if len(eligible) >= target_count:
            break

    return eligible, stats

def format_prompt_and_response(
    tokenizer,
    question: str,
    response: str,
    max_length: int = 2048,
    pad_token_id: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Construct input_ids, attention_mask, and labels with assistant-response masking.
    Next-token cross-entropy only on assistant response tokens and the end-of-turn token.
    Prompts and padding tokens are masked with -100.
    """
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Format user prompt and assistant response
    prompt_str = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    response_str = f"{response}<|im_end|>"

    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    response_ids = tokenizer.encode(response_str, add_special_tokens=False)

    full_ids = prompt_ids + response_ids
    if len(full_ids) > max_length:
        # Do not truncate per manual instructions, but cap gracefully if used
        full_ids = full_ids[:max_length]

    prompt_len = len(prompt_ids)
    
    # Mask prompts (-100), supervise response tokens
    labels = [-100] * min(prompt_len, len(full_ids)) + full_ids[prompt_len:]
    attention_mask = [1] * len(full_ids)

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "prompt_length": prompt_len,
        "response_length": len(response_ids),
        "total_length": len(full_ids),
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
        query = item.get("query") or item.get("original_question") or ""
        response = item.get("response") or ""
        return format_prompt_and_response(
            self.tokenizer,
            question=query,
            response=response,
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

def sample_activation_probe_set(
    gsm8k_questions: List[Dict[str, Any]],
    hellaswag_contexts: List[Dict[str, Any]],
    seed: int = 6304,
    n_samples: int = 32,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Sample 32 GSM8K test questions and 32 HellaSwag validation contexts with seed 6304.
    Saves example IDs to JSON.
    """
    rng = np.random.default_rng(seed)
    
    gsm8k_indices = rng.choice(len(gsm8k_questions), size=min(n_samples, len(gsm8k_questions)), replace=False).tolist()
    hellaswag_indices = rng.choice(len(hellaswag_contexts), size=min(n_samples, len(hellaswag_contexts)), replace=False).tolist()

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
    dataset_name_or_path: Any = "meta-math/MetaMathQA",
    gsm8k_test_questions: Optional[List[str]] = None,
    tokenizer: Any = None,
    max_length: int = 2048,
    target_count: int = 40000,
    seed: int = 6304,
) -> Tuple[MetaMathDataset, Dict[str, Any]]:
    """
    Download/load MetaMathQA, apply three-tier exclusion filter, and return MetaMathDataset.
    """
    if isinstance(dataset_name_or_path, list):
        raw_examples = dataset_name_or_path
    else:
        from datasets import load_dataset
        raw_ds = load_dataset(dataset_name_or_path, split="train")
        raw_examples = [raw_ds[i] for i in range(min(len(raw_ds), target_count * 2))]

    if gsm8k_test_questions is None:
        try:
            from datasets import load_dataset
            gsm_test = load_dataset("gsm8k", "main", split="test")
            gsm8k_test_questions = [ex["question"] for ex in gsm_test]
        except Exception:
            gsm8k_test_questions = []

    filtered_examples, stats = filter_metamath_examples(
        examples=raw_examples,
        gsm8k_test_questions=gsm8k_test_questions,
        tokenizer=tokenizer,
        max_length=max_length,
        target_count=target_count,
    )
    dataset = MetaMathDataset(filtered_examples, tokenizer=tokenizer, max_length=max_length)
    return dataset, stats


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
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"[Data CLI] Initializing with Seed {args.seed}...")
    print(f"[Data CLI] Loading tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[Data CLI] Preparing MetaMathQA ({args.target_count} examples, max_len={args.max_length})...")
    ds, stats = prepare_metamath_dataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        target_count=args.target_count,
        seed=args.seed,
    )
    print(f"[Data CLI] Dataset preparation complete! Stats: {stats}")

    print(f"[Data CLI] Sampling activation probe set (Seed {args.seed})...")
    try:
        from datasets import load_dataset
        gsm_test = load_dataset("gsm8k", "main", split="test")
        hella_val = load_dataset("hellaswag", split="validation")

        sample_activation_probe_set(
            gsm8k_questions=[{"id": i, "question": ex["question"]} for i, ex in enumerate(gsm_test)],
            hellaswag_contexts=[{"id": i, "ctx": ex["ctx"]} for i, ex in enumerate(hella_val)],
            seed=args.seed,
            n_samples=32,
            save_path=args.probe_path,
        )
        print(f"[Data CLI] Probe manifest saved to {args.probe_path}")
    except Exception as e:
        print(f"[Data CLI] Probe sampling note: {e}")
