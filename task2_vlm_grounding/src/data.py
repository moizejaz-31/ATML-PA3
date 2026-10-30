"""
Data preparation, programmatic question generation, and split curation for Task 2.
Curates:
  1. COCO Captions (6,000 train images x 3 = 18,000 captions; 96 dev captions)
  2. Annotation-Derived Grounding Pairs (Presence yes/no, Counting 1-4)
  3. Natural VQAv2 Complementary Pairs (Filtered for >= 8/10 annotator consensus)
"""
import re
import json
import os
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import torch
from torch.utils.data import Dataset

def normalize_vqa_answer(text: str) -> str:
    """
    Standard VQAv2 answer normalization:
    lowercase, remove punctuation, articles (a, an, the), and standardize numbers.
    """
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Normalize common digits
    digit_map = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                 "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
    words = [digit_map.get(w, w) for w in text.split()]
    return " ".join(words)

def generate_presence_pair(
    category: str,
    image_with_obj: str,
    image_without_obj: str,
) -> Dict[str, Any]:
    """
    Constructs a complementary presence question pair.
    Shared Question: 'Does this image contain [category]?'
    Image A: contains category -> Answer 'yes'
    Image B: lacks category -> Answer 'no'
    """
    question = f"Does this image contain {category}?"
    return {
        "type": "presence",
        "category": category,
        "question": question,
        "image_a": {
            "image_id": image_with_obj,
            "target": "yes",
            "all_answers": ["yes"] * 10,
        },
        "image_b": {
            "image_id": image_without_obj,
            "target": "no",
            "all_answers": ["no"] * 10,
        }
    }

def generate_counting_pair(
    category: str,
    image_count_a: str,
    count_a: int,
    image_count_b: str,
    count_b: int,
) -> Dict[str, Any]:
    """
    Constructs a complementary counting question pair.
    Shared Question: 'How many [category] can you see?'
    Image A: count_a (1-4)
    Image B: count_b (1-4, count_b != count_a)
    """
    plural_cat = category if category.endswith("s") else f"{category}s"
    question = f"How many {plural_cat} can you see?"
    return {
        "type": "counting",
        "category": category,
        "question": question,
        "image_a": {
            "image_id": image_count_a,
            "target": str(count_a),
            "all_answers": [str(count_a)] * 10,
        },
        "image_b": {
            "image_id": image_count_b,
            "target": str(count_b),
            "all_answers": [str(count_b)] * 10,
        }
    }

def filter_vqav2_pair(
    pair_candidate: Dict[str, Any],
    min_consensus: int = 8,
) -> Optional[Dict[str, Any]]:
    """
    Verifies that a complementary VQAv2 pair has:
      1. Identical question string.
      2. Distinct normalized consensus answers.
      3. At least min_consensus (8 of 10) human annotators agreeing on each image's consensus.
    """
    ans_a = [normalize_vqa_answer(a) for a in pair_candidate["answers_a"]]
    ans_b = [normalize_vqa_answer(a) for a in pair_candidate["answers_b"]]

    if not ans_a or not ans_b:
        return None

    # Count consensus
    from collections import Counter
    top_a, count_a = Counter(ans_a).most_common(1)[0]
    top_b, count_b = Counter(ans_b).most_common(1)[0]

    if count_a < min_consensus or count_b < min_consensus:
        return None

    if top_a == top_b:
        return None

    return {
        "type": "vqav2",
        "question_id": pair_candidate.get("question_id", "vqa_pair"),
        "question": pair_candidate["question"],
        "image_a": {
            "image_id": pair_candidate["image_a"],
            "target": top_a,
            "all_answers": pair_candidate["answers_a"],
        },
        "image_b": {
            "image_id": pair_candidate["image_b"],
            "target": top_b,
            "all_answers": pair_candidate["answers_b"],
        }
    }

class VLMDataset(Dataset):
    """
    PyTorch Dataset serving (visual_features, input_ids, labels) tuples.
    """
    def __init__(self, examples: List[Dict[str, Any]], feature_cache_dir: str, tokenizer, max_seq_len: int = 128):
        self.examples = examples
        self.feature_cache_dir = feature_cache_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        image_id = ex["image_id"]
        feat_path = os.path.join(self.feature_cache_dir, f"{image_id}.pt")
        if os.path.exists(feat_path):
            visual_feat = torch.load(feat_path, map_location="cpu")
        else:
            # Fallback zero features
            visual_feat = torch.zeros((196, 768), dtype=torch.float16)

        prompt = ex["prompt"]
        target = ex["target"]
        full_text = prompt + target + "<|im_end|>"

        enc_prompt = self.tokenizer(prompt, return_tensors="pt")
        enc_full = self.tokenizer(full_text, return_tensors="pt", max_length=self.max_seq_len, truncation=True)

        input_ids = enc_full["input_ids"][0]
        attention_mask = enc_full["attention_mask"][0]
        labels = input_ids.clone()
        # Mask prompt tokens with -100
        prompt_len = min(enc_prompt["input_ids"].shape[1], len(labels))
        labels[:prompt_len] = -100

        return {
            "visual_features": visual_feat,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def build_vlm_collate_fn(pad_token_id: int = 0):
    def collate_fn(batch):
        max_len = max(x["input_ids"].shape[0] for x in batch)
        B = len(batch)

        input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        labels = torch.full((B, max_len), -100, dtype=torch.long)
        visual_features = torch.stack([x["visual_features"] for x in batch])

        for i, item in enumerate(batch):
            L = item["input_ids"].shape[0]
            input_ids[i, :L] = item["input_ids"]
            attention_mask[i, :L] = item["attention_mask"]
            labels[i, :L] = item["labels"]

        return {
            "visual_features": visual_features,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    return collate_fn
