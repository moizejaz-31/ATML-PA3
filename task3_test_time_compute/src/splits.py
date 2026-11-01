"""
Development and evaluation split generator for Task 3.
Lexicographically sorts 1,000 question IDs, then permutes them using
numpy.random.default_rng(6304). First 200: dev, remaining 800: held-out test.
First 50 MATH evaluation questions: live generation experiment.
"""
import json
import os
from typing import List, Dict, Optional
import numpy as np

def create_splits(question_ids: List[str], seed: int = 6304, save_path: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Sort lexicographically and permute with default_rng(seed).
    """
    sorted_ids = sorted(set(str(q) for q in question_ids))
    permuted = np.random.default_rng(seed).permutation(sorted_ids).tolist()

    splits = {
        "seed": seed,
        "dev": permuted[:200],
        "test": permuted[200:],
        "live": permuted[200:250],
    }
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=1)
    return splits
