"""
Development and evaluation split generator for Task 3.
Lexicographically sorts 1,000 question IDs, then permutes them using
numpy.random.default_rng(6304). First 200: dev, remaining 800: held-out test.
First 50 MATH evaluation questions: live generation experiment.
"""
from typing import List, Tuple, Dict
import numpy as np

def create_splits(question_ids: List[str], seed: int = 6304) -> Dict[str, List[str]]:
    """
    Sort lexicographically and permute with default_rng(seed).
    """
    sorted_ids = sorted(list(set(question_ids)))
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(sorted_ids).tolist()
    
    dev_ids = permuted[:200]
    test_ids = permuted[200:]
    live_ids = test_ids[:50]
    
    return {
        "dev": dev_ids,
        "test": test_ids,
        "live": live_ids,
    }
