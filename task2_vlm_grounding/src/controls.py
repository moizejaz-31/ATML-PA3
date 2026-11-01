"""
Four Image Control Conditions for Task 2 Visual Grounding Evaluation. The question and original
references stay fixed across conditions:
1. correct: The original image.
2. swap: The partner image from the same complementary pair.
3. mismatched: An unrelated image from another pair in the same suite.
4. no_image: Visual tokens omitted; text instructions unchanged.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

CONDITIONS = ["correct", "swap", "mismatched", "no_image"]

def mismatched_mapping(pairs: List[Dict[str, Any]], seed: int = 6304) -> Dict[Tuple[int, str], str]:
    """
    Fixed before generation: each pair takes the same-side image of the next pair in a seeded cycle,
    which excludes both images of the original pair. The mapping is reused across comparisons.
    """
    if len(pairs) < 2:
        raise ValueError("Mismatched control needs at least two pairs")
    cycle = np.random.default_rng(seed).permutation(len(pairs)).tolist()
    mapping = {}
    for position, pair_index in enumerate(cycle):
        donor = pairs[cycle[(position + 1) % len(cycle)]]
        for side in ("a", "b"):
            mapping[(pair_index, side)] = donor[side]["image"]
    return mapping

def conditioned_image(pair: Dict[str, Any], pair_index: int, side: str, condition: str, mapping: Dict[Tuple[int, str], str]) -> Optional[str]:
    """Image key used for one question side under one condition, or None for no image."""
    other = "b" if side == "a" else "a"
    if condition == "correct":
        return pair[side]["image"]
    if condition == "swap":
        return pair[other]["image"]
    if condition == "mismatched":
        return mapping[(pair_index, side)]
    if condition == "no_image":
        return None
    raise ValueError(f"Unknown control condition: {condition}")
