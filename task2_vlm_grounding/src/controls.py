"""
Image control conditions for Task 2 visual grounding evaluation:
1. Correct: Original image.
2. Complementary swap: Partner image with different answer.
3. Mismatched: Unrelated image from another pair in the suite.
4. No image: Visual tokens omitted, masks/positions rebuilt.
"""
from typing import Dict, Any, List

def construct_mismatched_mapping(pair_ids: List[str], seed: int = 6304) -> Dict[str, str]:
    """
    Fixed derangement/permutation excluding both images of the original pair.
    """
    pass
