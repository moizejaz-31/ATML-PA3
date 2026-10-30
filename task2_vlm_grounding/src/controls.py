"""
Four Image Control Conditions for Task 2 Visual Grounding Evaluation:
1. Correct: The original image intended for the question.
2. Complementary Swap: The partner image from the same complementary pair (shared question, different target).
3. Mismatched: A fixed derangement mapping to an unrelated image from another pair.
4. No Image: Visual tokens omitted (tests language model prior alone).
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

def generate_derangement_mapping(image_ids: List[str], seed: int = 6304) -> Dict[str, str]:
    """
    Constructs a fixed derangement (permutation where pi(i) != i)
    for the mismatched image control condition.
    """
    rng = np.random.default_rng(seed)
    n = len(image_ids)
    if n <= 1:
        return {img: img for img in image_ids}

    # Derangement generation via rejection sampling
    indices = np.arange(n)
    while True:
        perm = rng.permutation(n)
        if np.all(perm != indices):
            break

    return {image_ids[i]: image_ids[perm[i]] for i in range(n)}

def prepare_control_inputs(
    pair: Dict[str, Any],
    control_condition: str,
    mismatched_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns (item_a, item_b) for evaluation under the requested control condition:
    'correct', 'complementary_swap', 'mismatched', or 'no_image'.
    """
    q = pair["question"]
    target_a = pair["image_a"]["target"]
    target_b = pair["image_b"]["target"]
    img_a = pair["image_a"]["image_id"]
    img_b = pair["image_b"]["image_id"]

    prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"

    if control_condition == "correct":
        item_a = {"image_id": img_a, "prompt": prompt, "target": target_a, "partner_target": target_b, "all_answers": pair["image_a"]["all_answers"]}
        item_b = {"image_id": img_b, "prompt": prompt, "target": target_b, "partner_target": target_a, "all_answers": pair["image_b"]["all_answers"]}
    elif control_condition == "complementary_swap":
        # Swapped images: question A paired with image B, question B paired with image A
        item_a = {"image_id": img_b, "prompt": prompt, "target": target_a, "partner_target": target_b, "all_answers": pair["image_a"]["all_answers"]}
        item_b = {"image_id": img_a, "prompt": prompt, "target": target_b, "partner_target": target_a, "all_answers": pair["image_b"]["all_answers"]}
    elif control_condition == "mismatched":
        m_a = mismatched_mapping.get(img_a, img_b) if mismatched_mapping else img_b
        m_b = mismatched_mapping.get(img_b, img_a) if mismatched_mapping else img_a
        item_a = {"image_id": m_a, "prompt": prompt, "target": target_a, "partner_target": target_b, "all_answers": pair["image_a"]["all_answers"]}
        item_b = {"image_id": m_b, "prompt": prompt, "target": target_b, "partner_target": target_a, "all_answers": pair["image_b"]["all_answers"]}
    elif control_condition == "no_image":
        item_a = {"image_id": None, "prompt": prompt, "target": target_a, "partner_target": target_b, "all_answers": pair["image_a"]["all_answers"]}
        item_b = {"image_id": None, "prompt": prompt, "target": target_b, "partner_target": target_a, "all_answers": pair["image_b"]["all_answers"]}
    else:
        raise ValueError(f"Unknown control condition: {control_condition}")

    return item_a, item_b
