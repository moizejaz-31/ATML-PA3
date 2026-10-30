"""
Evaluation metrics, pair diagnostics, and 4,000 pair bootstrap confidence intervals for Task 2.
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
from .data import normalize_vqa_answer

def compute_vqa_soft_accuracy(prediction: str, ground_truth_answers: List[str]) -> float:
    """
    Standard VQAv2 consensus metric:
    min(1.0, count(matching annotators) / 3)
    """
    norm_pred = normalize_vqa_answer(prediction)
    norm_gts = [normalize_vqa_answer(a) for a in ground_truth_answers]
    matches = sum(1 for a in norm_gts if a == norm_pred)
    return min(1.0, matches / 3.0)

# Alias for compatibility
compute_vqa_consensus_score = compute_vqa_soft_accuracy

def compute_image_swap_penalty(acc_correct: float, acc_swap: float) -> float:
    """Computes evidence dependence penalty: Acc(Correct) - Acc(Swap)."""
    return max(0.0, acc_correct - acc_swap)

def evaluate_pair_suite(
    predictions_a: List[str],
    predictions_b: List[str],
    targets_a: List[str],
    targets_b: List[str],
    all_answers_a: List[List[str]],
    all_answers_b: List[List[str]],
    is_vqa: bool = False,
) -> Dict[str, float]:
    """
    Calculates suite performance and pair diagnostics:
      - Individual question accuracy
      - Pair accuracy (both partner instances correct)
      - Image sensitivity (predictions differ between partner images)
      - Swapped following (prediction matches partner's gold target)
    """
    n_pairs = len(predictions_a)
    if n_pairs == 0:
        return {}

    scores_indiv = []
    both_correct = []
    sensitivities = []
    followings = []

    for i in range(n_pairs):
        pred_a, pred_b = predictions_a[i], predictions_b[i]
        tgt_a, tgt_b = targets_a[i], targets_b[i]
        gts_a, gts_b = all_answers_a[i], all_answers_b[i]

        norm_pred_a = normalize_vqa_answer(pred_a)
        norm_pred_b = normalize_vqa_answer(pred_b)
        norm_tgt_a = normalize_vqa_answer(tgt_a)
        norm_tgt_b = normalize_vqa_answer(tgt_b)

        if is_vqa:
            sc_a = compute_vqa_soft_accuracy(pred_a, gts_a)
            sc_b = compute_vqa_soft_accuracy(pred_b, gts_b)
        else:
            sc_a = 1.0 if norm_pred_a == norm_tgt_a else 0.0
            sc_b = 1.0 if norm_pred_b == norm_tgt_b else 0.0

        scores_indiv.extend([sc_a, sc_b])
        both_correct.append(1.0 if (sc_a >= 0.99 and sc_b >= 0.99) else 0.0)

        # Image sensitivity: prediction changes when image changes
        sensitivities.append(1.0 if norm_pred_a != norm_pred_b else 0.0)

        # Swapped following: under swap, does prediction equal partner target?
        f_a = 1.0 if norm_pred_a == norm_tgt_b else 0.0
        f_b = 1.0 if norm_pred_b == norm_tgt_a else 0.0
        followings.append((f_a + f_b) / 2.0)

    return {
        "individual_accuracy": float(np.mean(scores_indiv) * 100.0),
        "pair_accuracy": float(np.mean(both_correct) * 100.0),
        "image_sensitivity": float(np.mean(sensitivities) * 100.0),
        "swapped_following": float(np.mean(followings) * 100.0),
        "n_pairs": n_pairs,
    }

def bootstrap_pair_metrics(
    predictions_a: List[str],
    predictions_b: List[str],
    targets_a: List[str],
    targets_b: List[str],
    all_answers_a: List[List[str]],
    all_answers_b: List[List[str]],
    is_vqa: bool = False,
    n_resamples: int = 4000,
    seed: int = 6304,
) -> Dict[str, Tuple[float, float, float]]:
    """
    4,000 bootstrap resamples on complete pairs, preserving shared question structure.
    Returns (mean, ci_lower, ci_upper) at 95% confidence.
    """
    rng = np.random.default_rng(seed)
    n_pairs = len(predictions_a)
    boot_accs = []
    boot_pairs = []

    for _ in range(n_resamples):
        indices = rng.integers(0, n_pairs, size=n_pairs)
        sub_pred_a = [predictions_a[i] for i in indices]
        sub_pred_b = [predictions_b[i] for i in indices]
        sub_tgt_a = [targets_a[i] for i in indices]
        sub_tgt_b = [targets_b[i] for i in indices]
        sub_gts_a = [all_answers_a[i] for i in indices]
        sub_gts_b = [all_answers_b[i] for i in indices]

        metrics = evaluate_pair_suite(
            sub_pred_a, sub_pred_b, sub_tgt_a, sub_tgt_b,
            sub_gts_a, sub_gts_b, is_vqa=is_vqa
        )
        boot_accs.append(metrics["individual_accuracy"])
        boot_pairs.append(metrics["pair_accuracy"])

    return {
        "individual_accuracy": (
            float(np.mean(boot_accs)),
            float(np.percentile(boot_accs, 2.5)),
            float(np.percentile(boot_accs, 97.5)),
        ),
        "pair_accuracy": (
            float(np.mean(boot_pairs)),
            float(np.percentile(boot_pairs, 2.5)),
            float(np.percentile(boot_pairs, 97.5)),
        )
    }
