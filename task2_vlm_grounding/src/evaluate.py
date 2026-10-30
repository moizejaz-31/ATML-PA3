"""
Evaluation metrics and scoring protocols for Task 2.
"""
from typing import List

def compute_vqa_consensus_score(prediction: str, ground_truth_answers: List[str]) -> float:
    """
    Local VQA soft consensus accuracy across 10 human annotations:
    s(a_hat) = (1/10) * sum_{i=1}^{10} min(1, (1/3) * sum_{j!=i} 1[a_hat == a_j])
    """
    pred = prediction.strip().lower()
    gts = [a.strip().lower() for a in ground_truth_answers]
    n = len(gts)
    if n == 0:
        return 0.0

    score = 0.0
    for i in range(n):
        matches = sum(1 for j in range(n) if j != i and pred == gts[j])
        score += min(1.0, matches / 3.0)
    return score / n

def compute_image_swap_penalty(nll_partner: float, nll_correct: float) -> float:
    """
    Delta NLL = NLL(y | q, I_partner) - NLL(y | q, I_correct)
    Positive values indicate that correct visual context makes the reference answer more likely.
    """
    return nll_partner - nll_correct
