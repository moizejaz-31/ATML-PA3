"""
Adaptive sampling controllers:
1. Agreement-based stopping (lead vote share >= tau)
2. Quality-aware stopping (agreement + SC/RWC consensus + RWC margin >= gamma)
"""
from typing import List, Tuple, Dict
from collections import defaultdict
from .aggregation import select_self_consistency, rank_weighted_consensus
from .normalization import NO_ANSWER

def check_agreement_stopping(answers: List[str], tau: float) -> Tuple[bool, str]:
    """
    Check if a unique, non-NO_ANSWER leading answer has share >= tau.
    """
    n = len(answers)
    counts = defaultdict(int)
    for a in answers:
        counts[a] += 1
        
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    lead_ans, lead_count = sorted_items[0]
    
    if lead_ans == NO_ANSWER:
        return False, lead_ans
        
    # Check uniqueness
    if len(sorted_items) > 1 and sorted_items[1][1] == lead_count:
        return False, lead_ans
        
    if lead_count / n >= tau:
        return True, lead_ans
    return False, lead_ans

def compute_rwc_margin(answers: List[str], scores: List[float]) -> Tuple[float, str, str]:
    """
    Compute RWC vote margin: (W_top - W_second) / W_total.
    Returns (margin, sc_winner, rwc_winner).
    """
    N = len(scores)
    sc_winner = select_self_consistency(answers)
    rwc_winner = rank_weighted_consensus(answers, scores)
    
    if N == 1:
        return 1.0, sc_winner, rwc_winner

    # RWC weights
    sorted_indices = sorted(range(N), key=lambda i: scores[i], reverse=True)
    ranks = [0.0] * N
    i = 0
    while i < N:
        j = i
        while j < N and scores[sorted_indices[j]] == scores[sorted_indices[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for idx in range(i, j):
            ranks[sorted_indices[idx]] = avg_rank
        i = j

    vote_weights = defaultdict(float)
    for a, r in zip(answers, ranks):
        w = (N - r) / (N - 1.0)
        vote_weights[a] += w

    total_w = sum(vote_weights.values())
    sorted_w = sorted(vote_weights.values(), reverse=True)
    w_top = sorted_w[0]
    w_second = sorted_w[1] if len(sorted_w) > 1 else 0.0
    
    margin = (w_top - w_second) / max(total_w, 1e-8)
    return margin, sc_winner, rwc_winner

def check_quality_aware_stopping(answers: List[str], scores: List[float], tau: float, gamma: float) -> Tuple[bool, str]:
    """
    Quality-aware stopping:
    1. Agreement condition met (share >= tau)
    2. SC and RWC winners agree
    3. RWC margin >= gamma
    """
    agreed, lead_ans = check_agreement_stopping(answers, tau)
    if not agreed:
        return False, lead_ans
        
    margin, sc_winner, rwc_winner = compute_rwc_margin(answers, scores)
    if sc_winner == rwc_winner and margin >= gamma:
        return True, sc_winner
    return False, lead_ans
