"""
Answer aggregation and candidate selection rules for Task 3:
1. FirstAns (first candidate in order)
2. Majority Self-Consistency (1 vote per rollout)
3. Highest-Score Selection (argmax_i Q_i)
4. Rank-Weighted Consensus (RWC):
   w_i = (N - r_i) / (N - 1), W(a) = sum_{i: a_i = a} w_i
"""
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np

def select_first_ans(answers: List[str]) -> str:
    """Return earliest candidate answer."""
    return answers[0]

def select_self_consistency(answers: List[str]) -> str:
    """Majority voting with tie broken by earliest appearance."""
    counts = defaultdict(int)
    earliest_idx = {}
    for i, a in enumerate(answers):
        counts[a] += 1
        if a not in earliest_idx:
            earliest_idx[a] = i

    best_ans = max(counts.keys(), key=lambda a: (counts[a], -earliest_idx[a]))
    return best_ans

def select_highest_score(answers: List[str], scores: List[float]) -> str:
    """Return answer of candidate with highest score (earliest on tie)."""
    best_idx = 0
    best_score = scores[0]
    for i, s in enumerate(scores):
        if s > best_score:
            best_score = s
            best_idx = i
    return answers[best_idx]

def rank_weighted_consensus(answers: List[str], scores: List[float]) -> str:
    """
    RWC selection rule:
    w_i = (N - r_i) / (N - 1) where r_i is average rank in descending order (rank 1 is best).
    W(a) = sum_{i: a_i = a} w_i
    """
    N = len(scores)
    if N == 1:
        return answers[0]

    # Compute average rank for tied scores in descending order
    # (highest score gets rank 1)
    sorted_indices = sorted(range(N), key=lambda i: scores[i], reverse=True)
    ranks = [0.0] * N
    
    i = 0
    while i < N:
        j = i
        while j < N and scores[sorted_indices[j]] == scores[sorted_indices[i]]:
            j += 1
        # Indices sorted_indices[i...j-1] are tied
        avg_rank = (i + 1 + j) / 2.0
        for idx in range(i, j):
            ranks[sorted_indices[idx]] = avg_rank
        i = j

    vote_weights = defaultdict(float)
    earliest_idx = {}
    for idx, (a, r) in enumerate(zip(answers, ranks)):
        w = (N - r) / (N - 1.0)
        vote_weights[a] += w
        if a not in earliest_idx:
            earliest_idx[a] = idx

    best_ans = max(vote_weights.keys(), key=lambda a: (vote_weights[a], -earliest_idx[a]))
    return best_ans
