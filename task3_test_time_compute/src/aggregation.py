"""
Answer aggregation and candidate selection rules for Task 3:
1. FirstAns (first candidate in order)
2. Majority Self-Consistency (1 vote per rollout)
3. Highest-Score Selection (argmax_i Q_i)
4. Rank-Weighted Consensus (RWC):
   w_i = (N - r_i) / (N - 1), W(a) = sum_{i: a_i = a} w_i
Score and vote ties resolve to the earliest supporting rollout in the current order.
"""
from typing import List, Sequence
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata

def select_first_ans(answers: List):
    """Return earliest candidate answer."""
    return answers[0]

def select_self_consistency(answers: List):
    """Majority voting with tie broken by earliest appearance."""
    counts = defaultdict(int)
    earliest_idx = {}
    for i, a in enumerate(answers):
        counts[a] += 1
        if a not in earliest_idx:
            earliest_idx[a] = i

    return max(counts.keys(), key=lambda a: (counts[a], -earliest_idx[a]))

def select_highest_score(answers: List, scores: Sequence[float]):
    """Return answer of candidate with highest score (earliest on tie)."""
    return answers[int(np.argmax(scores))]

def average_ranks(scores: Sequence[float]) -> np.ndarray:
    """Descending average ranks: rank 1 is the best score, and ties share their mean rank."""
    return rankdata(-np.asarray(scores, dtype=float), method="average")

def rank_weighted_consensus(answers: List, scores: Sequence[float]):
    """
    RWC selection rule:
    w_i = (N - r_i) / (N - 1) where r_i is average rank in descending order (rank 1 is best).
    W(a) = sum_{i: a_i = a} w_i
    """
    n = len(scores)
    if n == 1:
        return answers[0]

    vote_weights = defaultdict(float)
    earliest_idx = {}
    for idx, (a, r) in enumerate(zip(answers, average_ranks(scores))):
        vote_weights[a] += (n - r) / (n - 1.0)
        if a not in earliest_idx:
            earliest_idx[a] = idx

    # Rounding guard: floating-point sums of equal votes must still tie
    return max(vote_weights.keys(), key=lambda a: (round(vote_weights[a], 9), -earliest_idx[a]))
