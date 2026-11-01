"""
Adaptive sampling controllers:
1. Agreement-based stopping (lead vote share >= tau)
2. Quality-aware stopping (agreement + SC/RWC consensus + RWC margin >= gamma)
Stopping is checked after n in {4, 8, 16}, otherwise at 32. Both controllers return the current
self-consistency answer. Scores, VisConf calibration, and ranks use revealed candidates only.
"""
from typing import Dict, List, Optional, Sequence, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd

from .aggregation import average_ranks, rank_weighted_consensus, select_self_consistency
from .data import Question
from .metrics import compute_visconf
from .normalization import NO_ANSWER

CHECK_POINTS = [4, 8, 16]
TAUS = [0.5, 0.625, 0.75, 0.875, 1.0]
GAMMAS = [0.0, 0.1, 0.25, 0.5]

def check_agreement_stopping(answers: List, tau: float, no_answer=NO_ANSWER) -> Tuple[bool, object]:
    """
    Check if a unique, non-NO_ANSWER leading answer has share >= tau.
    """
    n = len(answers)
    counts = defaultdict(int)
    for a in answers:
        counts[a] += 1

    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    lead_ans, lead_count = sorted_items[0]

    if lead_ans == no_answer:
        return False, lead_ans

    # Check uniqueness
    if len(sorted_items) > 1 and sorted_items[1][1] == lead_count:
        return False, lead_ans

    return lead_count / n >= tau, lead_ans

def compute_rwc_margin(answers: List, scores: Sequence[float]) -> Tuple[float, object, object]:
    """
    Compute RWC vote margin: (W_top - W_second) / W_total, with zero for an absent runner-up.
    Returns (margin, sc_winner, rwc_winner).
    """
    n = len(scores)
    sc_winner = select_self_consistency(answers)
    rwc_winner = rank_weighted_consensus(answers, scores)
    if n == 1:
        return 1.0, sc_winner, rwc_winner

    vote_weights = defaultdict(float)
    for a, r in zip(answers, average_ranks(scores)):
        vote_weights[a] += (n - r) / (n - 1.0)

    total = sum(vote_weights.values())
    ordered = sorted(vote_weights.values(), reverse=True)
    runner_up = ordered[1] if len(ordered) > 1 else 0.0
    return (ordered[0] - runner_up) / max(total, 1e-8), sc_winner, rwc_winner

def check_quality_aware_stopping(answers: List, scores: Sequence[float], tau: float, gamma: float, no_answer=NO_ANSWER) -> Tuple[bool, object]:
    """Agreement condition, plus SC and RWC winners agree, plus RWC margin >= gamma."""
    agreed, lead_ans = check_agreement_stopping(answers, tau, no_answer)
    if not agreed:
        return False, lead_ans
    margin, sc_winner, rwc_winner = compute_rwc_margin(answers, scores)
    return sc_winner == rwc_winner and margin >= gamma, sc_winner

def revealed_quality_scores(q: Question, prefix: np.ndarray) -> List[float]:
    """Self-certainty for MATH, VisConf for MathVista, both from revealed rollouts only."""
    if q.V is None:
        return q.K[prefix].tolist()
    return compute_visconf([q.kl_tokens[i] for i in prefix], q.V[prefix])

def simulate_order(q: Question, order: np.ndarray, tau: Optional[float], gamma: Optional[float]) -> Dict[str, float]:
    """One reveal order. tau=None always consumes 32; gamma=None is agreement-only."""
    no_answer_group = q.group_names.index(NO_ANSWER) if NO_ANSWER in q.group_names else -1
    stop_n = 32
    if tau is not None:
        for n in CHECK_POINTS:
            prefix = order[:n]
            answers = q.groups[prefix].tolist()
            if gamma is None:
                stop, _ = check_agreement_stopping(answers, tau, no_answer_group)
            else:
                stop, _ = check_quality_aware_stopping(answers, revealed_quality_scores(q, prefix), tau, gamma, no_answer_group)
            if stop:
                stop_n = n
                break
    prefix = order[:stop_n]
    answer = select_self_consistency(q.groups[prefix].tolist())
    return {"correct": float(q.group_correct[answer]), "cost": float(q.tokens[prefix].sum()), "rollouts": stop_n}

def evaluate_policy(questions: Sequence[Question], orders: Dict[str, np.ndarray], tau: Optional[float], gamma: Optional[float]) -> pd.DataFrame:
    """Per-question and per-order outcomes for one stopping policy."""
    rows = []
    for q in questions:
        for order_id, order in enumerate(orders[q.sample_id]):
            rows.append({"sample_id": q.sample_id, "order": order_id, **simulate_order(q, order, tau, gamma)})
    return pd.DataFrame(rows)

def _mean_over_orders(outcomes: pd.DataFrame) -> Dict[str, float]:
    per_question = outcomes.groupby("sample_id")[["correct", "cost", "rollouts"]].mean()
    return {"accuracy": float(per_question["correct"].mean()), "mean_cost": float(per_question["cost"].mean()), "mean_rollouts": float(per_question["rollouts"].mean())}

def select_policies(dev_questions: Sequence[Question], orders: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Optional[float]]]]:
    """
    Grid search on development questions. For each controller family, pick the lowest mean token cost
    with accuracy no more than 1 pp below fixed-32 self-consistency. Ties: higher accuracy, smaller tau and gamma.
    The always-32 policy is the fallback.
    """
    fixed = _mean_over_orders(evaluate_policy(dev_questions, orders, None, None))
    rows = [{"family": "fixed_32", "tau": None, "gamma": None, **fixed}]
    for tau in TAUS:
        rows.append({"family": "agreement", "tau": tau, "gamma": None, **_mean_over_orders(evaluate_policy(dev_questions, orders, tau, None))})
        for gamma in GAMMAS:
            rows.append({"family": "quality_aware", "tau": tau, "gamma": gamma, **_mean_over_orders(evaluate_policy(dev_questions, orders, tau, gamma))})
    grid = pd.DataFrame(rows)
    grid["admissible"] = grid["accuracy"] >= fixed["accuracy"] - 0.01

    selected = {}
    for family in ["agreement", "quality_aware"]:
        options = [(fixed["mean_cost"], -fixed["accuracy"], float("inf"), float("inf"), None, None)]
        for _, row in grid[(grid["family"] == family) & grid["admissible"]].iterrows():
            gamma = row["gamma"] if family == "quality_aware" else None
            options.append((row["mean_cost"], -row["accuracy"], row["tau"], gamma if gamma is not None else 0.0, row["tau"], gamma))
        best = min(options, key=lambda o: o[:4])
        selected[family] = {"tau": best[4], "gamma": best[5]}
    return grid, selected
