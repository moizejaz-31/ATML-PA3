"""
Offline evaluation across candidate budgets N in {1, 2, 4, 8, 16, 32}:
ten seeded permutations per question, nested prefixes, and every selection method.
Outcomes are averaged over orders within a question, then across questions.
"""
from typing import Callable, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

from .aggregation import rank_weighted_consensus, select_highest_score, select_self_consistency
from .data import Question
from .metrics import compute_visconf, compute_visconf_tokenwise

BUDGETS = [1, 2, 4, 8, 16, 32]
NUM_ORDERS = 10
SEED = 6304
LLM_METHODS = ["first_ans", "self_consistency", "mean_logp", "self_certainty", "self_certainty_rwc"]
VLM_METHODS = LLM_METHODS + ["vision_only", "vision_only_rwc", "visconf", "visconf_rwc", "visconf_token", "visconf_token_rwc"]

def make_orders(questions: Sequence[Question], num_orders: int = NUM_ORDERS, seed: int = SEED) -> Dict[str, np.ndarray]:
    """Ten permutations of the 32 rollout IDs per question, reused across methods and stopping policies."""
    rng = np.random.default_rng(seed)
    return {q.sample_id: np.stack([rng.permutation(32) for _ in range(num_orders)]) for q in questions}

def select_for_prefix(q: Question, prefix: np.ndarray, methods: Sequence[str]) -> Dict[str, int]:
    """Selected answer group per method. Every score is computed from this prefix only."""
    answers = q.groups[prefix].tolist()
    scores: Dict[str, List[float]] = {"mean_logp": q.L[prefix].tolist(), "self_certainty": q.K[prefix].tolist()}
    if q.V is not None and len(methods) > len(LLM_METHODS):
        kl = [q.kl_tokens[i] for i in prefix]
        scores["vision_only"] = q.V[prefix].tolist()
        scores["visconf"] = compute_visconf(kl, q.V[prefix])
        scores["visconf_token"] = compute_visconf_tokenwise(kl, [q.v_tokens[i] for i in prefix])

    selected = {}
    for method in methods:
        if method == "first_ans":
            selected[method] = answers[0]
        elif method == "self_consistency":
            selected[method] = select_self_consistency(answers)
        elif method.endswith("_rwc"):
            selected[method] = rank_weighted_consensus(answers, scores[method[:-4]])
        else:
            selected[method] = select_highest_score(answers, scores[method])
    return selected

def run_offline_budget_evaluation(
    questions: Sequence[Question],
    orders: Dict[str, np.ndarray],
    budgets: Sequence[int] = BUDGETS,
    methods: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Per-question results, averaged over orders. Columns: sample_id, N, method, accuracy, token_cost."""
    rows = []
    for q in questions:
        method_list = methods or (VLM_METHODS if q.V is not None else LLM_METHODS)
        totals: Dict[tuple, List[float]] = {}
        for order in orders[q.sample_id]:
            for n in budgets:
                prefix = order[:n]
                prefix_cost = float(q.tokens[prefix].sum())
                for method, group in select_for_prefix(q, prefix, method_list).items():
                    cost = float(q.tokens[prefix[0]]) if method == "first_ans" else prefix_cost
                    totals.setdefault((n, method), []).append((q.group_correct[group], cost))
        for (n, method), values in totals.items():
            values = np.array(values, dtype=float)
            rows.append({"sample_id": q.sample_id, "N": n, "method": method, "accuracy": values[:, 0].mean(), "token_cost": values[:, 1].mean()})
    return pd.DataFrame(rows)

def run_coverage_analysis(questions: Sequence[Question], orders: Dict[str, np.ndarray], budgets: Sequence[int] = BUDGETS) -> pd.DataFrame:
    """FirstAns accuracy, oracle coverage, distinct answer groups, and token cost per question and budget."""
    rows = []
    for q in questions:
        for n in budgets:
            prefixes = orders[q.sample_id][:, :n]
            rows.append({
                "sample_id": q.sample_id, "N": n,
                "first_ans": q.correct[prefixes[:, 0]].mean(),
                "oracle_coverage": np.mean([q.correct[p].any() for p in prefixes]),
                "distinct_groups": np.mean([len(set(q.groups[p])) for p in prefixes]),
                "token_cost": np.mean([q.tokens[p].sum() for p in prefixes]),
            })
    return pd.DataFrame(rows)

def decompose_failures(results: pd.DataFrame, coverage: pd.DataFrame, method: str = "self_consistency") -> pd.DataFrame:
    """Split selection errors into 'no correct candidate generated' and 'correct candidate available but missed'."""
    merged = results[results["method"] == method].merge(coverage, on=["sample_id", "N"])
    merged["correct"] = merged["accuracy"]
    merged["no_correct_candidate"] = 1.0 - merged["oracle_coverage"]
    merged["missed_available"] = merged["oracle_coverage"] - merged["accuracy"]
    return merged.groupby("N")[["correct", "no_correct_candidate", "missed_available"]].mean().reset_index()

def paired_bootstrap(a: np.ndarray, b: np.ndarray, resamples: int = 1000, seed: int = SEED) -> Dict[str, float]:
    """Question-level paired bootstrap of mean(a - b), in percentage points, with a 95% interval."""
    diff = (np.asarray(a) - np.asarray(b)) * 100.0
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(resamples, len(diff)))].mean(axis=1)
    return {"difference_pp": float(diff.mean()), "ci_low": float(np.percentile(means, 2.5)), "ci_high": float(np.percentile(means, 97.5))}

def compare_methods(results: pd.DataFrame, comparisons: Sequence[tuple], n: int = 32) -> pd.DataFrame:
    """Paired bootstrap for (method_a, method_b) accuracy differences at budget n."""
    table = results[results["N"] == n].pivot(index="sample_id", columns="method", values="accuracy")
    rows = []
    for a, b in comparisons:
        if a in table and b in table:
            rows.append({"method_a": a, "method_b": b, "N": n, **paired_bootstrap(table[a].to_numpy(), table[b].to_numpy())})
    return pd.DataFrame(rows)
