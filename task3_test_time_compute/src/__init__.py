"""Task 3 modules: data loading, splits, metrics, aggregation (RWC), live refinement, adaptive stopping."""
from .splits import create_splits
from .normalization import normalize_math_answer, normalize_mathvista_answer
from .metrics import compute_mean_logp, compute_self_certainty, compute_visual_engagement, compute_visconf
from .aggregation import select_highest_score, rank_weighted_consensus

__all__ = [
    "create_splits",
    "normalize_math_answer",
    "normalize_mathvista_answer",
    "compute_mean_logp",
    "compute_self_certainty",
    "compute_visual_engagement",
    "compute_visconf",
    "select_highest_score",
    "rank_weighted_consensus",
]
