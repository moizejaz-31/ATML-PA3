import numpy as np
from typing import Tuple, List, Callable, Optional

def bootstrap_ci(
    data: np.ndarray,
    stat_fn: Callable = np.mean,
    num_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 6304
) -> Tuple[float, float, float]:
    """
    Compute non-parametric bootstrap confidence interval.
    
    Returns:
        (observed_statistic, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    observed = float(stat_fn(data))
    indices = rng.integers(0, n, size=(num_bootstraps, n))
    resampled_stats = np.array([stat_fn(data[idx]) for idx in indices])
    
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(resampled_stats, alpha * 100))
    upper = float(np.percentile(resampled_stats, (1.0 - alpha) * 100))
    return observed, lower, upper

def paired_bootstrap_difference(
    group_a: np.ndarray,
    group_b: np.ndarray,
    num_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 6304
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for paired differences (group_a - group_b).
    """
    assert len(group_a) == len(group_b), "Groups must have equal length for paired bootstrap."
    diffs = group_a - group_b
    return bootstrap_ci(diffs, np.mean, num_bootstraps=num_bootstraps, ci=ci, seed=seed)
