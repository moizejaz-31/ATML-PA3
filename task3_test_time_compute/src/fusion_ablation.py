"""
Tokenwise fusion ablation for Task 3.
Z_i^token = (1 / T_i) sum_t c_{i,t} * v_{i,t}
Z_i^token - Z_i = Cov_t(c_{i,t}, v_{i,t})
"""
from typing import List, Tuple
import numpy as np

def compute_tokenwise_fusion(c_tokens: List[float], v_tokens: List[float]) -> Tuple[float, float]:
    """
    Compute tokenwise fusion Z_token and covariance Cov_t(c, v).
    """
    c = np.array(c_tokens)
    v = np.array(v_tokens)
    Z_token = float(np.mean(c * v))
    Z_traj = float(np.mean(c) * np.mean(v))
    cov = Z_token - Z_traj
    return Z_token, cov
