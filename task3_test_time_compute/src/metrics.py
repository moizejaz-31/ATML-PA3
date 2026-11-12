"""
Quality scoring and confidence metrics for Task 3:
1. Mean log-probability L_i
2. Self-certainty K_i
3. Visual engagement V_i
4. Candidate-calibrated VisConf Z_i = V_i * C_i, and its tokenwise ablation
"""
from typing import List, Sequence, Tuple
import numpy as np

def compute_mean_logp(logp_list: Sequence[float]) -> float:
    """L_i = (1 / T_i) sum_t log p_{i,t}(y_{i,t}). An empty rollout ranks last."""
    if len(logp_list) == 0:
        return float("-inf")
    return float(np.mean(logp_list))

def compute_self_certainty(kl_list: Sequence[float]) -> float:
    """K_i = (1 / T_i) sum_t D_KL(U || p_{i,t}). An empty rollout ranks last."""
    if len(kl_list) == 0:
        return float("-inf")
    return float(np.mean(kl_list))

def compute_token_engagement(image_att, prompt_att, gen_att) -> np.ndarray:
    """v_{i,t} = A^image / (A^image + A^prompt + A^generated), zero where the total is zero."""
    image_att, prompt_att, gen_att = [np.asarray(a, dtype=np.float64) for a in (image_att, prompt_att, gen_att)]
    total = image_att + prompt_att + gen_att
    return np.divide(image_att, total, out=np.zeros_like(total), where=total > 0)

def compute_visual_engagement(image_att, prompt_att, gen_att) -> float:
    """V_i = (1 / T_i) sum_t v_{i,t}"""
    if len(image_att) == 0:
        return 0.0
    return float(np.mean(compute_token_engagement(image_att, prompt_att, gen_att)))

def calibrate_tokens(candidate_kl_tokens: List[np.ndarray]) -> List[np.ndarray]:
    """
    c_{i,t} = clip_{[0,1]}((k_{i,t} - Q_{.05}(G)) / max(Q_{.95}(G) - Q_{.05}(G), 10^-8))
    G pools the current candidates only, so unseen rollouts never influence the calibration.
    """
    pooled = np.concatenate(candidate_kl_tokens) if candidate_kl_tokens else np.empty(0)
    if pooled.size == 0:
        return [np.empty(0) for _ in candidate_kl_tokens]
    q05, q95 = np.percentile(pooled, [5, 95], method="linear")
    scale = max(q95 - q05, 1e-8)
    return [np.clip((np.asarray(k) - q05) / scale, 0.0, 1.0) for k in candidate_kl_tokens]

def compute_visconf(candidate_kl_tokens: List[np.ndarray], candidate_vis_engagements: Sequence[float]) -> List[float]:
    """Z_i = V_i * C_i, where C_i averages calibrated confidence within the rollout."""
    calibrated = calibrate_tokens(candidate_kl_tokens)
    return [float(V * c.mean()) if c.size else 0.0 for c, V in zip(calibrated, candidate_vis_engagements)]

def compute_visconf_tokenwise(candidate_kl_tokens: List[np.ndarray], candidate_v_tokens: List[np.ndarray]) -> List[float]:
    """Z_i^token = (1 / T_i) sum_t c_{i,t} v_{i,t}"""
    calibrated = calibrate_tokens(candidate_kl_tokens)
    return [float(np.mean(c * v)) if c.size else 0.0 for c, v in zip(calibrated, candidate_v_tokens)]
