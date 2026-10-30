"""
Quality scoring and confidence metrics for Task 3:
1. Mean log-probability L_i
2. Self-certainty K_i
3. Visual engagement V_i
4. Candidate-calibrated VisConf Z_i = V_i * C_i
"""
from typing import List
import numpy as np

def compute_mean_logp(logp_list: List[float]) -> float:
    """L_i = (1 / T_i) sum_t log p_{i,t}(y_{i,t})"""
    if not logp_list:
        return 0.0
    return float(np.mean(logp_list))

def compute_self_certainty(kl_list: List[float]) -> float:
    """K_i = (1 / T_i) sum_t D_KL(U || p_{i,t})"""
    if not kl_list:
        return 0.0
    return float(np.mean(kl_list))

def compute_visual_engagement(image_att: List[float], prompt_att: List[float], gen_att: List[float]) -> float:
    """
    v_{i,t} = A^image / (A^image + A^prompt + A^generated)
    V_i = (1 / T_i) sum_t v_{i,t}
    """
    if not image_att:
        return 0.0
    v_t = []
    for a_img, a_prm, a_gen in zip(image_att, prompt_att, gen_att):
        denom = a_img + a_prm + a_gen
        v_t.append(a_img / denom if denom > 0 else 0.0)
    return float(np.mean(v_t))

def compute_visconf(
    candidate_kl_tokens: List[List[float]],
    candidate_vis_engagements: List[float],
) -> List[float]:
    """
    Candidate-calibrated VisConf Z_i = V_i * C_i
    c_{i,t} = clip_{[0,1]}((k_{i,t} - Q_{.05}(G)) / max(Q_{.95}(G) - Q_{.05}(G), 10^-8))
    """
    all_kl = [k for tokens in candidate_kl_tokens for k in tokens]
    if not all_kl:
        return [0.0] * len(candidate_kl_tokens)

    q05 = np.percentile(all_kl, 5, method="linear")
    q95 = np.percentile(all_kl, 95, method="linear")
    denom = max(q95 - q05, 1e-8)

    Z_scores = []
    for kl_tokens, V_i in zip(candidate_kl_tokens, candidate_vis_engagements):
        c_tokens = [np.clip((k - q05) / denom, 0.0, 1.0) for k in kl_tokens]
        C_i = float(np.mean(c_tokens)) if c_tokens else 0.0
        Z_scores.append(V_i * C_i)

    return Z_scores
