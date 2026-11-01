"""
Signal analysis and tokenwise fusion ablation for Task 3.
Z_i^token = (1 / T_i) sum_t c_{i,t} * v_{i,t}
Z_i^token - Z_i = Cov_t(c_{i,t}, v_{i,t})
"""
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data import Question
from .metrics import calibrate_tokens

def compute_tokenwise_fusion(c_tokens: Sequence[float], v_tokens: Sequence[float]) -> Tuple[float, float]:
    """
    Compute tokenwise fusion Z_token and covariance Cov_t(c, v).
    """
    c = np.asarray(c_tokens, dtype=float)
    v = np.asarray(v_tokens, dtype=float)
    z_token = float(np.mean(c * v))
    cov = z_token - float(np.mean(c) * np.mean(v))
    return z_token, cov

def quartile_correctness_gap(questions: Sequence[Question], score_name: str) -> pd.DataFrame:
    """Within each question at N=32: correctness in the score's top quartile minus its bottom quartile."""
    rows = []
    for q in questions:
        score = q.tokens if score_name == "length" else getattr(q, score_name)
        if score is None:
            continue
        order = np.argsort(score, kind="stable")
        rows.append({"sample_id": q.sample_id, "score": score_name,
                     "top": q.correct[order[-8:]].mean(), "bottom": q.correct[order[:8]].mean()})
    frame = pd.DataFrame(rows)
    frame["gap"] = frame["top"] - frame["bottom"]
    return frame

def within_question_associations(questions: Sequence[Question]) -> pd.DataFrame:
    """Spearman correlation of each signal with length and correctness inside each question."""
    rows = []
    for q in questions:
        signals = {"L": q.L, "K": q.K, "length": q.tokens.astype(float)}
        if q.V is not None:
            signals["V"] = q.V
        for name, values in signals.items():
            with np.errstate(invalid="ignore"):
                row = {"sample_id": q.sample_id, "signal": name}
                row["with_correct"] = spearmanr(values, q.correct).statistic if q.correct.std() > 0 and values.std() > 0 else np.nan
                row["with_length"] = spearmanr(values, q.tokens).statistic if name != "length" and q.tokens.std() > 0 and values.std() > 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)

def shortest_rollout_accuracy(questions: Sequence[Question]) -> float:
    """Length-only control: accuracy of always choosing the shortest rollout at N=32."""
    return float(np.mean([q.correct[np.argmin(q.tokens)] for q in questions]))

def confidence_vision_correlations(questions: Sequence[Question]) -> pd.DataFrame:
    """Per-rollout Pearson correlation between calibrated confidence and visual engagement."""
    rows = []
    for q in questions:
        calibrated = calibrate_tokens(q.kl_tokens)
        for i, (c, v) in enumerate(zip(calibrated, q.v_tokens)):
            if c.size > 2 and c.std() > 0 and v.std() > 0:
                _, cov = compute_tokenwise_fusion(c, v)
                rows.append({"sample_id": q.sample_id, "rollout_id": i, "correlation": float(np.corrcoef(c, v)[0, 1]), "covariance": cov, "correct": bool(q.correct[i])})
    return pd.DataFrame(rows)

def token_trajectory(q: Question, rollout_id: int, bins: int = 40) -> Dict[str, np.ndarray]:
    """Calibrated confidence and engagement along one rollout, averaged into equal position bins."""
    calibrated = calibrate_tokens(q.kl_tokens)[rollout_id]
    v = q.v_tokens[rollout_id]
    edges = np.linspace(0, len(v), min(bins, max(len(v), 1)) + 1).astype(int)
    pairs = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1) if edges[i + 1] > edges[i]]
    return {"position": np.array([s / max(len(v), 1) for s, _ in pairs]),
            "confidence": np.array([calibrated[s:e].mean() for s, e in pairs]),
            "engagement": np.array([v[s:e].mean() for s, e in pairs])}
