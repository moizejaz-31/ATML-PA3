"""
Data loading and parquet joining for Task 3 rollout tables.
Joins generations.parquet, probability_metrics.parquet, and attention_metrics.parquet
on (dataset, sample_id, rollout_id), then builds per-question arrays for fast analysis.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .metrics import compute_mean_logp, compute_self_certainty, compute_token_engagement
from .normalization import (MathEquivalence, NO_ANSWER, group_answers, normalize_math_answer,
                            normalize_mathvista_answer)

JOIN_KEYS = ["dataset", "sample_id", "rollout_id"]
DRIVE_FOLDER = "https://drive.google.com/drive/folders/1utX-Gfi9vvaoi1mi0pzNpDGL1o1fAh23"
MATHVISTA_METADATA = "https://raw.githubusercontent.com/lupantech/MathVista/main/data/testmini.json"
FOLDERS = {"math": "qwen25-math", "mathvista": "qwen25vl-mathvista"}
EXPECTED_ROWS = 32000

def load_and_join_rollouts(folder_path: str, is_vlm: bool = False) -> pd.DataFrame:
    """Load parquet tables from a model folder and join them. Every file must have 32,000 aligned rows."""
    tables = ["generations.parquet", "probability_metrics.parquet"] + (["attention_metrics.parquet"] if is_vlm else [])
    df = None
    for table in tables:
        part = pd.read_parquet(os.path.join(folder_path, table))
        if df is None:
            df = part
        else:
            # The attention table's per-token count list is not the scalar length in generations
            part = part.rename(columns={"n_generated_text_tokens": "attn_preceding_counts"})
            df = df.merge(part, on=JOIN_KEYS, validate="one_to_one")
    return df

def verify_alignment(df: pd.DataFrame, is_vlm: bool = False, expected_rows: int = EXPECTED_ROWS) -> Dict[str, Any]:
    """Check row counts and that every per-token list aligns with generated_token_ids."""
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rollouts, found {len(df)}")
    lengths = df["generated_token_ids"].map(len)
    columns = ["logp", "kl_u_p"] + (["img_attn_total", "prompt_text_attn_total", "generated_text_attn_total"] if is_vlm else [])
    for column in columns:
        if not (df[column].map(len) == lengths).all():
            raise ValueError(f"{column} is not aligned with generated_token_ids")
    if not (lengths == df["n_generated_text_tokens"]).all():
        raise ValueError("n_generated_text_tokens disagrees with generated_token_ids")
    return {"rollouts": len(df), "questions": df["sample_id"].nunique(), "retained_tokens": int(lengths.sum()),
            "capped": int(df["hit_max_new_tokens"].sum()), "null_extractions": int(df["extracted_answer"].isna().sum())}

def load_mathvista_metadata(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return {str(k): v for k, v in json.load(f).items()}

@dataclass
class Question:
    """All 32 rollouts of one question, ordered by rollout_id."""
    dataset: str
    sample_id: str
    gold: str
    groups: np.ndarray           # integer answer group per rollout
    group_names: List[str]
    group_correct: np.ndarray    # whether each group equals the gold answer
    tokens: np.ndarray
    capped: np.ndarray
    L: np.ndarray
    K: np.ndarray
    V: Optional[np.ndarray] = None
    kl_tokens: List[np.ndarray] = field(default_factory=list)
    v_tokens: List[np.ndarray] = field(default_factory=list)

    @property
    def correct(self) -> np.ndarray:
        return self.group_correct[self.groups]

def build_questions(df: pd.DataFrame, dataset: str, metadata: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Question]:
    """Normalize answers, form groups without gold answers, then grade with the same rules."""
    is_vlm = dataset == "mathvista"
    equivalent = MathEquivalence() if not is_vlm else (lambda a, b: a == b)
    questions = []
    for sample_id, rows in df.sort_values(["sample_id", "rollout_id"]).groupby("sample_id", sort=True):
        if len(rows) != 32:
            raise ValueError(f"{sample_id}: expected 32 rollouts, found {len(rows)}")
        if is_vlm:
            meta = metadata[str(sample_id)]
            normalized = [normalize_mathvista_answer(a, meta) for a in rows["extracted_answer"]]
            gold = normalize_mathvista_answer(rows["gold_answer"].iloc[0], meta)
        else:
            normalized = [normalize_math_answer(a) for a in rows["extracted_answer"]]
            gold = normalize_math_answer(rows["gold_answer"].iloc[0])

        mapping = group_answers(normalized, equivalent)
        representatives = list(dict.fromkeys(mapping[a] for a in normalized))
        index = {name: i for i, name in enumerate(representatives)}
        kl_tokens = [np.asarray(k, dtype=np.float32) for k in rows["kl_u_p"]]

        question = Question(
            dataset=dataset,
            sample_id=str(sample_id),
            gold=gold,
            groups=np.array([index[mapping[a]] for a in normalized]),
            group_names=representatives,
            group_correct=np.array([name != NO_ANSWER and equivalent(gold, name) for name in representatives]),
            tokens=rows["n_generated_text_tokens"].to_numpy(),
            capped=rows["hit_max_new_tokens"].to_numpy(),
            L=np.array([compute_mean_logp(l) for l in rows["logp"]]),
            K=np.array([compute_self_certainty(k) for k in kl_tokens]),
            kl_tokens=kl_tokens,
        )
        if is_vlm:
            question.v_tokens = [compute_token_engagement(i, p, g).astype(np.float32) for i, p, g in
                                 zip(rows["img_attn_total"], rows["prompt_text_attn_total"], rows["generated_text_attn_total"])]
            question.V = np.array([v.mean() if v.size else 0.0 for v in question.v_tokens])
        questions.append(question)
    return questions

def questions_table(questions: List[Question]) -> pd.DataFrame:
    """One row per rollout, for saving and for the signal analysis."""
    rows = []
    for q in questions:
        for i in range(len(q.groups)):
            rows.append({"dataset": q.dataset, "sample_id": q.sample_id, "rollout_id": i, "answer_group": q.group_names[q.groups[i]],
                         "correct": bool(q.correct[i]), "tokens": int(q.tokens[i]), "capped": bool(q.capped[i]),
                         "L": q.L[i], "K": q.K[i], "V": q.V[i] if q.V is not None else np.nan})
    return pd.DataFrame(rows)
