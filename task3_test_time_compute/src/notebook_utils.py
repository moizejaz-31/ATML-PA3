"""Shared paths, cached question loading, and plotting style for the Task 3 notebooks."""
import json
import os
import pickle
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data import FOLDERS, build_questions, load_and_join_rollouts, load_mathvista_metadata, Question
from .splits import create_splits

_TASK3_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.environ.get("TASK3_DATA_DIR", os.path.join(_TASK3_DIR, "data"))
RESULTS_DIR = os.environ.get("TASK3_RESULTS_DIR", os.path.join(_TASK3_DIR, "results"))
SPLITS_DIR = os.environ.get("TASK3_SPLITS_DIR", os.path.join(_TASK3_DIR, "splits"))
CACHE_DIR = os.environ.get("TASK3_CACHE_DIR", os.path.join(_TASK3_DIR, "checkpoints"))
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
DATASETS = {"math": "MATH (Qwen2.5-3B)", "mathvista": "MathVista (Qwen2.5-VL-3B)"}

METHOD_LABELS = {
    "first_ans": "FirstAns", "self_consistency": "Self-consistency", "mean_logp": "Highest log-prob",
    "self_certainty": "Highest self-certainty", "self_certainty_rwc": "Self-certainty + RWC",
    "vision_only": "Highest vision", "vision_only_rwc": "Vision + RWC",
    "visconf": "Highest VisConf", "visconf_rwc": "VisConf + RWC",
    "visconf_token": "Highest tokenwise VisConf", "visconf_token_rwc": "Tokenwise VisConf + RWC",
}
METHOD_COLORS = dict(zip(METHOD_LABELS, sns.color_palette("tab20", len(METHOD_LABELS))))

def setup_notebook() -> None:
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.titlesize": 13, "axes.labelsize": 11})
    for path in [RESULTS_DIR, FIG_DIR, SPLITS_DIR, CACHE_DIR]:
        os.makedirs(path, exist_ok=True)

def save_figure(name: str) -> None:
    """Save the current figure with the task's other figures, then show it inline."""
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{name}.png"))
    plt.show()

def load_questions(dataset: str) -> List[Question]:
    """Join tables, normalize, and group answers once. Later notebooks reuse the cached result."""
    cache_path = os.path.join(CACHE_DIR, f"questions_{dataset}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    is_vlm = dataset == "mathvista"
    df = load_and_join_rollouts(os.path.join(DATA_DIR, FOLDERS[dataset]), is_vlm=is_vlm)
    metadata = load_mathvista_metadata(os.path.join(DATA_DIR, "mathvista_testmini.json")) if is_vlm else None
    questions = build_questions(df, dataset, metadata)
    with open(cache_path + ".tmp", "wb") as f:
        pickle.dump(questions, f)
    os.replace(cache_path + ".tmp", cache_path)
    return questions

def load_splits(dataset: str, questions: List[Question]) -> Dict[str, List[str]]:
    return create_splits([q.sample_id for q in questions], save_path=os.path.join(SPLITS_DIR, f"{dataset}_splits_seed6304.json"))

def split_questions(dataset: str) -> Tuple[List[Question], List[Question], Dict[str, List[str]]]:
    """(development questions, held-out questions, split IDs). The IDs are saved for the repository."""
    questions = load_questions(dataset)
    splits = load_splits(dataset, questions)
    by_id = {q.sample_id: q for q in questions}
    return [by_id[i] for i in splits["dev"]], [by_id[i] for i in splits["test"]], splits

def require_results(name: str) -> pd.DataFrame:
    """Load a saved table or stop with a clear message. Figures never fall back to placeholder values."""
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Run the earlier Task 3 notebooks first.")
    return pd.read_csv(path)
