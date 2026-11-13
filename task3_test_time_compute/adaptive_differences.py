"""
Accuracy difference of each selected adaptive-stopping policy from fixed-32 self-consistency on held-out
questions (report Table 5, bottom), with 95% intervals from a question-level paired bootstrap
(1,000 resamples, seed 6304, the same routine as the other Task 3 comparisons).

Reads results/adaptive_outcomes_<dataset>_<policy>.csv and writes results/adaptive_differences.csv.

Usage (from the repository root):
    python -m task3_test_time_compute.adaptive_differences
"""
import os
import pandas as pd

from .src.evaluate import paired_bootstrap

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def per_question(dataset: str, policy: str) -> pd.DataFrame:
    """Accuracy and token cost per question, averaged over the ten reveal orders."""
    outcomes = pd.read_csv(os.path.join(RESULTS, f"adaptive_outcomes_{dataset}_{policy}.csv"), dtype={"sample_id": str})
    return outcomes.groupby("sample_id")[["correct", "cost"]].mean().sort_index()

def main() -> None:
    rows = []
    for dataset in ["math", "mathvista"]:
        fixed = per_question(dataset, "fixed_32")
        for policy in ["agreement", "quality_aware"]:
            adaptive = per_question(dataset, policy)
            assert adaptive.index.equals(fixed.index), "policies must cover the same questions"
            rows.append({"dataset": dataset, "policy": policy, "accuracy": adaptive["correct"].mean() * 100,
                         "token_saving_pct": (1 - adaptive["cost"].mean() / fixed["cost"].mean()) * 100,
                         **paired_bootstrap(adaptive["correct"].to_numpy(), fixed["correct"].to_numpy())})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS, "adaptive_differences.csv"), index=False)
    print(table.round(2).to_string(index=False))

if __name__ == "__main__":
    main()
