"""
Paired bootstrap differences between Task 1 conditions (report Table 9 and the intervals quoted in Section 2).

Reads the per-example outputs in results/eval/<condition>/<task>_samples.jsonl, resamples benchmark items
with replacement (2,000 resamples, seed 6304), and writes results/paired_differences.csv with the mean
difference in percentage points and its 95% percentile interval.

Usage (from the repository root):
    python -m task1_llm_finetuning.paired_differences
"""
import json
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(ROOT, "results", "eval")
OUT_PATH = os.path.join(ROOT, "results", "paired_differences.csv")
SEED, RESAMPLES = 6304, 2000

# task -> per-example field holding the primary metric (matches Table 1)
METRICS = {"gsm8k_cot": "correct_flexible_extract", "hellaswag": "acc_norm", "boolq": "acc", "arc_easy": "acc_norm"}
COMPARISONS = [
    ("full_ft_625", "baseline"), ("full_ft_1250", "baseline"), ("full_ft_l2sp_1250", "baseline"),
    ("lora_r4_1250", "baseline"), ("lora_r16_1250", "baseline"), ("oplora_r16_1250", "baseline"),
    ("full_ft_1250", "full_ft_625"), ("full_ft_l2sp_1250", "full_ft_1250"), ("lora_r16_1250", "lora_r4_1250"),
    ("oplora_r16_1250", "lora_r16_1250"), ("lora_r16_1250", "full_ft_1250"),
]

def per_example(condition: str, task: str) -> pd.Series:
    """Primary metric per benchmark item, indexed by doc_id."""
    rows = [json.loads(line) for line in open(os.path.join(EVAL_DIR, condition, f"{task}_samples.jsonl"), encoding="utf-8")]
    return pd.Series({r["doc_id"]: float(r[METRICS[task]]) for r in rows}).sort_index()

def paired_interval(a: pd.Series, b: pd.Series, rng: np.random.Generator) -> dict:
    assert a.index.equals(b.index), "conditions must cover the same items"
    diff = (a.to_numpy() - b.to_numpy()) * 100.0
    means = diff[rng.integers(0, len(diff), size=(RESAMPLES, len(diff)))].mean(axis=1)
    return {"difference_pp": diff.mean(), "ci_low": np.percentile(means, 2.5), "ci_high": np.percentile(means, 97.5), "n_items": len(diff)}

def main() -> None:
    cache = {}
    rows = []
    for task in METRICS:
        rng = np.random.default_rng(SEED)
        for a, b in COMPARISONS:
            for c in (a, b):
                if (c, task) not in cache:
                    cache[(c, task)] = per_example(c, task)
            rows.append({"comparison": f"{a} - {b}", "task": task, **paired_interval(cache[(a, task)], cache[(b, task)], rng)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT_PATH, index=False)
    print(table.round(2).to_string(index=False))
    print(f"Saved {OUT_PATH}")

if __name__ == "__main__":
    main()
