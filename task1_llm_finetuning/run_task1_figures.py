"""
Builds the Task 1 tables and figures from saved results only. Nothing here is typed in or simulated:
a section whose inputs are missing or incomplete is skipped and reported.

Inputs:  results/training/<run>/, results/eval/<condition>/, results/weight_diagnostics.csv,
         results/activation_diagnostics.csv, results/qualitative/annotations_42.csv
Outputs: results/*.csv, results/figures/*.png
"""
import json, os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from shared.src.plotting import set_style

RESULTS_DIR = os.path.join(script_dir, 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')

RUNS = {"full_ft": "Full FT", "full_ft_l2sp": "Full FT + L2-SP", "lora_r4": "LoRA-4", "lora_r16": "LoRA-16", "oplora_r16": "OPLoRA-16"}
CONDITIONS = {
    "baseline": "Baseline",
    "full_ft_625": "Full FT Midpoint",
    "full_ft_1250": "Full FT Final",
    "full_ft_l2sp_1250": "Full FT + L2-SP",
    "lora_r4_1250": "LoRA-4",
    "lora_r16_1250": "LoRA-16",
    "oplora_r16_1250": "OPLoRA-16",
}
BENCHMARKS = {"gsm8k_cot": ("GSM8K", 1319), "hellaswag": ("HellaSwag", 10042), "boolq": ("BoolQ", 3270), "arc_easy": ("ARC-Easy", 2376)}
RETENTION = ["HellaSwag", "BoolQ", "ARC-Easy"]
PALETTE = {"Full FT": "#D0021B", "Full FT + L2-SP": "#F5A623", "LoRA-4": "#4A90E2", "LoRA-16": "#2ECC71", "OPLoRA-16": "#9013FE"}
ANNOTATION_COLUMNS = ["format_compliance", "coherence_relevance", "repetition", "unnecessary_math_framing", "inappropriate_content"]

skipped = []

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_figure(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=300)
    plt.close()

def load_runs():
    """Runs that finished all their updates: {display name: (history, summary)}."""
    runs = {}
    for run, label in RUNS.items():
        history_path = os.path.join(RESULTS_DIR, 'training', run, 'history.json')
        summary_path = os.path.join(RESULTS_DIR, 'training', run, 'summary.json')
        if not (os.path.exists(history_path) and os.path.exists(summary_path)):
            skipped.append(f"training run {run}: no results")
            continue
        summary = read_json(summary_path)
        if not summary.get("completed"):
            skipped.append(f"training run {run}: not finished")
            continue
        runs[label] = (read_json(history_path), summary)
    return runs

def load_benchmarks():
    """One row per condition that has all four benchmarks on their complete splits."""
    rows = []
    for condition, label in CONDITIONS.items():
        row = {"Condition": label, "condition_id": condition}
        complete = True
        for task, (name, expected) in BENCHMARKS.items():
            path = os.path.join(RESULTS_DIR, 'eval', condition, f"{task}_scores.json")
            if not os.path.exists(path):
                skipped.append(f"benchmark {condition}/{task}: no results")
                complete = False
                continue
            scores = read_json(path)
            if scores["n"] != expected:
                skipped.append(f"benchmark {condition}/{task}: partial split ({scores['n']}/{expected})")
                complete = False
                continue
            row[name] = scores["primary"]
            if "acc" in scores:
                row[f"{name} raw acc"] = scores["acc"]
            if task == "gsm8k_cot":
                row["GSM8K strict"] = scores["exact_match_strict"]
                row["GSM8K truncated %"] = scores["truncated_rate"]
                row["GSM8K unextractable %"] = scores["invalid_flexible_rate"]
                row["GSM8K mean new tokens"] = scores["mean_new_tokens"]
        if complete:
            rows.append(row)
    return pd.DataFrame(rows)

def learning_figures(runs):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (history, _) in runs.items():
        loss = pd.Series(history["sft_loss"]).rolling(window=15, min_periods=1).mean()
        ax.plot(history["update"], loss, label=label, color=PALETTE[label], linewidth=2)
    ax.set_title('SFT Loss (15-update rolling mean)')
    ax.set_xlabel('Optimizer Updates')
    ax.set_ylabel('Cross-Entropy per Response Token')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    save_figure('task1_learning_curves.png')

    if "Full FT + L2-SP" in runs:
        history = runs["Full FT + L2-SP"][0]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(history["update"], history["l2sp_penalty"], color=PALETTE["Full FT + L2-SP"], linewidth=2)
        ax.set_title(r'L2-SP Penalty $\lambda \Omega(\theta, \theta_0)$')
        ax.set_xlabel('Optimizer Updates')
        ax.set_ylabel(r'$\lambda \Omega$')
        ax.grid(True, linestyle='--', alpha=0.7)
        save_figure('task1_l2sp_penalty.png')

def resource_table(runs):
    rows = []
    for label, (_, summary) in runs.items():
        final_key = f"evaluation_{summary['max_updates']}"
        recovery_sizes = [v for k, v in summary["checkpoint_sizes_mb"].items() if k.startswith("recovery_")]
        rows.append({
            "Method": label,
            "Trainable Params": summary["trainable_params"],
            "Training Time (min)": summary["training_seconds"] / 60.0,
            "Peak GPU Mem (MB)": summary["peak_gpu_mb"],
            "Final State Size (MB)": summary["checkpoint_sizes_mb"].get(final_key),
            "Recovery Checkpoint Size (MB)": max(recovery_sizes) if recovery_sizes else None,
            "Skipped Updates": len(summary["skipped_updates"]),
            "Final Displacement": summary["displacement"].get(str(summary["max_updates"])),
            "Midpoint Displacement": summary["displacement"].get(str(summary["max_updates"] // 2)),
            "L2-SP Time (min)": summary["l2sp_seconds"] / 60.0,
            "Reference Storage (MB)": summary.get("reference_storage_mb"),
            "SVD Time (s)": summary["preprocessing"].get("svd_seconds"),
            "Basis Storage (MB)": summary["preprocessing"].get("basis_storage_mb"),
            "Adapter Export (s)": summary.get("adapter_export_seconds"),
        })
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS_DIR, 'training_resource_summary.csv'), index=False)
    return table

def benchmark_outputs(table):
    baseline = table[table["condition_id"] == "baseline"]
    if baseline.empty:
        skipped.append("benchmark deltas: baseline not evaluated yet")
        table.to_csv(os.path.join(RESULTS_DIR, 'master_benchmark_results.csv'), index=False)
        return
    names = [name for name, _ in BENCHMARKS.values()]
    for name in names:
        table[f"{name} delta (pp)"] = table[name] - baseline.iloc[0][name]
    table["Mean Retention delta (pp)"] = table[[f"{n} delta (pp)" for n in RETENTION]].mean(axis=1)
    table.to_csv(os.path.join(RESULTS_DIR, 'master_benchmark_results.csv'), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(len(table))
    width = 0.2
    for i, (name, color) in enumerate(zip(names, ["#4A90E2", "#50E3C2", "#F5A623", "#9013FE"])):
        axes[0].bar(x + (i - 1.5) * width, table[name], width, label=name, color=color, edgecolor='black')
        axes[1].bar(x + (i - 1.5) * width, table[f"{name} delta (pp)"], width, label=name, color=color, edgecolor='black')
    axes[0].set_title('Primary Score (%)')
    axes[1].set_title('Change from Baseline (percentage points)')
    axes[1].axhline(0, color='black', linewidth=1)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(table["Condition"], rotation=30, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend()
    save_figure('task1_benchmarks.png')

    trained = table[table["condition_id"] != "baseline"]
    if not trained.empty:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.scatter(trained["GSM8K delta (pp)"], trained["Mean Retention delta (pp)"], s=120, edgecolors='black')
        for _, row in trained.iterrows():
            ax.annotate(row["Condition"], (row["GSM8K delta (pp)"], row["Mean Retention delta (pp)"]), textcoords="offset points", xytext=(7, 4))
        ax.axhline(0, color='gray', linestyle='--')
        ax.axvline(0, color='gray', linestyle='--')
        ax.set_xlabel('GSM8K Change (pp)')
        ax.set_ylabel('Mean Change on HellaSwag, BoolQ, ARC-Easy (pp)')
        ax.set_title('Adaptation vs. Retention')
        ax.grid(True, linestyle='--', alpha=0.7)
        save_figure('task1_adaptation_vs_retention.png')

def diagnostics_figures():
    weight_path = os.path.join(RESULTS_DIR, 'weight_diagnostics.csv')
    if os.path.exists(weight_path):
        weights = pd.read_csv(weight_path)
        weights["Condition"] = weights["condition"].map(CONDITIONS)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.barplot(data=weights, x='layer', y='rho_k', hue='Condition', ax=axes[0], edgecolor='black')
        axes[0].set_title(r'Subspace Interference $\rho_k$ ($k=128$)')
        sns.barplot(data=weights, x='layer', y='eta', hue='Condition', ax=axes[1], edgecolor='black')
        axes[1].set_title(r'Relative Update Magnitude $\eta$')
        save_figure('task1_weight_diagnostics.png')
    else:
        skipped.append("weight diagnostics: no results")

    activation_path = os.path.join(RESULTS_DIR, 'activation_diagnostics.csv')
    if os.path.exists(activation_path):
        activations = pd.read_csv(activation_path)
        activations["Condition"] = activations["condition"].map(CONDITIONS)
        activations.groupby(["Condition", "layer", "domain"])["R"].agg(["mean", "median", "std"]).reset_index().to_csv(
            os.path.join(RESULTS_DIR, 'activation_summary.csv'), index=False)
        layers = sorted(activations["layer"].unique())
        fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 5), squeeze=False)
        for ax, layer in zip(axes[0], layers):
            sns.boxplot(data=activations[activations["layer"] == layer], x='Condition', y='R', hue='domain', ax=ax)
            ax.set_title(f'$R(z)$, down_proj layer {layer}')
            ax.tick_params(axis='x', rotation=30)
        save_figure('task1_activation_diagnostics.png')
    else:
        skipped.append("activation diagnostics: no results")

def qualitative_outputs():
    path = os.path.join(RESULTS_DIR, 'qualitative', 'annotations_42.csv')
    if not os.path.exists(path):
        skipped.append("qualitative annotations: responses not generated yet")
        return
    sheet = pd.read_csv(path)
    annotated = sheet.dropna(subset=ANNOTATION_COLUMNS, how='all')
    if len(annotated) < 42:
        skipped.append(f"qualitative annotations: {len(annotated)}/42 annotated by hand")
        return
    numeric = annotated[ANNOTATION_COLUMNS].apply(pd.to_numeric, errors='coerce')
    summary = pd.concat([annotated[["condition", "format"]], numeric], axis=1).groupby(["condition", "format"]).mean()
    summary.to_csv(os.path.join(RESULTS_DIR, 'qualitative_summary.csv'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build Task 1 tables and figures from saved results")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()
    RESULTS_DIR = args.results_dir
    FIG_DIR = os.path.join(RESULTS_DIR, 'figures')

    set_style()
    os.makedirs(FIG_DIR, exist_ok=True)

    runs = load_runs()
    if runs:
        learning_figures(runs)
        resource_table(runs)

    benchmark_table = load_benchmarks()
    if not benchmark_table.empty:
        benchmark_outputs(benchmark_table)

    diagnostics_figures()
    qualitative_outputs()

    print("=== Task 1 tables and figures ===")
    print(f"Finished runs: {list(runs)}")
    print(f"Conditions with complete benchmarks: {len(benchmark_table)}/7")
    if skipped:
        print("Not built yet:")
        for reason in skipped:
            print(f"  - {reason}")
