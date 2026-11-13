"""
Report Figure 1: where the headline score and the behaviour come apart, one panel per task.
Reads only saved result files:
  (a) task1_llm_finetuning/results/paired_differences.csv
  (b) task2_vlm_grounding/results/final_by_question_type.csv
  (c) task3_test_time_compute/results/failure_decomposition_{math,mathvista}.csv

Usage (from the repository root):
    python make_overview_figure.py [output.png]
"""
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1] if len(sys.argv) > 1 else "task_overview.png"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), gridspec_kw={"width_ratios": [1.05, 1.0, 1.05]})

# (a) Task 1: equal adaptation, different retention (paired 95% CIs against the baseline)
ax = axes[0]
pd_t1 = pd.read_csv(f"{repo}/task1_llm_finetuning/results/paired_differences.csv")
def ci(condition, task):
    r = pd_t1[(pd_t1.comparison == f"{condition} - baseline") & (pd_t1.task == task)].iloc[0]
    return r.difference_pp, r.ci_low, r.ci_high
styles = [  # condition, label, colour, marker, label offset (points)
    ("full_ft_625", "Full FT (625)", BLUE, "o", (-24, -16)),
    ("full_ft_1250", "Full FT", BLUE, "s", (-8, -16)),
    ("full_ft_l2sp_1250", "Full FT\n+ L2-SP", ORANGE, "D", (6, 3)),
    ("lora_r4_1250", "LoRA-4", AQUA, "o", (-34, 5)),
    ("lora_r16_1250", "LoRA-16 (square)\nOPLoRA-16 (triangle)", AQUA, "s", (-24, 12)),
    ("oplora_r16_1250", None, VIOLET, "^", None),
]
runs = [(label, *ci(cond, "gsm8k_cot"), *ci(cond, "boolq"), c, m, off) for cond, label, c, m, off in styles]
ax.axhline(0, color=INK2, lw=0.7, ls="--", zorder=0)
for label, g, glo, ghi, b, blo, bhi, c, m, off in runs:
    ax.errorbar(g, b, xerr=[[g - glo], [ghi - g]], yerr=[[b - blo], [bhi - b]], fmt=m, ms=4.5,
                color=c, ecolor=c, elinewidth=0.8, capsize=0, alpha=0.95, mec="white", mew=0.6)
    if label:
        ax.annotate(label, (g, b), xytext=off, textcoords="offset points", fontsize=6.2, color=INK, linespacing=0.95)
ax.text(10.3, -1.6, "no BoolQ change", fontsize=6, color=INK2)
ax.set_xlim(10, 27.5); ax.set_ylim(-11.5, 15.5)
ax.set_xlabel("GSM8K gain over baseline (pp)")
ax.set_ylabel("BoolQ change (pp)")
ax.set_title("(a) Task 1: same maths gain,\ndifferent retention", fontsize=7.5, loc="left", color=INK)
ax.grid(True, color=GRID, lw=0.4); ax.set_axisbelow(True)

# (b) Task 2: correct image against the complementary swap, by question type
ax = axes[1]
qt = pd.read_csv(f"{repo}/task2_vlm_grounding/results/final_by_question_type.csv")
order = [("grounding", "presence", "Presence"), ("grounding", "counting", "Counting"),
         ("natural", "number", "VQA number"), ("natural", "other", "VQA other"), ("natural", "yes/no", "VQA yes/no")]
for y, (suite, typ, name) in enumerate(order):
    row = qt[(qt.suite == suite) & (qt.type == typ)].iloc[0]
    cor, swp = row["Correct image"], row["Complementary swap"]
    ax.plot([swp, cor], [y, y], color=GRID, lw=2.2, zorder=1, solid_capstyle="round")
    ax.scatter([cor], [y], s=26, color=BLUE, zorder=3, edgecolor="white", lw=0.6)
    ax.scatter([swp], [y], s=26, color=ORANGE, zorder=3, edgecolor="white", lw=0.6)
    if y == 0:
        ax.text(cor, y - 0.33, "correct", ha="center", fontsize=6, color=INK)
        ax.text(swp, y - 0.33, "swap", ha="center", fontsize=6, color=INK)
    ax.text(max(cor, swp) + 3, y, f"gap {cor - swp:.1f}", va="center", fontsize=6.3,
            color=INK, fontweight="bold" if typ == "yes/no" else "normal")
ax.set_yticks(range(len(order))); ax.set_yticklabels([o[2] for o in order])
ax.set_ylim(len(order) - 0.5, -0.8)
ax.set_xlim(0, 118); ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("Accuracy (%)")
ax.set_title("(b) Task 2: image use by question\ntype (final test)", fontsize=7.5, loc="left", color=INK)
ax.grid(True, axis="x", color=GRID, lw=0.4); ax.set_axisbelow(True)

# (c) Task 3: oracle coverage against majority-vote accuracy as N grows
ax = axes[2]
for name, c in [("math", BLUE), ("mathvista", ORANGE)]:
    d = pd.read_csv(f"{repo}/task3_test_time_compute/results/failure_decomposition_{name}.csv")
    cov = 100 - d.no_correct_candidate
    ax.fill_between(d.N, d.correct, cov, color=c, alpha=0.13, lw=0)
    ax.plot(d.N, cov, color=c, lw=1.4, ls="--", marker="o", ms=2.8)
    ax.plot(d.N, d.correct, color=c, lw=1.6, marker="o", ms=2.8)
    label = "MATH" if name == "math" else "MathVista"
    ax.text(36, cov.iloc[-1] + (1.7 if name == "math" else -1.7), f"{label} coverage {cov.iloc[-1]:.1f}",
            fontsize=6, va="center", color=INK)
    ax.text(36, d.correct.iloc[-1], f"{label} vote {d.correct.iloc[-1]:.1f}", fontsize=6, va="center", color=INK)
ax.text(2.6, 52.3, "shaded gap = selection failures", fontsize=6, color=INK2, ha="left")
ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
ax.set_xlim(0.9, 32); ax.set_ylim(50, 96)
ax.set_xlabel("Rollouts per question N")
ax.set_ylabel("Held-out accuracy (%)")
ax.set_title("(c) Task 3: coverage (dashed) vs.\nmajority vote (solid)", fontsize=7.5, loc="left", color=INK)
ax.grid(True, color=GRID, lw=0.4); ax.set_axisbelow(True)

fig.tight_layout(w_pad=1.2)
fig.subplots_adjust(right=0.83)
fig.savefig(out, dpi=300, facecolor="white")
print("saved", out)
