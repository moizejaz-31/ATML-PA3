# Pretrained Adaptation, Visual Grounding, and Test-Time Compute

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org/)
[![Random Seed: 6304](https://img.shields.io/badge/Seed-6304-green.svg)](shared/src/seed.py)
![Course: EE--5102/CS--6304](https://img.shields.io/badge/Course-ATML--Fall2026-purple.svg)

> **Course:** EE-5102 / CS-6304: Advanced Topics in Machine Learning  
> **Institution:** Department of Computer Science, Lahore University of Management Sciences (LUMS)  
> **Assignment:** Programming Assignment 3  

---

## Executive Summary & Core Research Questions

Pretrained foundation models serve as strong initializations for downstream applications, but adapting and deploying them introduces critical open questions regarding parameter updates, cross-modal integration, and inference-time scaling:

1. **Task 1: Large Language Model Fine-Tuning (Adaptation vs. Retention):**
   * *Domain Adaptation & Forgetting:* Fine-tuning `Qwen/Qwen3-0.6B` on mathematical solutions (`MetaMathQA`) and measuring mathematical reasoning gains (`GSM8K`) against catastrophic forgetting on general reasoning capabilities (`HellaSwag`, `BoolQ`, `ARC-Easy`).
   * *Update Restrictions:* Controlled comparison across five fine-tuning paradigms: Full Fine-Tuning (Full FT), Full FT with starting-point proximity regularization ($\text{L2-SP}, \lambda=10^{-3}$), Low-Rank Adaptation ($\text{LoRA}, r \in \{4, 16\}$), and Orthogonal Projection LoRA ($\text{OPLoRA}, r=16, k=128$) projecting adapter updates outside leading singular subspaces of the starting weights.
   * *Geometric & Activation Diagnostics:* Evaluating the subspace-interference ratio $\rho_k$, relative update magnitude $\eta$, and local prompt activation ratio $R(z)$ on layers 0, 13, and 27 of `down_proj`.
   * *Qualitative Audit:* Systematically evaluating 168 generations across 24 prompts from `first_plot_questions.yaml` (8 questions $\times$ 3 formats: plain, JSON, template) and annotating the 42-response focal subset.

2. **Task 2: Vision-Language Models and Visual Grounding:**
   * *Architecture:* Connecting a frozen `google/siglip-base-patch16-224` vision encoder to `Qwen/Qwen3-0.6B` via a trainable 2-layer MLP connector (`768 -> 1024 -> 1024`).
   * *Curriculum Training:* Sequential 3-stage curriculum with dynamic replay:
     1. Stage 1: Caption Alignment (18k COCO captions, connector only).
     2. Stage 2: Visual Instruction Tuning (presence, counting, caption replay; connector + LoRA).
     3. Stage 3: Natural VQA Specialization (VQAv2 complementary consensus pairs + multi-source replay).
   * *Visual Grounding & Control Suites:* Measuring true visual reliance versus language-prior bias across four distinct image conditions: Correct, Complementary Swap, Mismatched, and No Image.
   * *Metrics:* Normalized exact match, local VQA soft consensus accuracy across 10 human references, pair accuracy, identical-answer rate, reference negative log-likelihood, and image-swap penalty $\Delta\text{NLL}$.

3. **Task 3: Test-Time Compute: Sampling, Selection, and Adaptive Inference:**
   * *Offline Rollout Scaling:* Evaluating candidate budgets $N \in \{1, 2, 4, 8, 16, 32\}$ on 32,000 pre-generated rollouts each for `Qwen2.5-3B-Instruct` (MATH) and `Qwen2.5-VL-3B-Instruct` (MathVista).
   * *Quality-Aware Consensus:* Comparing FirstAns, Self-Consistency (majority voting), Mean Log-Probability, Self-Certainty ($K_i$), Visual Engagement ($V_i$), VisConf ($Z_i$), and Rank-Weighted Consensus (RWC).
   * *Live Sequential Refinement:* FP16 live generation on 50 MATH evaluation questions comparing a single longer response (3,072 tokens), parallel sampling ($3 \times 1,024$ tokens), and sequential critique/revision ($1,024 + 2,048$ tokens).
   * *Adaptive Inference:* Developing dynamic stopping controllers (agreement-based vs. quality-aware with RWC margin) tuned on 200 dev questions and evaluated on 800 held-out test questions to optimize the accuracy-token cost Pareto frontier.

---

## Repository Directory Structure

```
.
├── .gitignore                          # Comprehensive ignore rules (excluding weights/parquet/LaTeX)
├── README.md                           # Master documentation and reproduction guide
├── requirements.txt                    # Exact environment specifications
├── setup.py                            # Local editable package installer
├── make_overview_figure.py             # Report Figure 1, built from saved results
├── shared/
│   ├── README.md                       # Documentation for common utilities
│   ├── configs/
│   │   └── default_config.yaml         # Global seed (6304), device, and precision settings
│   └── src/
│       ├── seed.py                     # Deterministic seeding (Python, NumPy, PyTorch, CUDA)
│       ├── metrics.py                  # Paired bootstrap resampling & 95% CI calculation
│       ├── plotting.py                 # Publication-quality matplotlib/seaborn formatting
│       ├── diagrams.py                 # Architecture visualization utilities
│       └── utils.py                    # GPU memory tracking, timers, parameter counting
├── task1_llm_finetuning/
│   ├── README.md                       # Task 1 guide & experimental protocol
│   ├── configs/                        # Hyperparameter YAMLs (Full FT, L2-SP, LoRA-4, LoRA-16, OPLoRA)
│   ├── splits/                         # Probe set sample IDs (32 GSM8K + 32 HellaSwag)
│   ├── src/                            # Modular implementations (Data, L2-SP, LoRA, OPLoRA, Eval, Diagnostics, Qualitative)
│   ├── tests/                          # Fast CPU checks on a tiny random model
│   ├── checkpoints/                    # Saved checkpoints (not committed)
│   └── results/                        # Machine-readable output CSVs and JSONs
├── task2_vlm_grounding/
│   ├── README.md                       # Task 2 guide & curriculum protocol
│   ├── configs/                        # YAML configs for Stages 1, 2, 3 and 49-token ablation
│   ├── splits/                         # Image-disjoint dataset split manifests
│   ├── notebooks/                      # 7 step-by-step notebooks (Feature caching to grounding evaluation)
│   ├── src/                            # Modules (SigLIP caching, MLP connector, VLM, Sampler, Controls, Eval)
│   ├── checkpoints/                    # Curriculum checkpoint directories
│   └── results/                        # Grounding metric tables and bootstrap results
└── task3_test_time_compute/
    ├── README.md                       # Task 3 guide & test-time compute protocol
    ├── configs/                        # Experiment configs (Offline eval, Live refinement, Adaptive sampling)
    ├── data/                           # Directory for supplied ATML-PA3-Data rollouts
    ├── splits/                         # 200 dev, 800 test, 50 live question splits
    ├── notebooks/                      # 6 step-by-step notebooks (Oracle curves, RWC, Live refinement, Adaptive)
    ├── src/                            # Modules (Parquet joining, Normalization, VisConf, RWC, Stopping, Live)
    ├── checkpoints/                    # Storage for live generation cache and model states
    └── results/                        # Budget tables, Pareto curves, and failure analyses
```

---

## Environment Setup & Installation

### 1. Prerequisites
Ensure Python 3.10+ and a CUDA-capable GPU environment are available.

```bash
# Clone the repository
git clone https://github.com/moizejaz-31/ATML-PA3.git
cd ATML-PA3

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt

# Install local modules in editable mode
pip install -e .
```

### 2. Random Seed & Reproducibility
All experiments strictly enforce the assignment seed **`6304`**:
```python
from shared.src.seed import set_seed
set_seed(6304)
```

---

## Experimental Workflow

All experiments were run on Kaggle (GPU T4 x2 for Tasks 1 and 2, one T4 for Task 3). Every command is run from the repository root, and a stopped run resumes when re-run.

### Task 1: LLM Fine-Tuning
Run the commands in `task1_llm_finetuning/README.md` in order: data selection, training of the five variants, evaluation of the seven conditions, diagnostics, qualitative responses, and the figures. Outputs land in `task1_llm_finetuning/results/` (benchmark scores and per-example outputs, diagnostics, training histories, figures).

### Task 2: Vision-Language Models & Grounding
Run notebooks 1–6 in `task2_vlm_grounding/notebooks/` in order: feature caching, data selection, the three curriculum stages, and the final grounding evaluation. The committed notebooks keep their executed outputs. Notebook 7 is the optional 49-token experiment, which was not run.

### Task 3: Test-Time Compute
Place the supplied `ATML-PA3-Data` rollouts and the MathVista metadata in `task3_test_time_compute/data/`, then run notebooks 1–6 in `task3_test_time_compute/notebooks/` in order. The committed notebooks keep their executed outputs.

### Report statistics and Figure 1 (CPU, from saved results)
These read only the committed result files and take under a minute:
```bash
python -m task1_llm_finetuning.paired_differences      # Task 1 paired bootstrap intervals -> results/paired_differences.csv
python -m task3_test_time_compute.adaptive_differences # adaptive vs fixed-32 intervals   -> results/adaptive_differences.csv
python make_overview_figure.py task_overview.png       # report Figure 1
```

---

## Deliverables & Academic Integrity
* **Code & Experiments:** All source code, configs, notebooks, and machine-readable outputs are tracked in this repository.
* **Report:** 8-page NeurIPS-style PDF report synthesizing empirical evidence and answering the 14 research questions.
* **Author:** Moiz Ejaz (Roll Number / Student ID: 28100357).
* **Attribution:** The LoRA, OPLoRA, L2-SP, VLM connector and Task 3 selection/stopping code is our own implementation of the cited methods; no external code was copied. We build on PyTorch, Hugging Face Transformers and Datasets, lm-evaluation-harness 0.4.13 (benchmark evaluation) and math-verify (MATH answer equivalence). Models (Qwen3-0.6B, SigLIP-base) and datasets (MetaMathQA, GSM8K, HellaSwag, BoolQ, ARC-Easy, COCO, VQAv2, MATH, MathVista, ATML-PA3-Data) are downloaded at runtime and not committed. Code was written with assistance from Claude (Anthropic).
