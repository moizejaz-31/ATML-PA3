# Pretrained Adaptation, Visual Grounding, and Test-Time Compute

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org/)
[![Random Seed: 6304](https://img.shields.io/badge/Seed-6304-green.svg)](shared/src/seed.py)
[![Course: EE--5102/CS--6304](https://img.shields.io/badge/Course-ATML--Fall2026-purple.svg)](manual/ATML-PA3-manual.md)

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
├── discussion_questions.md             # Detailed breakdown of all 14 research questions
├── manual/
│   ├── ATML-PA3.pdf                    # Official course assignment manual
│   ├── manual_extracted.txt            # Raw text extraction of all 15 manual pages
│   └── ATML-PA3-manual.md              # Formatted Markdown specification of manual
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
│   ├── notebooks/                      # 5 step-by-step notebooks (Data, Training, Eval, Diagnostics, Qualitative)
│   ├── src/                            # Modular implementations (Data, L2-SP, LoRA, OPLoRA, SVD, Eval)
│   ├── checkpoints/                    # Saved checkpoints and run histories
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

### Task 1: LLM Fine-Tuning
1. **Baseline Evaluation:** Run `task1_llm_finetuning/notebooks/1_data_and_baselines.ipynb` to evaluate the unchanged `Qwen/Qwen3-0.6B` baseline zero-shot across GSM8K, HellaSwag, BoolQ, and ARC-Easy.
2. **Training Runs:** Execute `task1_llm_finetuning/notebooks/2_training_runs.ipynb` for Full FT, Full FT + L2-SP, LoRA-4, LoRA-16, and OPLoRA-16 with checkpointing at 312, 625, 937, and 1,250 updates.
3. **Diagnostics & Qualitative:** Execute notebooks `3`, `4`, and `5` for lm-eval benchmark comparisons, SVD subspace interference calculations, and qualitative prompt audits.

### Task 2: Vision-Language Models & Grounding
1. **Feature Caching:** Run `task2_vlm_grounding/notebooks/1_siglip_feature_caching.ipynb` to cache FP16 SigLIP patch features.
2. **Curriculum Training:** Progress through notebooks `3_stage1_caption_alignment.ipynb`, `4_stage2_visual_instruction_tuning.ipynb`, and `5_stage3_vqa_specialization.ipynb`.
3. **Grounding Evaluation:** Execute `6_grounding_evaluation_and_controls.ipynb` to evaluate the model under the four image conditions and compute bootstrap confidence intervals.

### Task 3: Test-Time Compute
1. **Data Ingestion:** Place `ATML-PA3-Data` parquet files into `task3_test_time_compute/data/`.
2. **Offline Evaluation:** Run `task3_test_time_compute/notebooks/1_data_loading_and_oracle_coverage.ipynb` and `2_quality_scoring_and_rwc.ipynb` to analyze candidate budgets $N \in \{1, 2, 4, 8, 16, 32\}$.
3. **Live Refinement & Adaptive Sampling:** Execute notebooks `4`, `5`, and `6` to evaluate the 3,072-token budget strategies on the 50 MATH problems and simulate adaptive stopping controllers.

---

## Deliverables & Academic Integrity
* **Code & Experiments:** All source code, configs, notebooks, and machine-readable outputs are tracked in this repository.
* **Report:** 8-page NeurIPS-style PDF report synthesizing empirical evidence and answering the 14 research questions.
* **Author:** Moiz Ejaz (Roll Number / Student ID: 28100357).
