# Task 1: Large Language Model Fine-Tuning

This task investigates supervised fine-tuning of `Qwen/Qwen3-0.6B` on mathematical reasoning (`MetaMathQA`), analyzing the trade-offs between domain adaptation (measured on GSM8K) and capability retention / catastrophic forgetting (measured on HellaSwag, BoolQ, and ARC-Easy).

## Fine-Tuning Variants
1. **Full Fine-Tuning (Full FT):** All parameters updated freely ($\lambda = 0$, peak LR $2 \times 10^{-5}$).
2. **Full FT with L2-SP:** Proximity penalty anchored at starting checkpoint $\theta_0$ ($\lambda = 10^{-3}$, peak LR $2 \times 10^{-5}$).
3. **LoRA-4:** Low-Rank Adaptation with rank $r=4, \alpha=8$ (peak LR $2 \times 10^{-4}$).
4. **LoRA-16:** Low-Rank Adaptation with rank $r=16, \alpha=32$ (peak LR $2 \times 10^{-4}$).
5. **OPLoRA-16:** Orthogonal Projection LoRA with rank $r=16, \alpha=32$, and protection rank $k=128$ projecting updates outside leading singular bases of $W_0$.

## Key Modules
* `src/data.py`: MetaMathQA dataset loader, sequence filtering, prompt/padding masking.
* `src/train.py`: Unified training loop supporting FP16 mixed precision, FP32 master/adapter weights, AdamW, and cosine decay.
* `src/l2sp.py`: Implementation of L2-SP regularization loss $\Omega(\theta, \theta_0)$.
* `src/lora.py`: Linear LoRA adapter layer targeting attention and MLP projections.
* `src/oplora.py`: Orthogonal projection adapter computing SVD bases and effective factors $\tilde{A}, \tilde{B}$.
* `src/diagnostics.py`: Subspace interference $\rho_k$, relative update magnitude $\eta$, and prompt activation ratio $R(z)$.
* `src/evaluate.py`: Zero-shot evaluation harness wrapper and GSM8K CoT answer extraction.

## Notebook Walkthrough
1. `notebooks/1_data_and_baselines.ipynb`: Data filtering, tokenization checks, and unchanged baseline evaluation.
2. `notebooks/2_training_runs.ipynb`: Execution and logging for the 5 fine-tuning variants.
3. `notebooks/3_benchmark_evaluations.ipynb`: Zero-shot evaluation across GSM8K, HellaSwag, BoolQ, and ARC-Easy.
4. `notebooks/4_weight_and_activation_diagnostics.ipynb`: Weight-space SVD projection and activation probe diagnostics.
5. `notebooks/5_qualitative_analysis.ipynb`: Qualitative prompt audit across plain, JSON, and template formats.
