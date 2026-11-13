# Task 1: Large Language Model Fine-Tuning

This task investigates supervised fine-tuning of `Qwen/Qwen3-0.6B` on mathematical reasoning (`MetaMathQA`), analyzing the trade-offs between domain adaptation (measured on GSM8K) and capability retention / catastrophic forgetting (measured on HellaSwag, BoolQ, and ARC-Easy).

## Fine-Tuning Variants
1. **Full Fine-Tuning (Full FT):** All parameters updated freely ($\lambda = 0$, peak LR $2 \times 10^{-5}$).
2. **Full FT with L2-SP:** Proximity penalty anchored at starting checkpoint $\theta_0$ ($\lambda = 10^{-3}$, peak LR $2 \times 10^{-5}$).
3. **LoRA-4:** Low-Rank Adaptation with rank $r=4, \alpha=8$ (peak LR $2 \times 10^{-4}$).
4. **LoRA-16:** Low-Rank Adaptation with rank $r=16, \alpha=32$ (peak LR $2 \times 10^{-4}$).
5. **OPLoRA-16:** Orthogonal Projection LoRA with rank $r=16, \alpha=32$, and protection rank $k=128$ projecting updates outside leading singular bases of $W_0$.

## Key Modules
* `src/data.py`: MetaMathQA selection, non-thinking chat formatting, prompt/padding masking, probe set.
* `src/train.py`: Training loop with effective-batch token normalization, FP16 autocast, resumable recovery checkpoints, and resource logging.
* `src/l2sp.py`: L2-SP penalty $\Omega(\theta, \theta_0)$ and displacement.
* `src/lora.py`: LoRA adapters, factor export, and merging into fresh backbones.
* `src/oplora.py`: OPLoRA adapters with SVD bases and projected factors $\tilde{A}, \tilde{B}$.
* `src/conditions.py`: The seven evaluation conditions and the merged-versus-unmerged logit check.
* `src/evaluate.py`: Zero-shot benchmarks through lm-evaluation-harness, with per-example outputs.
* `src/diagnostics.py`: Subspace interference $\rho_k$, relative update magnitude $\eta$, and activation ratio $R(z)$.
* `src/qualitative.py`: The 168 qualitative responses and the blank sheet for the 42 hand annotations.
* `run_task1_figures.py`: Tables and figures, built only from saved results.
* `tests/test_task1.py`: Fast CPU checks on a tiny random model.

## Reproducing
Run these from the repository root, in order (we ran them on Kaggle, GPU T4 x2, over several sessions):

```bash
python -m pytest task1_llm_finetuning/tests -q
python -m task1_llm_finetuning.src.data --seed 6304
python -m task1_llm_finetuning.src.train --config task1_llm_finetuning/configs/lora_r16.yaml
python -m task1_llm_finetuning.src.evaluate --condition lora_r16_1250
python -m task1_llm_finetuning.src.diagnostics
python -m task1_llm_finetuning.src.qualitative
python task1_llm_finetuning/run_task1_figures.py
```

Evaluation conditions are `baseline`, `full_ft_625`, `full_ft_1250`, `full_ft_l2sp_1250`, `lora_r4_1250`, `lora_r16_1250`, and `oplora_r16_1250`. Re-running a stopped command resumes it.

## Deviation From the Manual
The manual asks for recovery checkpoints at updates 312, 625, 937, and 1,250. All four are written, but by default each one replaces the previous one, and the last is removed once a run finishes, because a Full FT recovery file is about 7 GB and Kaggle keeps about 20 GB. Weights or adapters at every evaluation stage are always kept. Pass `--keep_all_recovery` to keep all four. State this in the report if you use the default.

## External Material
`configs/first_plot_questions.yaml` is from the [emergent-misalignment repository](https://github.com/emergent-misalignment/emergent-misalignment).
