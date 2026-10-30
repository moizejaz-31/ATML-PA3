# Task 2: Vision-Language Models and Visual Grounding

This task constructs a vision-language model (VLM) connecting a frozen `google/siglip-base-patch16-224` vision encoder to `Qwen/Qwen3-0.6B` using a trainable 2-layer MLP connector (`768 -> 1024 -> 1024`). The model is trained through a 3-stage curriculum and evaluated for genuine visual grounding versus language-prior exploitation using complementary image pairs and control conditions.

## Architecture
* **Vision Encoder:** `google/siglip-base-patch16-224` (frozen; 196 patch features $14 \times 14$ in spatial order; final layer norm applied once; cached in FP16).
* **Connector:** 2-layer MLP (`Linear 768 -> 1024, GELU, Linear 1024 -> 1024`), trainable across all stages.
* **Language Model:** `Qwen/Qwen3-0.6B` (`enable_thinking=False`, frozen backbone, LoRA in Stages 2-3 on `q_proj`, `k_proj`, `v_proj`, `o_proj` with $r=16, \alpha=32$).

## Three-Stage Curriculum (4,500 updates, 72,000 presentations)
1. **Stage 1: Caption Alignment (Updates 1–2,250):** 18,000 COCO caption examples (2 passes). Train connector only (peak LR $10^{-3}$).
2. **Stage 2: Visual Instruction Tuning (Updates 2,251–3,750):** Mixture: 50% presence pairs, 25% counting pairs, 25% caption replay. Connector peak LR $10^{-4}$, LoRA peak LR $2 \times 10^{-5}$.
3. **Stage 3: Natural VQA Specialization (Updates 3,751–4,500):** Mixture: 70% VQAv2 pairs, 10% presence replay, 10% counting replay, 10% caption replay. Connector peak LR $5 \times 10^{-5}$, LoRA peak LR $10^{-5}$.

## Visual Grounding Evaluation & Controls
* **Conditions:** Correct Image, Complementary Swap, Mismatched Image, No Image.
* **Metrics:** Normalized exact match, local VQA soft consensus accuracy across 10 human answers, pair accuracy, identical-answer rate, empty-answer rate, reference NLL, image-swap penalty $\Delta\text{NLL}$, and 95% bootstrap CIs (4,000 complete pair resamples).
* **Optional Token Compression:** 49 visual tokens via $2 \times 2$ spatial average pooling.

## Directory Structure
* `configs/`: Hyperparameter YAML files for the three curriculum stages and optional 49-token run.
* `splits/`: Train, development, and final-test split IDs (image-disjoint).
* `src/feature_cache.py`: Deterministic preprocessing and SigLIP feature caching.
* `src/connector.py`: 2-layer MLP connector with text embedding magnitude calibration.
* `src/vlm_model.py`: Hybrid VLM model wrapper inserting visual tokens before user questions.
* `src/data.py`: COCO caption loader, grounding question generator, and VQAv2 pair filter.
* `src/sampler.py`: Curriculum stage mixture sampler.
* `src/train.py`: Unified curriculum trainer with example-weighted causal loss.
* `src/controls.py`: Control image builders (swapped, mismatched, no-image).
* `src/evaluate.py`: Answer normalization, consensus scoring, $\Delta\text{NLL}$, and bootstrap intervals.
* `notebooks/`: Step-by-step notebooks (1 to 7) covering feature caching, data prep, the 3 training stages, grounding evaluation, and token compression ablation.
