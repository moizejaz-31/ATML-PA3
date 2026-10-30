# Task 3: Test-Time Compute: Sampling, Selection, and Adaptive Inference

This task investigates inference-time compute scaling on mathematical and multimodal reasoning tasks, utilizing pre-generated rollouts (`ATML-PA3-Data`) from `Qwen2.5-3B-Instruct` (MATH) and `Qwen2.5-VL-3B-Instruct` (MathVista), along with a live sequential refinement experiment.

## Core Components
1. **Offline Evaluation & Repeated Sampling:**
   * Candidate budgets $N \in \{1, 2, 4, 8, 16, 32\}$ evaluated over 10 random rollout orderings.
   * Oracle coverage vs. achieved selection accuracy.
   * Selection rules: FirstAns, Self-Consistency, Mean Log-Probability, Self-Certainty, VisConf (MathVista), and Rank-Weighted Consensus (RWC).
2. **Quality Scoring & Signal Analysis:**
   * Mean log-probability: $L_i = \frac{1}{T_i} \sum_t \log p_{i,t}(y_{i,t})$
   * Self-certainty: $K_i = \frac{1}{T_i} \sum_t D_{\text{KL}}(U \parallel p_{i,t})$
   * Visual engagement: $V_i = \frac{1}{T_i} \sum_t \frac{A_{i,t}^{\text{image}}}{A_{i,t}^{\text{image}} + A_{i,t}^{\text{prompt}} + A_{i,t}^{\text{generated}}}$
   * VisConf: $Z_i = V_i C_i$ with candidate-pooled 5th/95th percentile clipping.
   * Tokenwise fusion ablation ($Z_i^{\text{token}}$) vs. trajectory-level product.
3. **Live Sequential Refinement (50 MATH questions, 3,072-token budget):**
   * Single longer generation (capped at 3,072 new tokens).
   * Parallel sampling ($3 \times 1,024$ new tokens + self-consistency).
   * Sequential refinement (1,024 tokens initial + 2,048 tokens critique and revision).
4. **Adaptive Inference Policies:**
   * Stopping at $n \in \{4, 8, 16\}$ or fallback 32.
   * Agreement-based: Majority fraction $\ge \tau \in \{0.5, 0.625, 0.75, 0.875, 1.0\}$.
   * Quality-aware: Agreement + SC/RWC consensus + RWC margin $\ge \gamma \in \{0, 0.1, 0.25, 0.5\}$.
   * Tuning on 200 dev questions, held-out testing on 800 evaluation questions.

## Directory Structure
* `configs/`: Experiment configs for offline analysis, live refinement, and adaptive stopping.
* `data/`: Location for `ATML-PA3-Data` parquet files.
* `splits/`: Question split IDs (200 dev, 800 test, 50 live) generated with seed 6304.
* `src/data.py`: Parquet reader and table joining utilities.
* `src/splits.py`: Lexicographical sort and NumPy permutation splitter.
* `src/normalization.py`: `math-verify` answer normalization and MathVista parsing.
* `src/metrics.py`: Calculation of logp, self-certainty, visual engagement, and VisConf.
* `src/aggregation.py`: Candidate selection rules and Rank-Weighted Consensus.
* `src/fusion_ablation.py`: Tokenwise vs trajectory fusion analysis and covariance calculation.
* `src/live_generation.py`: Live generation harness for Qwen2.5-3B-Instruct FP16.
* `src/adaptive_stopping.py`: Dynamic stopping controllers (agreement and quality-aware).
* `src/evaluate.py`: Bootstrap resamplers, offline permutation runner, cost accounting.
* `notebooks/`: Numbered notebooks 1 to 6 covering the full experimental pipeline.
