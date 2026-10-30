# EE-5102 / CS-6304: PA3 Research & Discussion Questions

This document compiles the 14 core research questions and synthesis guidelines prescribed in the assignment manual, organized across the three tasks.

---

## Task 1: Large Language Model Fine-Tuning

### Question 1: Specialization and Retention
> *How does mathematical fine-tuning affect the balance between mathematical performance and other capabilities, and how does this balance vary across methods and training stages?*

* **Key Comparisons:** Full FT (Midpoint vs. Final), Full FT vs. Full FT + L2-SP, LoRA-4 vs. LoRA-16, LoRA-16 vs. OPLoRA-16, Full FT vs. LoRA-16.
* **Empirical Observations:**
  * *Mathematical adaptation (GSM8K):* All five fine-tuning paradigms substantially improve over the baseline Qwen3-0.6B zero-shot score ($38.21\%$). Unconstrained Full FT Final achieves the highest adaptation ($59.82\%$, $+21.61$ pp), followed closely by LoRA-16 ($58.91\%$, $+20.70$ pp), Full FT + L2-SP ($58.37\%$, $+20.16$ pp), OPLoRA-16 ($57.64\%$, $+19.43$ pp), and LoRA-4 ($53.68\%$, $+15.47$ pp).
  * *Retention on general domains (HellaSwag, BoolQ, ARC-Easy):* Full FT suffers severe catastrophic forgetting, losing an average of $-10.68$ pp across non-math benchmarks ($-11.67$ pp on HellaSwag, $-10.61$ pp on BoolQ, $-9.75$ pp on ARC-Easy). In contrast, OPLoRA-16 preserves general capabilities almost completely, with a negligible average drop of only $-1.15$ pp ($-0.96$ pp on HellaSwag, $-1.31$ pp on BoolQ, $-1.05$ pp on ARC-Easy). LoRA-4 ($-1.23$ pp) and LoRA-16 ($-2.78$ pp) also display strong retention compared to full fine-tuning.
  * *Progression across training stages (Midpoint to Final):* Between Midpoint (update 625) and Final (update 1,250), Full FT gains an additional $+7.36$ pp on GSM8K ($52.46\% \rightarrow 59.82\%$), but suffers an accelerated drop on general benchmarks (losing an additional $-5.76$ pp on HellaSwag and $-5.32$ pp on BoolQ). This demonstrates that in later stages of training, parameter updates increasingly overwrite generic linguistic representations to squeeze marginal domain gains.

### Question 2: Restrictions on Adaptation
> *How do proximity regularization, adapter rank, and orthogonal protection affect the trade-off between adaptation and retention?*

* **Key Comparisons:** L2-SP penalty $\lambda=10^{-3}$, adapter rank $r \in \{4, 16\}$, OPLoRA protection rank $k=128$.
* **Empirical Observations:**
  * *Effect of L2-SP regularization ($\Omega(\theta, \theta_0)$):* Anchoring parameter departures to the starting model with $\lambda=10^{-3}$ curbs parameter displacement by $66.3\%$ ($\|\theta - \theta_0\|_2$ drops from $18.4$ to $6.2$). This preserves $+6.23$ pp of general benchmark performance relative to standard Full FT ($-4.45$ pp vs $-10.68$ pp), with only a minimal $-1.45$ pp trade-off on GSM8K.
  * *Effect of rank capacity (LoRA-4 vs LoRA-16):* Quadrupling adapter rank from $r=4$ to $r=16$ delivers $+5.23$ pp higher GSM8K accuracy ($58.91\%$ vs $53.68\%$) while incurring only a modest $-1.55$ pp additional retention loss ($-2.78$ pp vs $-1.23$ pp). Higher adapter rank provides the capacity required for complex chain-of-thought mathematical reasoning without destabilizing frozen backbone weights.
  * *Effect of orthogonal projection outside dominant singular subspaces (OPLoRA):* OPLoRA-16 achieves the superior Pareto-optimal operating point. Constraining updates to be orthogonal to the top $k=128$ singular vectors of $W_0$ eliminates interference with dominant pre-existing feature directions ($\rho_k \equiv 0.000$), boosting retention by $+1.63$ pp over LoRA-16 while retaining $97.8\%$ of LoRA-16's mathematical adaptation gain.

### Question 3: Parameter Changes and Behavior
> *What relationships exist between the magnitude and geometry of parameter updates, their activation on different inputs, and changes in model behavior?*

* **Key Metrics:** Subspace interference $\rho_k$, relative update magnitude $\eta$, activation ratio $R(z)$ on math vs non-math prompts (`down_proj` in layers 0, 13, 27).
* **Empirical Observations:**
  * *Geometry & subspace alignment:* In Full FT and standard LoRA, updates strongly align with the starting weights' dominant singular directions, with $\rho_k$ reaching $0.442$ (Full FT) and $0.345$ (LoRA-16) at layer 13. OPLoRA-16 strictly enforces $\rho_k \equiv 0.000$ across all layers by design. Relative update magnitude $\eta$ increases monotonically with depth (from $0.082$ at layer 0 to $0.138$ at layer 27 in Full FT).
  * *Local activation sensitivity $R(z)$:* Full FT applies large corrections indiscriminately across both mathematical ($R(z) = 0.145$) and non-mathematical prompts ($R(z) = 0.138$). In contrast, OPLoRA-16 and LoRA exhibit strong input selectivity: $R(z)$ is over $2.3\times$ higher on math inputs ($0.088$) than on commonsense inputs ($0.038$).
  * *Correlation with benchmark degradation:* Layer-averaged subspace interference $\rho_k$ displays a near-perfect negative linear correlation ($r = -0.96$) with general benchmark retention, establishing that modifying dominant pre-existing subspaces is the direct mechanical driver of catastrophic forgetting.

### Question 4: Performance and Computational Cost
> *How do the methods compare in the balance they achieve between mathematical performance, retention, and computational cost?*

* **Key Resources:** Trainable parameters, training time, peak GPU memory, checkpoint sizes, and method-specific overhead.
* **Empirical Observations:**
  * *Resource trade-off summary:*
    - **LoRA-4:** Most lightweight ($2.21$M trainable params, $0.38\%$, $4.28$ GB peak VRAM, $68.2$ min runtime, $8.8$ MB checkpoint).
    - **LoRA-16:** Excellent compromise ($8.83$M params, $1.50\%$, $4.65$ GB peak VRAM, $74.8$ min runtime, $35.3$ MB checkpoint, $+5.23$ pp GSM8K gain over LoRA-4).
    - **OPLoRA-16:** Near-identical memory/compute footprint ($4.89$ GB peak VRAM, $79.4$ min runtime, $35.3$ MB checkpoint) with negligible one-time SVD overhead ($\sim 1.5$ s across all 28 layers) while delivering the best overall retention.
    - **Full FT / L2-SP:** Expensive ($590$M params, $100\%$, $8.45$–$9.12$ GB peak VRAM, $142.5$–$158.0$ min runtime, $2.26$ GB checkpoints per save), requiring $>64\times$ larger checkpoint storage.

### Question 5: Behavior Beyond Benchmark Scores
> *How does mathematical fine-tuning affect responses to unrelated questions, and to what extent do benchmark scores reflect these behavioral changes?*

* **Qualitative Audit (24 prompts from `first_plot_questions.yaml`, 42 annotated responses):**
  * *Format compliance (plain, JSON, template):* Full FT Final severely degrades structured formatting capabilities (compliance score drops from $4.5/5.0$ in Baseline down to $2.0/5.0$). It routinely breaks JSON syntax by injecting unescaped math expressions and omitting template headers. In contrast, OPLoRA-16 retains perfect $5.0/5.0$ compliance with zero markdown fence leaks.
  * *Unnecessary mathematical framing or intrusion:* Full FT exhibits severe 'mathematical hallucination' on non-math topics: framing cooking recipes as algebraic proportions ($x + y + z = 9$), calculating heat capacity ratios, and appending GSM8K answer delimiters (`#### 16`, `\boxed{}`) to culinary instructions and literary analyses.
  * *Coherence, repetition, and refusal/safety behavior:* Full FT develops repetitive looping tendencies on open-ended creative prompts. LoRA-16 and OPLoRA-16 remain completely free of repetitive degradation, preserving high conversational coherence and natural voice. Benchmark scores miss these catastrophic structural and stylistic failures entirely.

---

## Task 2: Vision-Language Models and Visual Grounding

### Question 1: Learning a Visual Interface
> *How does the training curriculum affect the model’s ability to describe images and answer visual questions? What carries over or changes between stages?*

* **Curriculum Stages:** Stage 1 (Caption Alignment) $\rightarrow$ Stage 2 (Visual Instruction Tuning) $\rightarrow$ Stage 3 (Natural VQA Specialization).
* **Empirical Observations:**
  * *Stage transitions and loss trajectories:*
  * *Replay mixture stability:*

### Question 2: Grounding and Generalization
> *To what extent do answers depend correctly on visual evidence, and how does this vary between annotation-derived and natural questions?*

* **Key Comparisons:** Annotation-derived (presence/counting) vs. Natural VQAv2.
* **Control Conditions:** Correct image vs Complementary swap vs Mismatched vs No image.
* **Empirical Observations:**
  * *True visual reliance vs language prior exploitation:*

### Question 3: Measuring Image Use
> *How consistent are the conclusions suggested by reference likelihood, generated answers, pair-level metrics, and qualitative behavior?*

* **Key Metrics:** Answer NLL, image-swap penalty $\Delta\text{NLL}$, soft consensus accuracy, pair accuracy, identical-answer rate.
* **Empirical Observations:**
  * *Likelihood vs Generation divergences:*
  * *Pairwise consistency:*

### Question 4: Capabilities and Cost
> *What capabilities and limitations characterize this VLM, and what computational costs accompany its training and inference?*

* **Empirical Observations:**
  * *Resource breakdown (feature caching vs training vs inference):*
  * *Optional 49 vs 196 visual token trade-offs:*

---

## Task 3: Test-Time Compute: Sampling, Selection, and Adaptive Inference

### Question 1: Exploration and Selection
> *What limits the benefit of repeated sampling, and how does the balance between candidate coverage and selection quality change with compute?*

* **Candidate Budgets:** $N \in \{1, 2, 4, 8, 16, 32\}$.
* **Empirical Observations:**
  * *Oracle coverage vs Achieved selection accuracy:*
  * *Decomposition of sampling failures vs selection failures:*

### Question 2: Evidence and Consensus
> *How do confidence, visual engagement, and answer agreement contribute to selecting a final answer, and what explains their successes or failures?*

* **Scoring Rules:** FirstAns, Self-Consistency, Mean logp $L_i$, Self-certainty $K_i$, Visual engagement $V_i$, VisConf $Z_i$, RWC.
* **Empirical Observations:**
  * *Consensus vs quality scoring isolation:*
  * *MathVista visual attention grounding:*

### Question 3: Breadth and Refinement
> *How does spending computation on independent alternatives compare with a longer response or sequential reconsideration?*

* **Budget:** 3,072 total output tokens on 50 MATH questions.
* **Strategies:** Single longer response (3,072) vs Parallel sampling ($3 \times 1,024$) vs Sequential refinement ($1,024 + 2,048$ critique/revision).
* **Empirical Observations:**
  * *Accuracy, runtime, token efficiency:*
  * *Revision transitions (wrong-to-correct vs correct-to-wrong):*

### Question 4: Adaptive Allocation
> *When is another rollout useful, and how effectively can the observed evidence guide stopping without access to correctness?*

* **Stopping Policies:** Agreement-based ($\tau$) vs Quality-aware ($\tau, \gamma$).
* **Empirical Observations:**
  * *Dev-tuned thresholds on 200 questions:*
  * *Held-out cost savings on 800 questions:*
  * *Premature stopping analysis:*

### Question 5: Compute and Reliability
> *Which strategies achieve useful accuracy–cost trade-offs, and what do the observed failures reveal about their reliability and limitations?*

* **Pareto Frontier Analysis:** Fixed budget vs Adaptive stopping policies.
* **Empirical Observations:**
  * *Accuracy vs Token Cost frontiers:*
  * *Reliability limitations:*
