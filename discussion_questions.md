# EE-5102 / CS-6304: PA3 Research & Discussion Questions

This document compiles the 14 core research questions and synthesis guidelines prescribed in the assignment manual, organized across the three tasks.

---

## Task 1: Large Language Model Fine-Tuning

### Question 1: Specialization and Retention
> *How does mathematical fine-tuning affect the balance between mathematical performance and other capabilities, and how does this balance vary across methods and training stages?*

### Question 2: Restrictions on Adaptation
> *How do proximity regularization, adapter rank, and orthogonal protection affect the trade-off between adaptation and retention?*

### Question 3: Parameter Changes and Behavior
> *What relationships exist between the magnitude and geometry of parameter updates, their activation on different inputs, and changes in model behavior?*

### Question 4: Performance and Computational Cost
> *How do the methods compare in the balance they achieve between mathematical performance, retention, and computational cost?*

### Question 5: Behavior Beyond Benchmark Scores
> *How does mathematical fine-tuning affect responses to unrelated questions, and to what extent do benchmark scores reflect these behavioral changes?*

---

## Task 2: Vision-Language Models and Visual Grounding

### Question 1: Learning a Visual Interface
> *How does the training curriculum affect the model’s ability to describe images and answer visual questions? What carries over or changes between stages?*

### Question 2: Grounding and Generalization
> *To what extent do answers depend correctly on visual evidence, and how does this vary between annotation-derived and natural questions?*

### Question 3: Measuring Image Use
> *How consistent are the conclusions suggested by reference likelihood, generated answers, pair-level metrics, and qualitative behavior?*

### Question 4: Capabilities and Cost
> *What capabilities and limitations characterize this VLM, and what computational costs accompany its training and inference?*

---

## Task 3: Test-Time Compute: Sampling, Selection, and Adaptive Inference

### Question 1: Exploration and Selection
> *What limits the benefit of repeated sampling, and how does the balance between candidate coverage and selection quality change with compute?*

### Question 2: Evidence and Consensus
> *How do confidence, visual engagement, and answer agreement contribute to selecting a final answer, and what explains their successes or failures?*

### Question 3: Breadth and Refinement
> *How does spending computation on independent alternatives compare with a longer response or sequential reconsideration?*

### Question 4: Adaptive Allocation
> *When is another rollout useful, and how effectively can the observed evidence guide stopping without access to correctness?*

### Question 5: Compute and Reliability
> *Which strategies achieve useful accuracy–cost trade-offs, and what do the observed failures reveal about their reliability and limitations?*

