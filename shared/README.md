# Shared Utilities and Protocols

This directory contains cross-task utilities, common mathematical metrics, reproducibility harnesses, plotting functions, and runtime helpers shared across Tasks 1, 2, and 3.

## Directory Structure
* `configs/`: Global default configuration parameters (e.g. course seed `6304`, device flags).
* `src/seed.py`: Deterministic seed setting across Python, NumPy, PyTorch, and CUDA.
* `src/metrics.py`: Paired bootstrap resampling, confidence interval estimation, exact match, and accuracy routines.
* `src/plotting.py`: Publication-ready styling (NeurIPS compatible) for loss curves, Pareto trade-offs, and distributions.
* `src/diagrams.py`: Architecture and pipeline diagram utilities.
* `src/utils.py`: GPU memory monitoring, wall-clock timing, and checkpoint save/load helpers.
