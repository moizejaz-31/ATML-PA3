"""
Offline rollout order evaluation runner across candidate budgets:
Simulates nested prefixes over 10 random permutations of the 32 rollouts.
"""
from typing import List, Dict, Any
import numpy as np

def run_offline_budget_evaluation(question_data, budgets=[1, 2, 4, 8, 16, 32], num_permutations=10, seed=6304):
    """
    Simulate offline candidate selection across budgets and random orders.
    """
    pass
