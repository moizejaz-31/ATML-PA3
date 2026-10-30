import os
import random
import numpy as np
import torch

DEFAULT_SEED = 6304
SEED = DEFAULT_SEED

def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True):
    """
    Set random seeds for python random, numpy, and torch across CPU and CUDA.
    
    Args:
        seed: Random integer seed (default: 6304).
        deterministic: If True, configures CuDNN for deterministic reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    print(f"[Seed] Fixed random seed to {seed} (deterministic={deterministic})")
