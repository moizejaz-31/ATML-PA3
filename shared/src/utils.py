import time
import torch

class Timer:
    """Context manager for timing code execution."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start

def get_gpu_memory():
    """Return current and max allocated GPU memory in MB."""
    if not torch.cuda.is_available():
        return {"current_mb": 0.0, "peak_mb": 0.0}
    return {
        "current_mb": torch.cuda.memory_allocated() / (1024 ** 2),
        "peak_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
    }

def count_trainable_parameters(model):
    """Count trainable vs total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "fraction": trainable / max(total, 1)}
