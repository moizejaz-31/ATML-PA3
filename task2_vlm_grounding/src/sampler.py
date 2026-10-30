"""
Curriculum stage mixture sampler for Task 2.
- Stage 2: 50% presence pairs, 25% counting pairs, 25% caption replay.
- Stage 3: 70% natural VQAv2, 10% presence, 10% counting, 10% caption replay.
Each draw contributes two examples (complete pair, or 2 caption examples).
"""
import numpy as np

class CurriculumSampler:
    def __init__(self, stage: int, seed: int = 6304):
        self.stage = stage
        self.rng = np.random.default_rng(seed)
        if stage == 2:
            self.probs = {"presence": 0.50, "counting": 0.25, "caption": 0.25}
        elif stage == 3:
            self.probs = {"vqa": 0.70, "presence": 0.10, "counting": 0.10, "caption": 0.10}
        else:
            self.probs = {"caption": 1.0}

    def sample_task_type(self) -> str:
        keys = list(self.probs.keys())
        values = list(self.probs.values())
        return self.rng.choice(keys, p=values)
