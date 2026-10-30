"""
Curriculum mixture replay sampler for Task 2.
- Stage 2 (Visual Instruction): 50% presence pairs, 25% counting pairs, 25% caption replay.
- Stage 3 (Natural VQA Specialization): 70% natural VQAv2, 10% presence, 10% counting, 10% caption replay.
Each draw contributes two examples: a complete complementary pair, or two caption examples.
"""
import numpy as np
from typing import Dict, Any, List, Tuple

class CurriculumSampler:
    def __init__(
        self,
        stage: int,
        presence_pairs: List[Dict[str, Any]] = None,
        counting_pairs: List[Dict[str, Any]] = None,
        caption_examples: List[Dict[str, Any]] = None,
        vqa_pairs: List[Dict[str, Any]] = None,
        seed: int = 6304,
    ):
        self.stage = stage
        self.rng = np.random.default_rng(seed)
        self.presence = presence_pairs or []
        self.counting = counting_pairs or []
        self.captions = caption_examples or []
        self.vqa = vqa_pairs or []

        if stage == 2:
            self.categories = ["presence", "counting", "caption"]
            self.probs = [0.50, 0.25, 0.25]
        elif stage == 3:
            self.categories = ["vqa", "presence", "counting", "caption"]
            self.probs = [0.70, 0.10, 0.10, 0.10]
        else:
            raise ValueError(f"CurriculumSampler only configured for stages 2 and 3, got stage {stage}")

    def draw_pair_or_replay(self) -> List[Dict[str, Any]]:
        """
        Draws one curriculum item yielding exactly two training examples.
        Preserves complete question pairs.
        """
        cat = self.rng.choice(self.categories, p=self.probs)

        if cat == "presence" and self.presence:
            idx = self.rng.integers(len(self.presence))
            pair = self.presence[idx]
            q = pair["question"]
            return [
                {"image_id": pair["image_a"]["image_id"], "prompt": f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n", "target": pair["image_a"]["target"]},
                {"image_id": pair["image_b"]["image_id"], "prompt": f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n", "target": pair["image_b"]["target"]},
            ]
        elif cat == "counting" and self.counting:
            idx = self.rng.integers(len(self.counting))
            pair = self.counting[idx]
            q = pair["question"]
            return [
                {"image_id": pair["image_a"]["image_id"], "prompt": f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n", "target": pair["image_a"]["target"]},
                {"image_id": pair["image_b"]["image_id"], "prompt": f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n", "target": pair["image_b"]["target"]},
            ]
        elif cat == "vqa" and self.vqa:
            idx = self.rng.integers(len(self.vqa))
            pair = self.vqa[idx]
            q = pair["question"]
            return [
                {"image_id": pair["image_a"]["image_id"], "prompt": f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n", "target": pair["image_a"]["target"]},
                {"image_id": pair["image_b"]["image_id"], "prompt": f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n", "target": pair["image_b"]["target"]},
            ]
        else:
            # Caption replay (draw 2 captions)
            if len(self.captions) >= 2:
                idxs = self.rng.choice(len(self.captions), size=2, replace=False)
                exs = []
                for i in idxs:
                    cap = self.captions[i]
                    prompt = "<|im_start|>user\nProvide a description for this image.<|im_end|>\n<|im_start|>assistant\n"
                    exs.append({"image_id": cap["image_id"], "prompt": prompt, "target": cap["caption"]})
                return exs
            elif len(self.captions) == 1:
                cap = self.captions[0]
                prompt = "<|im_start|>user\nProvide a description for this image.<|im_end|>\n<|im_start|>assistant\n"
                return [{"image_id": cap["image_id"], "prompt": prompt, "target": cap["caption"]}] * 2
            else:
                return []
