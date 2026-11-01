"""
Curriculum example streams for Task 2.
- Stage 1: two passes over the caption examples.
- Stage 2 (Visual Instruction): 50% presence pairs, 25% counting pairs, 25% caption replay.
- Stage 3 (Natural VQA Specialization): 70% natural VQAv2, 10% presence, 10% counting, 10% caption replay.
Each draw contributes two examples: a complete question pair, or two caption examples.
Draws are seeded with no batch or stage quotas. Stream state can be saved and restored.
"""
from typing import Any, Dict, List
import numpy as np

STAGE_MIXTURES = {
    2: {"presence": 0.50, "counting": 0.25, "caption": 0.25},
    3: {"natural": 0.70, "presence": 0.10, "counting": 0.10, "caption": 0.10},
}

def pair_examples(pair: Dict[str, Any], kind: str) -> List[Dict[str, Any]]:
    return [{"image": pair[side]["image"], "question": pair["question"], "target": pair[side]["target"], "kind": kind} for side in ("a", "b")]

def caption_example(caption: Dict[str, str]) -> Dict[str, Any]:
    return {"image": caption["image"], "question": None, "target": caption["caption"], "kind": "caption"}

class CaptionStream:
    """Stage 1: seeded shuffled passes over all caption examples."""
    def __init__(self, captions: List[Dict[str, str]], seed: int):
        self.captions = captions
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(captions))
        self.position = 0

    def next_examples(self, count: int) -> List[Dict[str, Any]]:
        examples = []
        while len(examples) < count:
            if self.position == len(self.order):
                self.order = self.rng.permutation(len(self.captions))
                self.position = 0
            examples.append(caption_example(self.captions[int(self.order[self.position])]))
            self.position += 1
        return examples

    def state(self) -> Dict[str, Any]:
        return {"rng": self.rng.bit_generator.state, "order": self.order.tolist(), "position": self.position}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng"]
        self.order = np.array(state["order"])
        self.position = state["position"]

class CurriculumSampler:
    """Stages 2 and 3: seeded draws from the stage mixture. Mixture counts are logged."""
    def __init__(self, stage: int, manifest: Dict[str, Any], seed: int):
        self.mixture = STAGE_MIXTURES[stage]
        self.rng = np.random.default_rng(seed)
        self.pools = {"presence": manifest["presence_pool"], "counting": manifest["counting_pool"], "natural": manifest["vqa_train"], "caption": manifest["caption_train"]}
        self.names = list(self.mixture)
        self.probabilities = [self.mixture[n] for n in self.names]
        self.counts = {n: 0 for n in self.names}

    def draw(self) -> List[Dict[str, Any]]:
        name = str(self.rng.choice(self.names, p=self.probabilities))
        self.counts[name] += 1
        pool = self.pools[name]
        if name == "caption":
            first, second = self.rng.choice(len(pool), size=2, replace=False)
            return [caption_example(pool[int(first)]), caption_example(pool[int(second)])]
        return pair_examples(pool[int(self.rng.integers(len(pool)))], name)

    def next_examples(self, count: int) -> List[Dict[str, Any]]:
        examples: List[Dict[str, Any]] = []
        while len(examples) < count:
            examples.extend(self.draw())
        return examples

    def state(self) -> Dict[str, Any]:
        return {"rng": self.rng.bit_generator.state, "counts": dict(self.counts)}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng"]
        self.counts = dict(state["counts"])
