"""
Vision-Language Model: frozen SigLIP features + trainable connector + Qwen3-0.6B.
Visual tokens are inserted inside the user message, before the question, preserving Qwen's chat structure.
No-image inputs omit the visual tokens and rebuild masks and positions.
"""
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from .connector import VLMConnector

SENTINEL = "\u0001VISUAL\u0001"
CAPTION_INSTRUCTION = "Describe this image."
END_OF_TURN = "<|im_end|>"
QUESTION_CAP = 80
CAPTION_TARGET_CAP = 64
ANSWER_TARGET_CAP = 16

class ExampleEncoder:
    """Tokenizes question and target with the manual's content-token caps, cutting at the visual position."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        template = tokenizer.apply_chat_template([{"role": "user", "content": SENTINEL + "QUESTION"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        head, tail = template.split(SENTINEL)
        before, after = tail.split("QUESTION")
        self.prefix_ids = tokenizer.encode(head, add_special_tokens=False)
        self.question_lead_ids = tokenizer.encode(before, add_special_tokens=False)
        self.assistant_ids = tokenizer.encode(after, add_special_tokens=False)
        self.end_of_turn_id = tokenizer.encode(END_OF_TURN, add_special_tokens=False)

    def encode(self, example: Dict[str, Any]) -> Dict[str, List[int]]:
        question = example["question"] if example["question"] is not None else CAPTION_INSTRUCTION
        question_ids = self.tokenizer.encode(question, add_special_tokens=False)[:QUESTION_CAP]
        cap = ANSWER_TARGET_CAP if example["question"] is not None else CAPTION_TARGET_CAP
        target_ids = self.tokenizer.encode(example["target"], add_special_tokens=False)[:cap] if example.get("target") else []
        return {"prefix": self.prefix_ids, "suffix": self.question_lead_ids + question_ids + self.assistant_ids, "target": target_ids + self.end_of_turn_id}

class SigLIPQwenVLM(nn.Module):
    def __init__(self, llm_backbone: nn.Module, connector: VLMConnector):
        super().__init__()
        self.llm = llm_backbone
        self.connector = connector

    def _embed(self, ids: List[int], device: torch.device) -> torch.Tensor:
        return self.llm.get_input_embeddings()(torch.tensor(ids, dtype=torch.long, device=device))

    def assemble(
        self,
        encoded: List[Dict[str, List[int]]],
        features: List[Optional[torch.Tensor]],
        include_target: bool = True,
        pad_side: str = "right",
    ) -> Dict[str, torch.Tensor]:
        """
        Build inputs_embeds, attention mask, positions, and labels. Visual tokens, prompts, and padding are masked.
        Features of None give a no-image input with text instructions unchanged.
        """
        device = next(self.connector.parameters()).device
        embed_dtype = self.llm.get_input_embeddings().weight.dtype
        pieces, labels = [], []
        for item, feature in zip(encoded, features):
            parts = [self._embed(item["prefix"], device)]
            label = [-100] * len(item["prefix"])
            if feature is not None:
                visual = self.connector(feature.to(device, dtype=torch.float32)).to(embed_dtype)
                parts.append(visual)
                label += [-100] * visual.shape[0]
            parts.append(self._embed(item["suffix"], device))
            label += [-100] * len(item["suffix"])
            if include_target:
                parts.append(self._embed(item["target"], device))
                label += item["target"]
            pieces.append(torch.cat(parts, dim=0))
            labels.append(label)

        longest = max(p.shape[0] for p in pieces)
        batch = len(pieces)
        dim = pieces[0].shape[1]
        padding = self._embed([self.llm.config.pad_token_id or 0], device)[0]
        inputs_embeds = padding.expand(batch, longest, dim).clone()
        attention_mask = torch.zeros((batch, longest), dtype=torch.long, device=device)
        position_ids = torch.zeros((batch, longest), dtype=torch.long, device=device)
        label_tensor = torch.full((batch, longest), -100, dtype=torch.long, device=device)
        for i, (piece, label) in enumerate(zip(pieces, labels)):
            length = piece.shape[0]
            start = longest - length if pad_side == "left" else 0
            inputs_embeds[i, start:start + length] = piece
            attention_mask[i, start:start + length] = 1
            position_ids[i, start:start + length] = torch.arange(length, device=device)
            label_tensor[i, start:start + length] = torch.tensor(label, dtype=torch.long, device=device)
        return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask, "position_ids": position_ids, "labels": label_tensor}

    def forward(self, encoded: List[Dict[str, List[int]]], features: List[Optional[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, labels) for teacher-forced targets."""
        batch = self.assemble(encoded, features)
        labels = batch.pop("labels")
        return self.llm(**batch).logits, labels

    @torch.no_grad()
    def generate_answers(self, encoded: List[Dict[str, List[int]]], features: List[Optional[torch.Tensor]], tokenizer, max_new_tokens: int = 16) -> List[str]:
        """Greedy generation with thinking disabled, using left padding so that batching cannot change answers."""
        batch = self.assemble(encoded, features, include_target=False, pad_side="left")
        batch.pop("labels")
        output = self.llm.generate(
            inputs_embeds=batch["inputs_embeds"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        return [text.strip() for text in tokenizer.batch_decode(output, skip_special_tokens=True)]
