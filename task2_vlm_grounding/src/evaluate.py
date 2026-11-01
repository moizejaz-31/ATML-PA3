"""
Development and final evaluation for Task 2:
- Teacher-forced caption loss and reference-answer NLL (termination excluded) under image conditions.
- Greedy generation (16 tokens, batch 8) under four image conditions for both final-test suites.
- Normalized exact match, local VQA soft consensus accuracy, pair diagnostics, empty and identical answers.
- 95% intervals from 4,000 bootstrap resamples of complete pairs, matching pair IDs across conditions.
"""
import json
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .controls import CONDITIONS, conditioned_image, mismatched_mapping
from .data import normalize_vqa_answer

SEED = 6304
BOOTSTRAP_RESAMPLES = 4000

def compute_vqa_soft_accuracy(prediction: str, answers: List[str]) -> float:
    """s(a) = (1/10) sum_i min(1, (1/3) sum_{j != i} 1[a = a_j]), on normalized answers."""
    pred = normalize_vqa_answer(prediction)
    refs = [normalize_vqa_answer(a) for a in answers]
    matches = [r == pred for r in refs]
    total = sum(matches)
    return float(np.mean([min(1.0, (total - int(m)) / 3.0) for m in matches]))

def example_weighted_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy averaged within each example over supervised tokens, then across examples."""
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    token_losses = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), ignore_index=-100, reduction="none").view(shift_labels.shape)
    mask = (shift_labels != -100).float()
    return ((token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)).mean()

@torch.no_grad()
def teacher_forced_nll(model, encoder, store, examples: List[Dict[str, Any]], exclude_termination: bool, batch_size: int = 8) -> np.ndarray:
    """Per-example mean token NLL. Examples carry image=None for no-image inputs."""
    model.eval()
    device = next(model.connector.parameters()).device
    use_amp = device.type == "cuda"
    values = []
    for first in range(0, len(examples), batch_size):
        batch = examples[first:first + batch_size]
        encoded = [encoder.encode(e) for e in batch]
        features = [store.get(e["image"]) if e["image"] is not None else None for e in batch]
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            logits, labels = model(encoded, features)
        shift_logits, shift_labels = logits[:, :-1].float(), labels[:, 1:]
        losses = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), ignore_index=-100, reduction="none").view(shift_labels.shape)
        mask = shift_labels != -100
        if exclude_termination:
            for row in range(mask.shape[0]):
                mask[row, mask[row].nonzero()[-1]] = False
        values.extend(((losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).cpu().tolist())
    return np.array(values)

def suite_examples(pairs: List[Dict[str, Any]], condition: str, mapping) -> List[Dict[str, Any]]:
    """Two question sides per pair, with the original references and the condition's image."""
    examples = []
    for index, pair in enumerate(pairs):
        for side in ("a", "b"):
            other = "b" if side == "a" else "a"
            examples.append({"pair_id": index, "side": side, "question": pair["question"], "target": pair[side]["target"], "answers": pair[side]["answers"], "partner_target": pair[other]["target"],
                             "type": pair.get("answer_type") or pair["type"], "image": conditioned_image(pair, index, side, condition, mapping), "original_image": pair[side]["image"]})
    return examples

def development_metrics(model, encoder, store, manifest: Dict[str, Any], batch_size: int = 8) -> Dict[str, float]:
    """Caption reference loss, and reference-answer NLL under correct, swap, and no-image inputs."""
    captions = [{"image": x["image"], "question": None, "target": x["caption"]} for x in manifest["caption_dev"]]
    metrics = {"caption_loss": float(teacher_forced_nll(model, encoder, store, captions, False, batch_size).mean())}
    for suite, pairs in [("grounding", manifest["grounding_dev"]), ("natural", manifest["vqa_dev"])]:
        mapping = mismatched_mapping(pairs)
        for condition in ["correct", "swap", "no_image"]:
            metrics[f"{suite}_nll_{condition}"] = float(teacher_forced_nll(model, encoder, store, suite_examples(pairs, condition, mapping), True, batch_size).mean())
        metrics[f"{suite}_swap_penalty"] = metrics[f"{suite}_nll_swap"] - metrics[f"{suite}_nll_correct"]
    return metrics

@torch.no_grad()
def generate_suite(model, encoder, store, tokenizer, examples: List[Dict[str, Any]], batch_size: int = 8) -> pd.DataFrame:
    """Generate and score every example of one suite and condition."""
    model.eval()
    device = next(model.connector.parameters()).device
    use_amp = device.type == "cuda"
    rows = []
    start = time.perf_counter()
    for first in range(0, len(examples), batch_size):
        batch = examples[first:first + batch_size]
        encoded = [encoder.encode({**e, "target": None}) for e in batch]
        features = [store.get(e["image"]) if e["image"] is not None else None for e in batch]
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            predictions = model.generate_answers(encoded, features, tokenizer)
        for example, prediction in zip(batch, predictions):
            rows.append({**{k: example[k] for k in ["pair_id", "side", "question", "target", "partner_target", "type", "image", "original_image"]},
                         "prediction": prediction, "normalized": normalize_vqa_answer(prediction), "references": json.dumps(example["answers"])})
    elapsed = time.perf_counter() - start
    frame = pd.DataFrame(rows)
    frame["seconds_per_answer"] = elapsed / max(len(frame), 1)
    return frame

def score_frame(frame: pd.DataFrame, suite: str) -> pd.DataFrame:
    """Adds per-example scores: exact match for grounding, soft consensus for natural VQA."""
    frame = frame.copy()
    canonical = frame["target"].map(normalize_vqa_answer)
    frame["exact"] = (frame["normalized"] == canonical).astype(float)
    frame["follows_partner"] = (frame["normalized"] == frame["partner_target"].map(normalize_vqa_answer)).astype(float)
    frame["empty"] = (frame["normalized"] == "").astype(float)
    if suite == "natural":
        frame["score"] = [compute_vqa_soft_accuracy(p, json.loads(r)) for p, r in zip(frame["prediction"], frame["references"])]
    else:
        frame["score"] = frame["exact"]
    return frame

def pair_level(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per pair: mean score and NLL, both predictions canonical, and identical predictions."""
    groups = frame.groupby("pair_id")
    table = pd.DataFrame({"score": groups["score"].mean(), "nll": groups["nll"].mean(), "pair_correct": groups["exact"].min(),
                          "identical": groups["normalized"].agg(lambda x: float(x.iloc[0] == x.iloc[1])), "empty": groups["empty"].mean(), "follows_partner": groups["follows_partner"].mean()})
    return table

def bootstrap_interval(values: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = SEED) -> Dict[str, float]:
    """Percentile interval over complete pairs. Differences use the same sampled pairs because rows align."""
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "ci_low": float(np.percentile(means, 2.5)), "ci_high": float(np.percentile(means, 97.5))}

def summarize_final(frames: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """Suite and condition tables, per-type tables, and correct-image advantages with pair-bootstrap intervals."""
    summary, by_type, advantages = [], [], []
    for suite, conditions in frames.items():
        pairs = {c: pair_level(f) for c, f in conditions.items()}
        for condition, table in pairs.items():
            row = {"suite": suite, "condition": condition, "pairs": len(table), "seconds_per_answer": float(conditions[condition]["seconds_per_answer"].iloc[0]), "reference_nll": float(table["nll"].mean())}
            for metric in ["score", "pair_correct", "identical", "empty", "follows_partner"]:
                interval = bootstrap_interval(table[metric].to_numpy())
                row.update({metric: interval["mean"] * 100, f"{metric}_low": interval["ci_low"] * 100, f"{metric}_high": interval["ci_high"] * 100})
            summary.append(row)
            types = conditions[condition].groupby("type")["score"].mean().mul(100)
            by_type += [{"suite": suite, "condition": condition, "type": t, "score": s} for t, s in types.items()]
        for other in ["swap", "mismatched", "no_image"]:
            accuracy = bootstrap_interval((pairs["correct"]["score"] - pairs[other]["score"]).to_numpy() * 100)
            nll = bootstrap_interval((pairs[other]["nll"] - pairs["correct"]["nll"]).to_numpy())
            advantages.append({"suite": suite, "versus": other, "accuracy_advantage_pp": accuracy["mean"], "accuracy_low": accuracy["ci_low"], "accuracy_high": accuracy["ci_high"],
                               "nll_advantage": nll["mean"], "nll_low": nll["ci_low"], "nll_high": nll["ci_high"]})
    return {"summary": pd.DataFrame(summary), "by_type": pd.DataFrame(by_type), "advantages": pd.DataFrame(advantages)}

def run_final_evaluation(model, encoder, store, tokenizer, manifest: Dict[str, Any], output_dir: str) -> Dict[str, pd.DataFrame]:
    """Both final-test suites under all four conditions. Raw generations and per-example metrics are saved."""
    os.makedirs(output_dir, exist_ok=True)
    frames: Dict[str, Dict[str, pd.DataFrame]] = {}
    for suite, pairs in [("grounding", manifest["grounding_test"]), ("natural", manifest["vqa_test"])]:
        mapping = mismatched_mapping(pairs)
        frames[suite] = {}
        for condition in CONDITIONS:
            examples = suite_examples(pairs, condition, mapping)
            frame = generate_suite(model, encoder, store, tokenizer, examples)
            frame["nll"] = teacher_forced_nll(model, encoder, store, examples, True)
            frame = score_frame(frame, suite)
            frame.to_csv(os.path.join(output_dir, f"final_{suite}_{condition}.csv"), index=False)
            frames[suite][condition] = frame
            print(f"[Final] {suite} {condition}: score {frame['score'].mean() * 100:.2f}%", flush=True)
    tables = summarize_final(frames)
    for name, table in tables.items():
        table.to_csv(os.path.join(output_dir, f"final_{name}.csv"), index=False)
    return tables
