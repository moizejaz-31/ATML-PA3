"""
Fast CPU checks for Task 1 on a tiny random Qwen3. Run from the repository root:
    python -m pytest task1_llm_finetuning/tests -q
"""
import json
import os
import torch
import pytest
from transformers import Qwen3Config, Qwen3ForCausalLM

from task1_llm_finetuning.src.conditions import apply_evaluation_state
from task1_llm_finetuning.src.l2sp import L2SPRegularizer
from task1_llm_finetuning.src.lora import apply_lora_to_model, export_adapter_factors, delta_from_factors
from task1_llm_finetuning.src.oplora import apply_oplora_to_model
from task1_llm_finetuning.src.diagnostics import compute_subspace_interference
from task1_llm_finetuning.src.train import chunked_next_token_loss, count_valid_tokens, summed_next_token_loss, train_task1_model

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def tiny_model():
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         max_position_embeddings=64, tie_word_embeddings=True)
    return Qwen3ForCausalLM(config).float()

class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, n):
        generator = torch.Generator().manual_seed(1)
        self.items = []
        for _ in range(n):
            ids = torch.randint(0, 64, (12,), generator=generator)
            labels = ids.clone()
            labels[:4] = -100
            self.items.append({"input_ids": ids, "attention_mask": torch.ones(12, dtype=torch.long), "labels": labels})
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        return self.items[i]

class TinyTokenizer:
    pad_token_id = 0

def base_config(**overrides):
    config = dict(method="lora", run_name="tiny", seed=6304, micro_batch_size=2, gradient_accumulation_steps=2,
                  max_updates=6, warmup_updates=1, learning_rate=1e-3, save_checkpoints=[2, 4, 6], eval_checkpoints=[6],
                  lora_r=4, lora_alpha=8)
    config.update(overrides)
    return config

@pytest.mark.parametrize("apply_adapters,kwargs", [(apply_lora_to_model, {}), (apply_oplora_to_model, {"k": 4})])
def test_every_target_is_wrapped_exactly_once(apply_adapters, kwargs):
    plain = tiny_model()
    expected = sum(4 * (m.in_features + m.out_features) for name, m in plain.named_modules() if name.split(".")[-1] in TARGETS)
    model = apply_adapters(tiny_model(), r=4, lora_alpha=8, **kwargs)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == expected
    assert isinstance(model.model.layers[0].self_attn.q_proj.base_layer, torch.nn.Linear)

def test_matching_seed_gives_identical_lora_and_oplora_starting_factors():
    # tiny_model() reseeds, so seed immediately before each adapter creation
    plain = tiny_model()
    torch.manual_seed(6304)
    lora = apply_lora_to_model(plain, r=4, lora_alpha=8)
    plain = tiny_model()
    torch.manual_seed(6304)
    oplora = apply_oplora_to_model(plain, r=4, lora_alpha=8, k=4)
    assert torch.equal(lora.model.layers[1].mlp.up_proj.lora_A, oplora.model.layers[1].mlp.up_proj.lora_A)

@pytest.mark.parametrize("apply_adapters,kwargs", [(apply_lora_to_model, {}), (apply_oplora_to_model, {"k": 4})])
def test_merged_backbone_matches_unmerged_adapters(apply_adapters, kwargs):
    model = apply_adapters(tiny_model(), r=4, lora_alpha=8, lora_dropout=0.0, **kwargs)
    for name, p in model.named_parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, std=0.05)
    model.eval()
    ids = torch.arange(10)[None]
    state = {"kind": "adapters", "adapters": export_adapter_factors(model)}
    merged = apply_evaluation_state(tiny_model().eval(), state)
    assert (model(ids).logits - merged(ids).logits).abs().max() < 1e-3
    assert (model(ids).logits - tiny_model().eval()(ids).logits).abs().max() > 1e-3

def test_oplora_update_avoids_protected_subspaces():
    model = apply_oplora_to_model(tiny_model(), r=4, lora_alpha=8, k=4)
    layer = model.model.layers[0].mlp.down_proj
    torch.nn.init.normal_(layer.lora_B, std=0.1)
    delta = delta_from_factors(export_adapter_factors(model)["model.layers.0.mlp.down_proj"])
    assert delta.norm() > 0
    assert compute_subspace_interference(layer.Uk, delta) < 1e-8
    assert compute_subspace_interference(layer.Vk, delta.T) < 1e-8

def test_loss_is_normalized_over_all_tokens_in_the_effective_batch():
    logits = torch.randn(4, 6, 8)
    labels = torch.randint(0, 8, (4, 6))
    labels[0, :4] = -100
    per_token = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 8), labels[:, 1:].reshape(-1), ignore_index=-100)
    sums = [summed_next_token_loss(logits[i:i + 2], labels[i:i + 2]) for i in (0, 2)]
    counts = [count_valid_tokens(labels[i:i + 2]) for i in (0, 2)]
    assert torch.isclose(sum(sums) / sum(counts), per_token)

def test_l2sp_counts_tied_parameters_once_and_is_not_averaged():
    model = tiny_model()
    reference = L2SPRegularizer(model, lambda_reg=1e-3)
    with torch.no_grad():
        model.lm_head.weight.add_(1.0)  # tied with the embeddings
    _, omega, scaled = reference.compute_penalty(model)
    assert omega == pytest.approx(0.5 * model.lm_head.weight.numel(), rel=1e-5)
    assert scaled == pytest.approx(1e-3 * omega)
    assert reference.compute_displacement(model) == pytest.approx(model.lm_head.weight.numel() ** 0.5, rel=1e-5)

def test_chunked_loss_matches_full_logits_loss_and_gradients():
    model = tiny_model()
    ids = torch.randint(0, 64, (2, 12))
    labels = ids.clone()
    labels[0, :5] = -100
    labels[1, 9:] = -100
    full = summed_next_token_loss(model(input_ids=ids).logits, labels)
    full.backward()
    expected = model.lm_head.weight.grad.clone()
    model.zero_grad()
    chunked = chunked_next_token_loss(model.model(input_ids=ids).last_hidden_state, labels, model.lm_head, chunk=3)
    chunked.backward()
    assert torch.isclose(full, chunked, rtol=1e-5)
    assert torch.allclose(expected, model.lm_head.weight.grad, atol=1e-5)

def test_direct_l2sp_gradient_matches_autograd():
    model = tiny_model()
    reference = L2SPRegularizer(model, lambda_reg=1e-3)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    penalty, omega, _ = reference.compute_penalty(model)
    penalty.backward()
    expected = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad()
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    direct_omega, scaled = reference.add_penalty_gradient(model, grad_scale=8.0)
    assert direct_omega == pytest.approx(omega, rel=1e-5) and scaled == pytest.approx(1e-3 * omega, rel=1e-5)
    for n, p in model.named_parameters():
        assert torch.allclose(p.grad, 8.0 * expected[n], atol=1e-7)

def adapter_model():
    plain = tiny_model()
    torch.manual_seed(6304)
    return apply_lora_to_model(plain, r=4, lora_alpha=8)

def test_stopped_run_resumes_to_the_same_weights(tmp_path):
    train = lambda model, save, limit: train_task1_model(
        model, TinyDataset(24), base_config(), TinyTokenizer(),
        save_dir=str(save / "ckpt"), results_dir=str(save / "res"), time_limit_min=limit)

    straight = train(adapter_model(), tmp_path / "straight", None)
    assert straight["completed"]

    stopped = train(adapter_model(), tmp_path / "resumed", 0.0)  # stops after the first update
    assert not stopped["completed"]
    resumed = train(adapter_model(), tmp_path / "resumed", None)
    assert resumed["completed"]

    with open(tmp_path / "resumed" / "res" / "tiny" / "history.json") as f:
        assert json.load(f)["update"] == list(range(1, 7))
    for path in ["straight", "resumed"]:
        assert os.path.exists(tmp_path / path / "ckpt" / "tiny" / "adapters_update_6.pt")
        assert not list((tmp_path / path / "ckpt" / "tiny").glob("recovery_*"))
    a = torch.load(tmp_path / "straight" / "ckpt" / "tiny" / "adapters_update_6.pt", weights_only=True)["adapters"]
    b = torch.load(tmp_path / "resumed" / "ckpt" / "tiny" / "adapters_update_6.pt", weights_only=True)["adapters"]
    for name in a:
        # Exact on the CPU; the tolerance only allows for floating-point noise on a GPU
        assert torch.allclose(a[name]["A"], b[name]["A"], atol=1e-5) and torch.allclose(a[name]["B"], b[name]["B"], atol=1e-5)

def test_too_little_data_is_rejected_before_training(tmp_path):
    with pytest.raises(ValueError, match="need 24 examples"):
        train_task1_model(adapter_model(), TinyDataset(10), base_config(), TinyTokenizer(),
                          save_dir=str(tmp_path / "c"), results_dir=str(tmp_path / "r"))
