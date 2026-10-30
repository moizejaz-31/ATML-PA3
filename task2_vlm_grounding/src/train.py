"""
Curriculum training engine for Task 2.
Implements example-weighted causal cross-entropy loss:
L = - (1 / |B|) sum_{i in B} (1 / |R_i|) sum_{t in R_i} log p(y_{i,t} | I_i, q_i, y_{i,<t})
"""
import torch
import torch.nn as nn

def compute_example_weighted_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Computes cross-entropy averaged within each example over valid response tokens,
    then averaged across the batch.
    """
    # Shift logits and labels for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    batch_size = shift_labels.shape[0]
    total_loss = 0.0
    valid_examples = 0
    
    loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)
    
    for i in range(batch_size):
        item_logits = shift_logits[i]
        item_labels = shift_labels[i]
        mask = (item_labels != ignore_index)
        n_tokens = mask.sum().item()
        if n_tokens > 0:
            token_losses = loss_fn(item_logits, item_labels)
            example_loss = token_losses[mask].mean()
            total_loss = total_loss + example_loss
            valid_examples += 1
            
    return total_loss / max(valid_examples, 1)
