"""Answer-recovery option-set objective used in EAQuant."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F


def row_target_log_scores(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    target_mask: torch.Tensor,
    normalize_by_length: bool = False,
) -> torch.Tensor:
    """Score each option row by its selected target-token log probability."""

    input_ids = input_ids.to(logits.device)
    target_mask = target_mask.to(logits.device)
    shift_mask = target_mask[..., 1:].bool()
    shift_input_ids = input_ids[..., 1:]
    shift_logits = logits[..., :-1, :]
    target_lengths = shift_mask.sum(dim=-1)
    row_scores = torch.zeros(
        shift_mask.shape[0],
        device=shift_logits.device,
        dtype=torch.float32,
    )
    if torch.any(shift_mask):
        selected_logits = shift_logits[shift_mask]
        selected_token_ids = shift_input_ids[shift_mask]
        selected_log_probs = F.log_softmax(
            selected_logits.float(),
            dim=-1,
        ).gather(
            dim=-1,
            index=selected_token_ids.unsqueeze(-1),
        ).squeeze(-1)
        row_indices = (
            torch.arange(shift_mask.shape[0], device=shift_mask.device)
            .unsqueeze(1)
            .expand_as(shift_mask)[shift_mask]
        )
        row_scores.scatter_add_(0, row_indices, selected_log_probs)
    if normalize_by_length:
        row_scores = row_scores / target_lengths.clamp_min(1).to(row_scores.dtype)
    return row_scores.masked_fill(target_lengths <= 0, float("-inf"))


def option_distribution_kl(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
) -> torch.Tensor:
    """KL between teacher and student distributions over one answer set."""

    return F.kl_div(
        F.log_softmax(student_scores.float(), dim=-1).unsqueeze(0),
        F.softmax(teacher_scores.float(), dim=-1).unsqueeze(0),
        reduction="batchmean",
    )


def option_set_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    input_ids: torch.Tensor,
    target_mask: torch.Tensor,
    group_sizes: Iterable[int] | torch.Tensor,
    normalize_by_length: bool = False,
) -> torch.Tensor:
    """Average recovery KL across independently scored answer-option sets."""

    if isinstance(group_sizes, torch.Tensor):
        sizes = [int(size) for size in group_sizes.tolist()]
    else:
        sizes = [int(size) for size in group_sizes]

    zero = student_logits.sum() * 0.0
    total_loss = zero
    total_groups = 0
    start = 0
    for size in sizes:
        end = start + max(size, 0)
        if size <= 0:
            start = end
            continue
        student_scores = row_target_log_scores(
            student_logits[start:end],
            input_ids[start:end],
            target_mask[start:end],
            normalize_by_length=normalize_by_length,
        )
        teacher_scores = row_target_log_scores(
            teacher_logits[start:end],
            input_ids[start:end],
            target_mask[start:end],
            normalize_by_length=normalize_by_length,
        )
        start = end
        if student_scores.numel() == 0 or teacher_scores.numel() == 0:
            continue
        if not torch.all(torch.isfinite(student_scores)) or not torch.all(
            torch.isfinite(teacher_scores)
        ):
            continue
        total_loss = total_loss + option_distribution_kl(
            student_scores,
            teacher_scores,
        )
        total_groups += 1

    if total_groups <= 0:
        return zero
    return total_loss / float(total_groups)
