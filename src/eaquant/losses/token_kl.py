"""Token-level KL objective used in the EAQuant paper experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_token_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target_mask: torch.Tensor,
    token_weights: torch.Tensor | None = None,
    sequence_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute teacher-to-student KL only at selected next-token positions.

    ``target_mask[..., t]`` selects the logit at ``t - 1`` that predicts token
    ``t``. Token weights take precedence over sequence weights, matching the
    implementation used for the reported OSTQuant and OmniQuant experiments.
    """

    teacher_logits = teacher_logits.to(student_logits.device)
    target_mask = target_mask.to(student_logits.device)
    shift_mask = target_mask[..., 1:].bool()
    zero = student_logits.sum() * 0.0

    if student_logits.shape[0] == 0 or not torch.any(shift_mask):
        return zero

    selected_student = student_logits[..., :-1, :][shift_mask]
    selected_teacher = teacher_logits[..., :-1, :][shift_mask]
    if selected_student.numel() == 0:
        return zero

    if token_weights is not None:
        shifted_weights = token_weights.to(
            student_logits.device,
            dtype=torch.float32,
        )[..., 1:]
        selected_weights = shifted_weights[shift_mask].clamp_min(0.0)
        per_token_kl = F.kl_div(
            F.log_softmax(selected_student.float(), dim=-1),
            F.softmax(selected_teacher.float(), dim=-1),
            reduction="none",
        ).sum(dim=-1)
        total_weight = selected_weights.sum()
        if float(total_weight.item()) <= 0:
            return per_token_kl.mean()
        return (per_token_kl * selected_weights).sum() / total_weight

    if sequence_weights is None:
        return F.kl_div(
            F.log_softmax(selected_student.float(), dim=-1),
            F.softmax(selected_teacher.float(), dim=-1),
            reduction="batchmean",
        )

    weights = sequence_weights.to(student_logits.device, dtype=torch.float32)
    total_loss = zero
    total_weight = torch.zeros((), device=student_logits.device)
    for row_idx in range(student_logits.shape[0]):
        row_mask = shift_mask[row_idx]
        if not torch.any(row_mask):
            continue
        student_row = student_logits[row_idx, :-1, :][row_mask]
        teacher_row = teacher_logits[row_idx, :-1, :][row_mask]
        row_loss = F.kl_div(
            F.log_softmax(student_row.float(), dim=-1),
            F.softmax(teacher_row.float(), dim=-1),
            reduction="batchmean",
        )
        row_weight = weights[row_idx] if row_idx < weights.shape[0] else 1.0
        total_loss = total_loss + row_loss * row_weight
        total_weight = total_weight + row_weight

    if float(total_weight.item()) <= 0:
        return zero
    return total_loss / total_weight
