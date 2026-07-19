"""Shared EAQuant loss functions used by all quantization backends."""

from .recovery_kl import option_distribution_kl, option_set_kl, row_target_log_scores
from .token_kl import masked_token_kl

__all__ = [
    "masked_token_kl",
    "option_distribution_kl",
    "option_set_kl",
    "row_target_log_scores",
]
