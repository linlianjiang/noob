"""Test-time temporal prototype alignment (paper Sec. III-C)."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch import Tensor


class TemporalPrototypeMemory(nn.Module):
    """Observability-filtered class prototypes, smoothed over frames.

    Implements Eq. 11 (EMA update) and Eq. 12 (prototype-aligned target).

    ``beta`` is the weight of the *new* evidence, so the paper's "EMA decay
    0.99" is ``beta=0.01`` here.

    Args:
        num_classes (int): number of segmentation classes ``C``.
        feat_dim (int): width of the feature the frozen classifier consumes.
        beta (float): weight of the new evidence in Eq. 11.
        tau_p (float): confidence threshold ``tau_p`` of Eq. 11.
        temperature (float): softmax temperature for Eq. 12; the paper's bare
            softmax is ``1.0``.
        ignore_index (int, optional): class excluded from the memory and from
            the alignment target.
    """

    def __init__(self,
                 num_classes: int,
                 feat_dim: int,
                 beta: float = 0.01,
                 tau_p: float = 0.6,
                 temperature: float = 1.0,
                 ignore_index: Optional[int] = None,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.beta = beta
        self.tau_p = tau_p
        self.temperature = temperature
        self.ignore_index = ignore_index
        self.eps = eps

        self.register_buffer('prototypes', torch.zeros(num_classes, feat_dim))
        self.register_buffer('initialized',
                             torch.zeros(num_classes, dtype=torch.bool))

    def reset(self) -> None:
        """Clear the memory (called when a new adaptation stream starts)."""
        self.prototypes.zero_()
        self.initialized.zero_()

    @torch.no_grad()
    def update(self, feats: Tensor, probs: Tensor, obs: Tensor) -> None:
        """EMA update of Eq. 11 from one frame of cell-level evidence.

        Args:
            feats (Tensor): (R, D) cell features ``f_r``.
            probs (Tensor): (R, C) cell predictions ``p_r``.
            obs (Tensor): (R,) observability ``o_r``.
        """
        conf, pred = probs.max(dim=1)
        weight = (conf >= self.tau_p).float() * obs * conf
        if self.ignore_index is not None:
            weight = weight * (pred != self.ignore_index).float()

        num = torch_scatter.scatter_add(
            feats * weight.unsqueeze(1), pred, dim=0,
            dim_size=self.num_classes)
        den = torch_scatter.scatter_add(
            weight, pred, dim=0, dim_size=self.num_classes)

        new = num / (den.unsqueeze(1) + self.eps)
        seen = den > self.eps

        ema = (1 - self.beta) * self.prototypes + self.beta * new
        # first sighting of a class: set outright, nothing to smooth against
        updated = torch.where(self.initialized.unsqueeze(1), ema, new)
        updated = F.normalize(updated, dim=1, eps=self.eps)

        self.prototypes[seen] = updated[seen]
        self.initialized |= seen

    def valid_mask(self) -> Tensor:
        mask = self.initialized.clone()
        if self.ignore_index is not None:
            mask[self.ignore_index] = False
        return mask

    def align(self, feats: Tensor) -> Optional[Tensor]:
        """Prototype-aligned target ``q_r`` of Eq. 12; ``None`` until seeded."""
        valid = self.valid_mask()
        if int(valid.sum()) == 0:
            return None
        f_bar = F.normalize(feats, dim=1, eps=self.eps)
        m_bar = F.normalize(self.prototypes, dim=1, eps=self.eps)
        logits = (f_bar @ m_bar.t()) / self.temperature
        logits = logits.masked_fill(~valid.unsqueeze(0), float('-inf'))
        return torch.softmax(logits, dim=1)


def prototype_alignment_loss(q: Tensor,
                             p: Tensor,
                             obs: Tensor,
                             reduction: str = 'weighted_mean',
                             eps: float = 1e-8) -> Tensor:
    """``L_TTA = sum_r o_r w_r D_KL(q_r || p_r)`` (Eqs. 13-14).

    Args:
        q (Tensor): (R, C) prototype-aligned target.
        p (Tensor): (R, C) model prediction, must carry gradient.
        obs (Tensor): (R,) observability gate ``o_r``.
        reduction (str): ``'sum'`` is the literal Eq. 14; ``'weighted_mean'``
            (used by the configs) normalises by ``sum_r o_r w_r`` so the step
            size does not depend on the frame's cell count.
    """
    q_max, q_arg = q.max(dim=1)
    p_at_q = p.gather(1, q_arg.unsqueeze(1)).squeeze(1)
    w = q_max * (1.0 - p_at_q)  # Eq. 13

    log_q = torch.log(q.clamp_min(eps))
    log_p = torch.log(p.clamp_min(eps))
    kl = (q * (log_q - log_p)).sum(dim=1)

    weight = obs * w
    if reduction == 'sum':
        return (weight * kl).sum()
    if reduction == 'weighted_mean':
        return (weight * kl).sum() / weight.sum().clamp_min(eps)
    if reduction == 'mean':
        return (weight * kl).mean()
    raise ValueError(f'unknown reduction: {reduction}')
