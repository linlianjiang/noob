"""Observability-constrained dynamic prompt tuning (Sec. III-B), plus the
vanilla VPT adapter used as the Table-IV baseline.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of an (B, C, H, W) map."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class ObsPromptAdapter(nn.Module):
    """Input-conditioned prompt adapter with observability-gated residual.

    Implements Eqs. 8-10:

        C_i        = phi_i(LN(F_{i-1}))                                  (8)
        Delta_i(r) = Softmax(Q_r K_r^T / sqrt(d)) V_r                    (9)
        F_i(r)     = F_{i-1}(r) + o_r * Delta_i(r)                      (10)

    Cross-attention is strictly per-location (1x1 convs); all spatial structure
    enters through the observability gate. ``out_proj`` is zero-initialised, so
    a fresh adapter is the identity.

    Args:
        channels (int): channel width of the backbone stage input.
        embed_dims (int): bottleneck / attention width ``d``.
        prompt_size (int): number of prompt tokens ``p`` per location.
    """

    def __init__(self,
                 channels: int,
                 embed_dims: int = 24,
                 prompt_size: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.embed_dims = embed_dims
        self.prompt_size = prompt_size

        self.norm_prompt_in = LayerNorm2d(channels)
        self.prompt_gen = nn.Conv2d(channels, embed_dims, 1)  # phi_i, Eq. 8
        self.norm_prompt = LayerNorm2d(embed_dims)

        self.norm_q = LayerNorm2d(channels)
        self.q_proj = nn.Conv2d(channels, embed_dims, 1)
        self.k_proj = nn.Conv2d(embed_dims, prompt_size * embed_dims, 1)
        self.v_proj = nn.Conv2d(embed_dims, prompt_size * embed_dims, 1)
        self.out_proj = nn.Conv2d(embed_dims, channels, 1)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def residual(self, x: Tensor) -> Tensor:
        """The un-gated prompt-induced residual ``Delta_i`` of Eq. 9."""
        b, _, h, w = x.shape
        d, p = self.embed_dims, self.prompt_size

        prompt = self.prompt_gen(self.norm_prompt_in(x))  # Eq. 8
        prompt = self.norm_prompt(prompt)

        q = self.q_proj(self.norm_q(x))
        k = self.k_proj(prompt)
        v = self.v_proj(prompt)

        # (B, C, H, W) -> (B*H*W, tokens, d): one query, p keys/values per cell
        q = q.permute(0, 2, 3, 1).reshape(b * h * w, 1, d)
        k = k.permute(0, 2, 3, 1).reshape(b * h * w, p, d)
        v = v.permute(0, 2, 3, 1).reshape(b * h * w, p, d)

        attn = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(d), dim=-1)
        out = (attn @ v).reshape(b, h, w, d).permute(0, 3, 1, 2).contiguous()
        return self.out_proj(out)

    def forward(self, x: Tensor, obs: Tensor) -> Tensor:
        """Eq. 10. ``obs`` is (B, 1, H, W) at the resolution of ``x``."""
        return x + obs * self.residual(x)


class VPTAdapter(nn.Module):
    """Vanilla VPT baseline (Table IV): ``f~ = W [f ; p]``.

    ``M`` prompt tokens are concatenated to every location and projected back
    with an identity-initialised ``W``. Prompts are globally shared and
    spatially uniform.
    """

    def __init__(self, channels: int, num_tokens: int = 5) -> None:
        super().__init__()
        self.channels = channels
        self.num_tokens = num_tokens
        self.prompts = nn.Parameter(torch.zeros(num_tokens * channels))
        nn.init.trunc_normal_(self.prompts, std=0.02)
        self.proj = nn.Conv2d(channels * (1 + num_tokens), channels, 1)

        # identity init: pass f through, ignore the prompts at t=0
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            eye = torch.eye(channels).view(channels, channels, 1, 1)
            self.proj.weight[:, :channels] = eye

    def forward(self, x: Tensor, obs: Tensor = None) -> Tensor:
        b, _, h, w = x.shape
        p = self.prompts.view(1, -1, 1, 1).expand(b, -1, h, w)
        return self.proj(torch.cat((x, p), dim=1))
