"""Beam-termination and neighborhood-supported observability (Sec. III-A)."""

from typing import Dict, Sequence

import torch
import torch.nn.functional as F
import torch_scatter
from torch import Tensor


def _masked_median_iqr(values: Tensor, batch_idx: Tensor, batch_size: int,
                       eps: float) -> Tensor:
    """Median-IQR normalisation (Eq. 3), computed per frame."""
    out = torch.zeros_like(values)
    for b in range(batch_size):
        sel = batch_idx == b
        if not torch.any(sel):
            continue
        v = values[sel]
        med = torch.median(v)
        q = torch.quantile(v.float(), torch.tensor([0.25, 0.75], device=v.device))
        iqr = q[1] - q[0]
        out[sel] = (v - med) / (iqr + eps)
    return out


def _neighbor_counts(active: Tensor, k: int, circular_azimuth: bool) -> Tensor:
    """``n_r^(k) = |N_k(r)|``: active cells in the k x k neighborhood (Eq. 5).

    Args:
        active (Tensor): (B, 1, H, W) binary map of active cells.
        k (int): neighborhood size.
        circular_azimuth (bool): wrap the azimuth (W) axis. Default zero
            padding is the literal reading of the paper.
    """
    pad = k // 2
    if circular_azimuth:
        # circular along W, zero padding along H (elevation is not cyclic).
        x = F.pad(active, (pad, pad, 0, 0), mode='circular')
        x = F.pad(x, (0, 0, pad, pad), mode='constant', value=0.0)
        counts = F.avg_pool2d(x, kernel_size=k, stride=1, padding=0)
    else:
        counts = F.avg_pool2d(
            active, kernel_size=k, stride=1, padding=pad,
            count_include_pad=True)
    return counts * (k * k)


@torch.no_grad()
def compute_observability(points: Tensor,
                          coors: Tensor,
                          batch_size: int,
                          H: int,
                          W: int,
                          tau_d: float = 0.5,
                          lam: float = 0.5,
                          scales: Sequence[int] = (3, 5),
                          eps: float = 1e-6,
                          circular_azimuth: bool = False,
                          representative_depth: str = 'min') -> Dict[str, Tensor]:
    """Geometry-consistent observability ``o_r = o_r^beam * o_r^neigh`` (Eq. 7).

    Args:
        points (Tensor): (N, >=3) point features; the first three columns are
            xyz in the sensor frame.
        coors (Tensor): (N, 3) int64 per-point range-view coordinates
            ``(batch_idx, v, u)`` produced by the frustum region grouping.
        batch_size (int): number of frames in the batch.
        H, W (int): range-view grid size.
        tau_d (float): depth-consistency threshold of Eq. 1, in metres.
        lam (float): range-penalisation strength ``lambda`` of Eq. 2.
        scales (Sequence[int]): neighborhood sizes ``S`` of Eq. 6.
        representative_depth (str): ``'min'`` (nearest return) or
            ``'median'``.

    Returns:
        dict with
            - ``obs_map``   (B, 1, H, W) float, ``o_r`` (0 on inactive cells)
            - ``active_map``(B, 1, H, W) float, binary active-cell mask
            - ``obs_point`` (N,) float, ``o_r`` broadcast back to each point
            - ``obs_beam_map`` / ``obs_neigh_map`` (B, 1, H, W), for analysis
    """
    device = points.device
    depth = torch.linalg.norm(points[:, :3].float(), 2, dim=1)

    # ---- cell indexing ---------------------------------------------------
    flat = (coors[:, 0] * H + coors[:, 1]) * W + coors[:, 2]
    uniq, inverse = torch.unique(flat, return_inverse=True)
    n_cells = uniq.numel()

    # ---- beam-termination observability (Eqs. 1-4) -----------------------
    if representative_depth == 'min':
        d_r = torch_scatter.scatter_min(depth, inverse, dim=0)[0]
    elif representative_depth == 'median':
        d_r = torch_scatter.scatter_mean(depth, inverse, dim=0)
    else:
        raise ValueError(
            f'unknown representative_depth: {representative_depth}')

    consistent = (torch.abs(depth - d_r[inverse]) < tau_d).float()
    h_r = torch_scatter.scatter_add(consistent, inverse, dim=0)  # Eq. 1

    s_r = torch.log(h_r + eps) - lam * torch.log(d_r + eps)  # Eq. 2

    cell_batch = uniq // (H * W)
    o_beam = torch.sigmoid(
        _masked_median_iqr(s_r, cell_batch, batch_size, eps))  # Eqs. 3-4

    # ---- neighborhood-supported observability (Eqs. 5-6) -----------------
    active_map = torch.zeros(batch_size * H * W, device=device)
    active_map[uniq] = 1.0
    active_map = active_map.view(batch_size, 1, H, W)

    o_neigh = torch.zeros(n_cells, device=device)
    for k in scales:
        counts = _neighbor_counts(active_map, k, circular_azimuth)
        n_k = counts.view(-1)[uniq]
        o_neigh = o_neigh + torch.sigmoid(torch.log1p(n_k))
    o_neigh = o_neigh / len(scales)

    # ---- geometry-consistent observability (Eq. 7) -----------------------
    o_r = o_beam * o_neigh

    def _scatter_dense(vals: Tensor) -> Tensor:
        dense = torch.zeros(batch_size * H * W, device=device)
        dense[uniq] = vals
        return dense.view(batch_size, 1, H, W)

    return dict(
        obs_map=_scatter_dense(o_r),
        obs_beam_map=_scatter_dense(o_beam),
        obs_neigh_map=_scatter_dense(o_neigh),
        active_map=active_map,
        obs_point=o_r[inverse],
        cell_index=uniq,
        point_to_cell=inverse,
    )


def pool_observability(obs_map: Tensor, active_map: Tensor,
                       stride: int) -> Tensor:
    """Bring ``o_r`` to the resolution of a strided backbone stage.

    Averaged over active sub-cells only, so inactive cells do not drag the
    score down.
    """
    if stride == 1:
        return obs_map
    obs_sum = F.avg_pool2d(obs_map, stride, stride) * stride * stride
    act_sum = F.avg_pool2d(active_map, stride, stride) * stride * stride
    return obs_sum / act_sum.clamp(min=1.0)
