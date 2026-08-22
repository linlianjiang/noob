"""Geometry-constrained test-time prompt tuning segmentor.

Ties the three components together:
  A. observability            -- computed in the data preprocessor (Sec. III-A)
  B. prompt adapters          -- inside the backbone (Sec. III-B)
  C. temporal prototype align -- here, at test time (Sec. III-C)
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch_scatter
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import OptConfigType
from torch import Tensor

from frnet.models.obs import (TemporalPrototypeMemory,
                              prototype_alignment_loss)
from .frnet import FRNet


@MODELS.register_module()
class FRNetObs(FRNet):
    """FRNet + observability-constrained prompt tuning.

    Args:
        proto_cfg (dict, optional): prototype-memory config (``beta``,
            ``tau_p``, ``temperature``). ``None`` disables Sec. III-C
            (Table-V (C) row).
        loss_reduction (str): reduction of Eq. 14; see
            :func:`prototype_alignment_loss`.
        update_before_align (bool): fold the current frame into the memory
            before querying it (Fig. 2(C)). If False, frame ``t``'s target uses
            only frames ``< t``.
        freeze_backbone (bool): train only the prompt adapters. Always True for
            the paper's protocol; exposed for ablation.
        pred_label_map (Sequence[int], optional): LUT applied to predicted
            class ids for cross-dataset evaluation (Table III). Adaptation
            itself always runs in the source label space.
        tta_objective (str): ``'prototype'`` for Eq. 14, or ``'entropy'`` for
            the Tent baseline of Table III (entropy minimisation, no gating, no
            memory; pair it with ``freeze_backbone=False``).
    """

    def __init__(self,
                 *args,
                 proto_cfg: OptConfigType = None,
                 loss_reduction: str = 'weighted_mean',
                 update_before_align: bool = True,
                 freeze_backbone: bool = True,
                 pred_label_map: Optional[Sequence[int]] = None,
                 tta_objective: str = 'prototype',
                 obs_in_loss: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert tta_objective in ('prototype', 'entropy')
        self.tta_objective = tta_objective
        self.obs_in_loss = obs_in_loss
        self.loss_reduction = loss_reduction
        self.update_before_align = update_before_align
        self.freeze_backbone = freeze_backbone

        if pred_label_map is None:
            self.pred_label_map = None
        else:
            self.register_buffer(
                'pred_label_map',
                torch.as_tensor(pred_label_map, dtype=torch.long),
                persistent=False)

        self.prototypes: Optional[TemporalPrototypeMemory] = None
        if proto_cfg is not None:
            self.prototypes = TemporalPrototypeMemory(
                num_classes=self.decode_head.num_classes,
                feat_dim=self.decode_head.channels,
                ignore_index=self.decode_head.ignore_index,
                **proto_cfg)

        if self.freeze_backbone:
            self._freeze_non_prompt()

    # ------------------------------------------------------------------
    # parameter freezing
    # ------------------------------------------------------------------
    def _prompt_module(self) -> Optional[nn.Module]:
        return getattr(self.backbone, 'prompt_adapters', None)

    def _freeze_non_prompt(self) -> None:
        """Freeze everything but the prompt adapters."""
        for p in self.parameters():
            p.requires_grad_(False)
        prompt = self._prompt_module()
        if prompt is not None:
            for p in prompt.parameters():
                p.requires_grad_(True)

    def train(self, mode: bool = True):
        """Keep frozen submodules in eval mode, so BN running stats stay put."""
        super().train(mode)
        if not self.freeze_backbone:
            return self
        prompt = self._prompt_module()
        prompt_ids = set()
        if prompt is not None:
            prompt_ids = {id(m) for m in prompt.modules()}
        for module in self.modules():
            if id(module) not in prompt_ids and module is not self:
                module.train(False)
        if prompt is not None:
            prompt.train(mode)
        return self

    def prompt_parameters(self) -> List[nn.Parameter]:
        prompt = self._prompt_module()
        return [] if prompt is None else list(prompt.parameters())

    def param_stats(self) -> Dict[str, float]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        return dict(
            total=total,
            trainable=trainable,
            trainable_ratio=trainable / max(total, 1))

    # ------------------------------------------------------------------
    # cell-level quantities used by Sec. III-C
    # ------------------------------------------------------------------
    def _cell_evidence(self, voxel_dict: dict) -> Tuple[Tensor, Tensor, Tensor]:
        """Aggregate per-point head outputs onto active range-view cells.

        The classifier is linear, so averaging its input over a cell's points
        equals averaging the logits.

        Returns:
            tuple: ``(f_r, p_r, o_r)`` of shapes (R, D), (R, C), (R,).
        """
        inverse = voxel_dict['point_to_cell']
        cell_index = voxel_dict['cell_index']
        n_cells = cell_index.numel()

        f_cell = torch_scatter.scatter_mean(
            voxel_dict['point_feats_head'], inverse, dim=0, dim_size=n_cells)
        p_cell = torch.softmax(self.decode_head.cls_seg(f_cell), dim=1)
        o_cell = voxel_dict['obs_map'].reshape(-1)[cell_index]
        return f_cell, p_cell, o_cell

    def tta_loss(self, voxel_dict: dict) -> Optional[Tensor]:
        """``L_TTA`` of Eq. 14, or ``None`` if the memory has no evidence yet."""
        if self.tta_objective == 'entropy':
            # Tent baseline: entropy minimisation, no gating, no memory
            p = torch.softmax(voxel_dict['seg_logit'], dim=1)
            return -(p * torch.log(p.clamp_min(1e-8))).sum(dim=1).mean()
        if self.prototypes is None:
            return None
        f_cell, p_cell, o_cell = self._cell_evidence(voxel_dict)
        if not self.obs_in_loss:
            # Table V (D): drop o_r from Eqs. 11 and 14 as well
            o_cell = torch.ones_like(o_cell)

        if self.update_before_align:
            self.prototypes.update(f_cell.detach(), p_cell.detach(), o_cell)
            q = self.prototypes.align(f_cell.detach())
        else:
            q = self.prototypes.align(f_cell.detach())
            self.prototypes.update(f_cell.detach(), p_cell.detach(), o_cell)

        if q is None:
            return None
        return prototype_alignment_loss(
            q, p_cell, o_cell, reduction=self.loss_reduction)

    # ------------------------------------------------------------------
    # online adaptation step
    # ------------------------------------------------------------------
    def adapt(self,
              batch_inputs_dict: dict,
              batch_data_samples: SampleList,
              compute_loss: bool = True) -> Tuple[Optional[Tensor], dict]:
        """One forward pass producing both the TTA loss and the predictions.

        The loss is returned un-backwarded; the caller owns the optimiser step.
        ``compute_loss=False`` skips Sec. III-C, so a post-update re-forward
        does not fold the same frame into the memory twice.
        """
        voxel_dict = self.extract_feat(batch_inputs_dict)
        voxel_dict = self.decode_head.forward(voxel_dict)
        loss = self.tta_loss(voxel_dict) if compute_loss else None
        return loss, voxel_dict

    def predict_from_voxel_dict(self, voxel_dict: dict,
                                batch_data_samples: SampleList) -> SampleList:
        """Turn cached logits into ``Det3DDataSample``s without re-running."""
        batch_input_metas = [d.metainfo for d in batch_data_samples]
        seg_logits = voxel_dict['seg_logit']
        coors = voxel_dict['coors']

        seg_logits_list = []
        for batch_idx, input_metas in enumerate(batch_input_metas):
            logits = seg_logits[coors[:, 0] == batch_idx]
            if 'num_points' in input_metas:
                logits = logits[:input_metas['num_points']]
            seg_logits_list.append(logits.transpose(0, 1))
        return self.postprocess_result(seg_logits_list, batch_data_samples)

    def postprocess_result(self, seg_logits_list: List[Tensor],
                           batch_data_samples: SampleList) -> SampleList:
        """Standard post-processing, then the cross-dataset label re-mapping."""
        samples = super().postprocess_result(seg_logits_list,
                                             batch_data_samples)
        if self.pred_label_map is None:
            return samples
        lut = self.pred_label_map
        for sample in samples:
            pred = sample.pred_pts_seg.pts_semantic_mask
            sample.pred_pts_seg.pts_semantic_mask = lut[pred.clamp(
                0, lut.numel() - 1)]
        return samples

    def reset_adaptation(self) -> None:
        """Drop accumulated temporal evidence (new sequence / new run)."""
        if self.prototypes is not None:
            self.prototypes.reset()
