from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS

from frnet.models.obs import ObsPromptAdapter, VPTAdapter, pool_observability
from .frnet_backbone import FRNetBackbone


@MODELS.register_module()
class ObsPromptFRNetBackbone(FRNetBackbone):
    """FRNet backbone with a prompt adapter in front of each residual stage.

    The backbone stays frozen; the adapters inject an input-conditioned
    residual gated by the observability score (Eq. 10).

    ``adapter_type``:
        - ``'obs'``  observability-modulated dynamic prompt (Sec. III-B).
        - ``'vpt'``  vanilla VPT baseline of Table IV: globally shared,
          spatially uniform tokens with no observability gate.
        - ``'none'`` plain FRNet, for the Table-V (A) row.

    Args:
        adapter_type (str): see above.
        prompt_embed_dims (int): attention width ``d`` of the adapter.
        prompt_size (int): number of prompt tokens ``p`` per location.
        vpt_num_tokens (int): ``M`` for the VPT baseline.
        use_observability (bool): gate the prompt residual by ``o_r``. False
            with ``adapter_type='obs'`` gives the Table-V (B) row.
        adapter_position (str): ``'pre'`` (default) applies the gated residual
            to the stage input; ``'post'`` runs the adapter as a parallel
            branch off ``F_{i-1}`` and adds it to the stage output. Eq. 10 does
            not disambiguate the two.
    """

    def __init__(self,
                 *args,
                 stem_channels: int = 128,
                 out_channels: Sequence[int] = (128, 128, 128, 128),
                 adapter_type: str = 'obs',
                 prompt_embed_dims: int = 24,
                 prompt_size: int = 4,
                 vpt_num_tokens: int = 5,
                 use_observability: bool = True,
                 adapter_position: str = 'pre',
                 **kwargs) -> None:
        super().__init__(
            *args, stem_channels=stem_channels, out_channels=out_channels,
            **kwargs)
        assert adapter_type in ('obs', 'vpt', 'none')
        assert adapter_position in ('pre', 'post')
        self.adapter_type = adapter_type
        self.use_observability = use_observability
        self.adapter_position = adapter_position
        if adapter_position == 'post':
            assert adapter_type != 'vpt', \
                'the VPT baseline replaces the feature rather than producing ' \
                'a residual, so it has no parallel-branch form'
            assert all(c == out_channels[0] for c in out_channels) and \
                stem_channels == out_channels[0], \
                'the parallel branch adds Delta_i (computed at the stage ' \
                'input width) to the stage output, so all stage widths must ' \
                'match'

        # stage input width: stem output for stage 0, previous stage after
        stage_in_channels = [stem_channels] + list(out_channels[:-1])

        if adapter_type == 'obs':
            self.prompt_adapters = nn.ModuleList([
                ObsPromptAdapter(c, prompt_embed_dims, prompt_size)
                for c in stage_in_channels
            ])
        elif adapter_type == 'vpt':
            self.prompt_adapters = nn.ModuleList(
                [VPTAdapter(c, vpt_num_tokens) for c in stage_in_channels])
        else:
            self.prompt_adapters = None

    @property
    def prompt_parameters(self):
        if self.prompt_adapters is None:
            return []
        return list(self.prompt_adapters.parameters())

    def _stage_input_strides(self) -> Sequence[int]:
        """Overall stride of the feature map *entering* each residual stage."""
        return [1] + list(self.strides[:-1])

    def forward(self, voxel_dict: dict) -> dict:
        """FRNet's forward with a prompt adapter applied to each stage input.

        Mirrors :meth:`FRNetBackbone.forward`; the only additions are the
        ``prompt_adapters`` calls before each ``res_layer``.
        """
        point_feats = voxel_dict['point_feats'][-1]
        voxel_feats = voxel_dict['voxel_feats']
        voxel_coors = voxel_dict['voxel_coors']
        pts_coors = voxel_dict['coors']
        batch_size = pts_coors[-1, 0].item() + 1

        obs_map = voxel_dict.get('obs_map', None)
        active_map = voxel_dict.get('active_map', None)

        x = self.frustum2pixel(voxel_feats, voxel_coors, batch_size, stride=1)
        x = self.stem(x)
        map_point_feats = self.pixel2point(x, pts_coors, stride=1)
        fusion_point_feats = torch.cat((map_point_feats, point_feats), dim=1)
        point_feats = self.point_stem(fusion_point_feats)
        stride_voxel_coors, frustum_feats = self.point2frustum(
            point_feats, pts_coors, stride=1)
        pixel_feats = self.frustum2pixel(
            frustum_feats, stride_voxel_coors, batch_size, stride=1)
        fusion_pixel_feats = torch.cat((pixel_feats, x), dim=1)
        x = self.fusion_stem(fusion_pixel_feats)

        in_strides = self._stage_input_strides()
        outs = [x]
        out_points = [point_feats]
        for i, layer_name in enumerate(self.res_layers):
            # ---- observability-constrained prompt tuning (Eqs. 8-10) ----
            delta = None
            if self.prompt_adapters is not None:
                if self.use_observability and obs_map is not None:
                    gate = pool_observability(obs_map, active_map,
                                              in_strides[i])
                else:
                    gate = torch.ones(
                        (1, 1, 1, 1), device=x.device, dtype=x.dtype)
                if self.adapter_position == 'pre':
                    x = self.prompt_adapters[i](x, gate)
                else:
                    # parallel branch: Eqs. 8-9 read F_{i-1}, added post-stage
                    delta = gate * self.prompt_adapters[i].residual(x)

            res_layer = getattr(self, layer_name)
            x = res_layer(x)

            if delta is not None:
                if delta.shape[-2:] != x.shape[-2:]:
                    delta = F.adaptive_avg_pool2d(delta, x.shape[-2:])
                x = x + delta

            # frustum-to-point fusion
            map_point_feats = self.pixel2point(
                x, pts_coors, stride=self.strides[i])
            fusion_point_feats = torch.cat((map_point_feats, point_feats),
                                           dim=1)
            point_feats = self.point_fusion_layers[i](fusion_point_feats)

            # point-to-frustum fusion
            stride_voxel_coors, frustum_feats = self.point2frustum(
                point_feats, pts_coors, stride=self.strides[i])
            pixel_feats = self.frustum2pixel(
                frustum_feats,
                stride_voxel_coors,
                batch_size,
                stride=self.strides[i])
            fusion_pixel_feats = torch.cat((pixel_feats, x), dim=1)
            fuse_out = self.pixel_fusion_layers[i](fusion_pixel_feats)
            # residual-attentive
            attention_map = self.attention_layers[i](fuse_out)
            x = fuse_out * attention_map + x
            outs.append(x)
            out_points.append(point_feats)

        for i in range(len(outs)):
            if outs[i].shape != outs[0].shape:
                outs[i] = F.interpolate(
                    outs[i],
                    size=outs[0].size()[2:],
                    mode='bilinear',
                    align_corners=True)

        outs[0] = torch.cat(outs, dim=1)
        out_points[0] = torch.cat(out_points, dim=1)

        for layer_name, point_layer_name in zip(self.fuse_layers,
                                                self.point_fuse_layers):
            fuse_layer = getattr(self, layer_name)
            outs[0] = fuse_layer(outs[0])
            point_fuse_layer = getattr(self, point_layer_name)
            out_points[0] = point_fuse_layer(out_points[0])

        voxel_dict['voxel_feats'] = outs
        voxel_dict['point_feats_backbone'] = out_points
        return voxel_dict
