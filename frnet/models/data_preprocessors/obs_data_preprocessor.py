from typing import Sequence

import torch
from mmdet3d.registry import MODELS

from frnet.models.obs import compute_observability
from .data_preprocessor import FrustumRangePreprocessor


@MODELS.register_module()
class ObsFrustumRangePreprocessor(FrustumRangePreprocessor):
    """Frustum grouping that also emits the observability score (Sec. III-A).

    Computed here because the operator consumes statistics the grouping already
    produces; the result travels with ``voxel_dict``.

    Args:
        tau_d (float): depth-consistency threshold of Eq. 1, in metres.
        lam (float): range-penalisation strength ``lambda`` of Eq. 2.
        obs_scales (Sequence[int]): neighborhood sizes ``S`` of Eq. 6.
        circular_azimuth (bool): treat the azimuth axis as cyclic when
            counting neighbors. Defaults to False.
        representative_depth (str): ``'min'`` or ``'median'`` for ``d_r``.
    """

    def __init__(self,
                 *args,
                 tau_d: float = 0.5,
                 lam: float = 0.5,
                 obs_scales: Sequence[int] = (3, 5),
                 circular_azimuth: bool = False,
                 representative_depth: str = 'min',
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tau_d = tau_d
        self.lam = lam
        self.obs_scales = tuple(obs_scales)
        self.circular_azimuth = circular_azimuth
        self.representative_depth = representative_depth

    @torch.no_grad()
    def frustum_region_group(self, points, data_samples) -> dict:
        voxel_dict = super().frustum_region_group(points, data_samples)
        obs = compute_observability(
            points=voxel_dict['voxels'],
            coors=voxel_dict['coors'],
            batch_size=len(points),
            H=self.H,
            W=self.W,
            tau_d=self.tau_d,
            lam=self.lam,
            scales=self.obs_scales,
            circular_azimuth=self.circular_azimuth,
            representative_depth=self.representative_depth)
        voxel_dict.update(obs)
        return voxel_dict
