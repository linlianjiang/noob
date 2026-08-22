from typing import Sequence

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MapToSharedLabelSpace(BaseTransform):
    """Re-map ground-truth class ids into the cross-dataset shared space.

    Applied after ``PointSegClassMapping``, on the dataset's own class ids.

    Args:
        label_map (Sequence[int]): lookup table indexed by source class id.
    """

    def __init__(self, label_map: Sequence[int]) -> None:
        self.label_map = np.asarray(label_map, dtype=np.int64)

    def transform(self, results: dict) -> dict:
        if 'pts_semantic_mask' in results:
            mask = np.asarray(results['pts_semantic_mask'], dtype=np.int64)
            mask = np.clip(mask, 0, len(self.label_map) - 1)
            results['pts_semantic_mask'] = self.label_map[mask]
        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_source_classes=' \
               f'{len(self.label_map)})'
