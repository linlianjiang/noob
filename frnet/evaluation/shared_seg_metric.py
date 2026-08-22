from typing import Dict, List, Sequence

import numpy as np
from mmdet3d.evaluation.metrics.seg_metric import SegMetric
from mmdet3d.registry import METRICS


@METRICS.register_module()
class SharedSpaceSegMetric(SegMetric):
    """Score a cross-dataset run in a shared label space.

    Only the ground truth is re-mapped here, at scoring time; the model
    re-maps its predictions through ``pred_label_map``. Adaptation stays in the
    source label space.

    Args:
        gt_label_map (Sequence[int]): lookup table from the *target dataset's*
            class ids into the shared space.
        classes (Sequence[str]): shared class names, in shared-id order.
        ignore_index (int): shared-space id excluded from the mIoU.
    """

    def __init__(self,
                 gt_label_map: Sequence[int],
                 classes: Sequence[str],
                 ignore_index: int,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.gt_label_map = np.asarray(gt_label_map, dtype=np.int64)
        self.shared_classes = list(classes)
        self.shared_ignore_index = ignore_index

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            ann = data_sample['eval_ann_info']
            mask = np.asarray(ann['pts_semantic_mask'], dtype=np.int64)
            mask = np.clip(mask, 0, len(self.gt_label_map) - 1)
            ann['pts_semantic_mask'] = self.gt_label_map[mask]
        super().process(data_batch, data_samples)

    def compute_metrics(self, results: List) -> Dict[str, float]:
        meta = dict(self.dataset_meta or {})
        meta['label2cat'] = {
            i: name
            for i, name in enumerate(self.shared_classes)
        }
        meta['ignore_index'] = self.shared_ignore_index
        self.dataset_meta = meta
        return super().compute_metrics(results)
