"""Shared label space for cross-dataset TTA (Table III).

The paper does not state which label space the cross-dataset mIoU is computed
in, and the two taxonomies are not nested (19 vs. 16 classes). This mapping
keeps every category with an unambiguous counterpart on both sides (e.g. KITTI
``road`` + ``parking`` -> nuScenes ``driveable_surface``) and ignores the rest
(KITTI ``bicyclist``/``motorcyclist``, nuScenes ``barrier``/``traffic_cone``).
It is an assumption of the reproduction -- absolute numbers move with it.
"""

from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
SHARED_CLASSES: List[str] = [
    'car',
    'bicycle',
    'motorcycle',
    'truck',
    'other-vehicle',
    'person',
    'drivable-surface',
    'sidewalk',
    'terrain',
    'other-ground',
    'vegetation',
    'manmade',
]
SHARED_IGNORE_INDEX = len(SHARED_CLASSES)  # 12

# --- SemanticKITTI (19 classes + 19 = unlabeled) ---------------------------
_K = {
    'car': 0, 'bicycle': 1, 'motorcycle': 2, 'truck': 3, 'other-vehicle': 4,
    'person': 5, 'bicyclist': 6, 'motorcyclist': 7, 'road': 8, 'parking': 9,
    'sidewalk': 10, 'other-ground': 11, 'building': 12, 'fence': 13,
    'vegetation': 14, 'trunck': 15, 'terrian': 16, 'pole': 17,
    'traffic-sign': 18, 'unlabeled': 19,
}
SEMANTICKITTI_TO_SHARED: Dict[int, int] = {
    _K['car']: 0,
    _K['bicycle']: 1,
    _K['motorcycle']: 2,
    _K['truck']: 3,
    _K['other-vehicle']: 4,
    _K['person']: 5,
    _K['bicyclist']: SHARED_IGNORE_INDEX,
    _K['motorcyclist']: SHARED_IGNORE_INDEX,
    _K['road']: 6,
    _K['parking']: 6,
    _K['sidewalk']: 7,
    _K['other-ground']: 9,
    _K['building']: 11,
    _K['fence']: 11,
    _K['vegetation']: 10,
    _K['trunck']: 10,
    _K['terrian']: 8,
    _K['pole']: 11,
    _K['traffic-sign']: 11,
    _K['unlabeled']: SHARED_IGNORE_INDEX,
}

# --- nuScenes (16 classes + 16 = unlabeled) --------------------------------
_N = {
    'barrier': 0, 'bicycle': 1, 'bus': 2, 'car': 3, 'construction_vehicle': 4,
    'motorcycle': 5, 'pedestrian': 6, 'traffic_cone': 7, 'trailer': 8,
    'truck': 9, 'driveable_surface': 10, 'other_flat': 11, 'sidewalk': 12,
    'terrain': 13, 'manmade': 14, 'vegetation': 15, 'unlabeled': 16,
}
NUSCENES_TO_SHARED: Dict[int, int] = {
    _N['barrier']: SHARED_IGNORE_INDEX,
    _N['bicycle']: 1,
    _N['bus']: 4,
    _N['car']: 0,
    _N['construction_vehicle']: 4,
    _N['motorcycle']: 2,
    _N['pedestrian']: 5,
    _N['traffic_cone']: SHARED_IGNORE_INDEX,
    _N['trailer']: 4,
    _N['truck']: 3,
    _N['driveable_surface']: 6,
    _N['other_flat']: 9,
    _N['sidewalk']: 7,
    _N['terrain']: 8,
    _N['manmade']: 11,
    _N['vegetation']: 10,
    _N['unlabeled']: SHARED_IGNORE_INDEX,
}


def as_lut(mapping: Dict[int, int]) -> List[int]:
    """Dense lookup table indexed by source class id."""
    size = max(mapping) + 1
    lut = np.full(size, SHARED_IGNORE_INDEX, dtype=np.int64)
    for src, dst in mapping.items():
        lut[src] = dst
    return lut.tolist()


SEMANTICKITTI_TO_SHARED_LUT = as_lut(SEMANTICKITTI_TO_SHARED)
NUSCENES_TO_SHARED_LUT = as_lut(NUSCENES_TO_SHARED)
