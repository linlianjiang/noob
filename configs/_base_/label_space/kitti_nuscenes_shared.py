"""Shared SemanticKITTI/nuScenes label space for cross-dataset TTA (Table III).

GENERATED from ``frnet/datasets/label_space.py`` (the single source of truth);
regenerate with ``tools/gen_label_space_config.py``. The tables are materialised
here because an mmengine config cannot ``import`` from a package without
switching the whole inheritance chain to lazy-import mode.
"""

shared_classes = ['car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person',
 'drivable-surface', 'sidewalk', 'terrain', 'other-ground', 'vegetation',
 'manmade']

shared_ignore_index = 12

semantickitti_to_shared = [0, 1, 2, 3, 4, 5, 12, 12, 6, 6, 7, 9, 11, 11, 10, 10, 8, 11, 11, 12]

nuscenes_to_shared = [12, 1, 4, 0, 4, 2, 5, 12, 4, 3, 6, 9, 7, 8, 11, 10, 12]

