"""Regenerate configs/_base_/label_space/kitti_nuscenes_shared.py.

``frnet/datasets/label_space.py`` is the single source of truth; mmengine
configs cannot import from it without switching the whole inheritance chain to
lazy-import mode, so the tables are materialised into a base config instead.

    python tools/gen_label_space_config.py
"""

import pprint

from frnet.datasets.label_space import (NUSCENES_TO_SHARED_LUT,
                                        SEMANTICKITTI_TO_SHARED_LUT,
                                        SHARED_CLASSES, SHARED_IGNORE_INDEX)

OUT = 'configs/_base_/label_space/kitti_nuscenes_shared.py'

HEADER = '''"""Shared SemanticKITTI/nuScenes label space for cross-dataset TTA (Table III).

GENERATED from ``frnet/datasets/label_space.py`` (the single source of truth);
regenerate with ``tools/gen_label_space_config.py``. The tables are materialised
here because an mmengine config cannot ``import`` from a package without
switching the whole inheritance chain to lazy-import mode.
"""

'''


def _fmt(name, value):
    return f'{name} = ' + pprint.pformat(value, width=76, compact=True) + '\n\n'


def main():
    body = (_fmt('shared_classes', SHARED_CLASSES) +
            _fmt('shared_ignore_index', SHARED_IGNORE_INDEX) +
            _fmt('semantickitti_to_shared', SEMANTICKITTI_TO_SHARED_LUT) +
            _fmt('nuscenes_to_shared', NUSCENES_TO_SHARED_LUT))
    with open(OUT, 'w') as f:
        f.write(HEADER + body)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
