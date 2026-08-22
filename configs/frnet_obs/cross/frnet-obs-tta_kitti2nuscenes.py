"""Table III -- SemanticKITTI -> nuScenes online TTA.

Adaptation runs unsupervised in the source (19-class) space; predictions and
ground truth are re-mapped to the shared taxonomy only for scoring (see
``frnet/datasets/label_space.py`` for the mapping and its caveats). The range
projection follows the target sensor (nuScenes: 32 beams, +10/-30 deg).
"""

_base_ = [
    '../../_base_/datasets/nuscenes_seg.py',
    '../../_base_/models/frnet_obs.py',
    '../../_base_/default_runtime.py',
    '../../_base_/label_space/kitti_nuscenes_shared.py',
]
custom_imports = dict(
    imports=[
        'frnet.datasets', 'frnet.datasets.transforms', 'frnet.models',
        'frnet.engine', 'frnet.evaluation'
    ],
    allow_failed_imports=False)

model = dict(
    # Source model: SemanticKITTI head and label space, target sensor geometry.
    data_preprocessor=dict(
        H=32, W=480, fov_up=10.0, fov_down=-30.0, ignore_index=19),
    backbone=dict(output_shape=(32, 480)),
    decode_head=dict(num_classes=20, ignore_index=19),
    proto_cfg=dict(beta=0.01, tau_p=0.6, temperature=1.0),
    pred_label_map={{_base_.semantickitti_to_shared}})

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=False))

test_evaluator = dict(
    type='SharedSpaceSegMetric',
    gt_label_map={{_base_.nuscenes_to_shared}},
    classes={{_base_.shared_classes}},
    ignore_index={{_base_.shared_ignore_index}})

test_cfg = dict(
    type='OnlineTTALoop',
    optimizer=dict(type='Adam', lr=5e-4, weight_decay=0.0),
    steps_per_frame=1,
    predict_after_update=False,
    reset_memory=True,
    log_interval=100)

train_cfg = None
val_cfg = None
train_dataloader = None
val_dataloader = None
val_evaluator = None
optim_wrapper = None
param_scheduler = None

load_from = 'work_dirs/frnet-obs-prompt-pretrain_semantickitti/best_miou.pth'
