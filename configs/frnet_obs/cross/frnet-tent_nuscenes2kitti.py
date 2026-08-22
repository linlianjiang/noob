"""Table III "Tent" row: entropy minimisation over all parameters, run in
the same harness so timing and trainable fraction are measured identically.
"""
_base_ = ['./frnet-obs-tta_nuscenes2kitti.py']

model = dict(
    backbone=dict(adapter_type='none'),
    proto_cfg=None,
    tta_objective='entropy',
    freeze_backbone=False)
test_cfg = dict(optimizer=dict(type='Adam', lr=1e-4, weight_decay=0.0))
load_from = 'checkpoints/frnet-nuscenes_seg.pth'
