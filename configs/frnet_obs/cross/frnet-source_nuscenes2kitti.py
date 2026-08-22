"""Table III "Source" row: the nuScenes model on SemanticKITTI with no adaptation."""
_base_ = ['./frnet-obs-tta_nuscenes2kitti.py']

model = dict(
    # No prompts at all -- the released FRNet weights, evaluated as-is.
    backbone=dict(adapter_type='none'),
    proto_cfg=None)
test_cfg = dict(steps_per_frame=0)
load_from = 'checkpoints/frnet-nuscenes_seg.pth'
