"""Table IV row (C), nuScenes side."""
_base_ = ['../frnet-obs-prompt-pretrain_nuscenes.py']

model = dict(
    backbone=dict(adapter_type='vpt', vpt_num_tokens=5),
    proto_cfg=None)
