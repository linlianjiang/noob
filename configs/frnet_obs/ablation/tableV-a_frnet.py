"""Table V (A): plain FRNet -- no prompts, no observability, no prototypes.

Must score identically to `configs/frnet/frnet-semantickitti_seg.py`.
"""
_base_ = ['../frnet-obs-tta_semantickitti.py']

model = dict(backbone=dict(adapter_type='none'), proto_cfg=None)
test_cfg = dict(steps_per_frame=0)
