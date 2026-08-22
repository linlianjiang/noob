"""Table IV row (C): vanilla VPT on FRNet -- M=5 globally shared tokens,
identity-initialised projection, no observability gate.
"""
_base_ = ['../frnet-obs-prompt-pretrain_semantickitti.py']

model = dict(
    backbone=dict(adapter_type='vpt', vpt_num_tokens=5),
    proto_cfg=None)
