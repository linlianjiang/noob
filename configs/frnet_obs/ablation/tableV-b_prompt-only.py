"""Table V (B): prompt adaptation only -- no observability gate, no
prototypes, so updates are driven uniformly by every location.
"""
_base_ = ['../frnet-obs-tta_semantickitti.py']

model = dict(
    backbone=dict(use_observability=False),
    proto_cfg=None,
    # no prototype target -> no test-time objective; evaluated without updates
    tta_objective='prototype')
test_cfg = dict(steps_per_frame=0)
