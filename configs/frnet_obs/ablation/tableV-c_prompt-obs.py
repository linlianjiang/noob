"""Table V (C): observability + prompt, without temporal prototype alignment."""
_base_ = ['../frnet-obs-tta_semantickitti.py']

model = dict(backbone=dict(use_observability=True), proto_cfg=None)
test_cfg = dict(steps_per_frame=0)
