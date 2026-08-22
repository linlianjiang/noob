"""Stage 2 -- online TTA on the SemanticKITTI val stream. Backbone and
classifier stay frozen; the prompts take one L_TTA step (Eq. 14) per frame.
"""

_base_ = ['./frnet-obs-prompt-pretrain_semantickitti.py']

model = dict(
    # _delete_: the pre-training config sets proto_cfg=None
    proto_cfg=dict(_delete_=True, beta=0.01, tau_p=0.6, temperature=1.0),
    loss_reduction='weighted_mean',
    update_before_align=True,
    auxiliary_head=None)  # supervised-only

# acquisition order, one frame at a time
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=False))

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

# stage-1 output
load_from = 'work_dirs/frnet-obs-prompt-pretrain_semantickitti/best_miou.pth'
