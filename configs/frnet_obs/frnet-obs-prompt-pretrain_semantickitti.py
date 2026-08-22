"""Stage 1 -- supervised prompt pre-training on SemanticKITTI.

Paper: 10 epochs, Adam lr 1e-4, backbone frozen. Inherits FRNet's own recipe
and changes only the modules used, what is trainable, and the optimiser.
``load_from`` must be the official FRNet checkpoint; the adapters are
zero-initialised, so training starts exactly at that model.
"""

_base_ = ['../frnet/frnet-semantickitti_seg.py']

custom_imports = dict(
    imports=[
        'frnet.datasets', 'frnet.datasets.transforms', 'frnet.models',
        'frnet.engine', 'frnet.evaluation'
    ],
    allow_failed_imports=False)

model = dict(
    type='FRNetObs',
    data_preprocessor=dict(
        type='ObsFrustumRangePreprocessor',
        tau_d=0.5,   # Eq. 1, metres
        lam=0.5,     # Eq. 2
        obs_scales=(3, 5),  # Eq. 6
        circular_azimuth=False,
        representative_depth='min'),
    backbone=dict(
        type='ObsPromptFRNetBackbone',
        adapter_type='obs',
        prompt_embed_dims=24,
        prompt_size=4,
        use_observability=True),
    decode_head=dict(type='ObsFRHead'),
    proto_cfg=None,  # Sec. III-C is test-time only
    freeze_backbone=True)

lr = 1e-4
# only the adapters have requires_grad=True, so a full-model optimiser is fine
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=lr, weight_decay=0.0))

param_scheduler = [dict(type='ConstantLR', factor=1.0, by_epoch=True)]

train_cfg = dict(
    _delete_=True, type='EpochBasedTrainLoop', max_epochs=10, val_interval=1)
val_cfg = dict()
test_cfg = dict()

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    sampler=dict(_delete_=True, type='DefaultSampler', shuffle=True))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=1, save_best='miou'))
log_processor = dict(by_epoch=True)

# released FRNet SemanticKITTI weights
load_from = 'checkpoints/frnet-semantickitti_seg.pth'
