"""Stage 1 -- supervised prompt pre-training on nuScenes.

See ``frnet-obs-prompt-pretrain_semantickitti.py`` for the rationale; this is
the same recipe on top of FRNet's nuScenes config.
"""

_base_ = ['../frnet/frnet-nuscenes_seg.py']

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
        tau_d=0.5,
        lam=0.5,
        obs_scales=(3, 5),
        circular_azimuth=False,
        representative_depth='min'),
    backbone=dict(
        type='ObsPromptFRNetBackbone',
        adapter_type='obs',
        prompt_embed_dims=24,
        prompt_size=4,
        use_observability=True),
    decode_head=dict(type='ObsFRHead'),
    proto_cfg=None,
    freeze_backbone=True)

lr = 1e-4
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

load_from = 'checkpoints/frnet-nuscenes_seg.pth'
