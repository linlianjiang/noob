"""Base model for observability-constrained test-time prompt tuning.

Everything outside ``backbone.prompt_adapters`` is frozen; the adapters are the
only trainable parameters, both during supervised prompt pre-training and
during online test-time adaptation.
"""

_base_ = ['./frnet.py']

model = dict(
    type='FRNetObs',
    data_preprocessor=dict(
        type='ObsFrustumRangePreprocessor',
        # Eq. 1: depth-consistency threshold for beam terminations (metres).
        tau_d=0.5,
        # Eq. 2: strength of the range penalisation.
        lam=0.5,
        # Eq. 6: multi-scale Chebyshev neighborhood sizes S.
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
    proto_cfg=dict(
        # Eq. 11 is m <- (1 - beta) m + beta * new, so beta is the weight of the
        # new evidence; beta=0.01 is the "EMA decay 0.99" of the paper.
        beta=0.01,
        tau_p=0.6,
        temperature=1.0),
    loss_reduction='weighted_mean',
    update_before_align=True,
    freeze_backbone=True)
