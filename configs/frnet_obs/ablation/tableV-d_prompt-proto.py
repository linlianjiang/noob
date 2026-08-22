"""Table V (D): prompt + temporal prototypes, no observability anywhere --
``use_observability=False`` drops the Eq. 10 gate, ``obs_in_loss=False`` drops
``o_r`` from Eqs. 11 and 14.
"""
_base_ = ['../frnet-obs-tta_semantickitti.py']

model = dict(backbone=dict(use_observability=False), obs_in_loss=False)
