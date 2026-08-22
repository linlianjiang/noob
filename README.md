# No Adaptation Without Observation
## Observability-Constrained Test-Time Prompt Tuning for LiDAR Semantic Segmentation (IROS 2026)

[Linlian Jiang](https://linlianjiang.github.io/)<sup>1,2</sup>,
[Wentao Ju](https://openreview.net/profile?id=~Wentao_Ju1)<sup>1</sup>,
[Sadman Rakib Pinon](https://srpinon.com/)<sup>1,2</sup>,
[Jianwei Xian](https://openreview.net/profile?id=~Jianwei_Xian1)<sup>1</sup>,
[Zhixiang Chi](https://scholar.google.ca/citations?hl=en&user=0s-HzGIAAAAJ&view_op=list_works&sortby=pubdate)<sup>3</sup>,
[Xinxin Zuo](https://sites.google.com/site/xinxinzuohome/home)<sup>1*</sup>,
[Yang Wang](https://users.encs.concordia.ca/~wayang/)<sup>1,2*</sup>

<sup>1</sup>Concordia University &nbsp; <sup>2</sup>Mila – Quebec AI Institute &nbsp; <sup>3</sup>University of Toronto

[Project Page](https://linlianjiang.github.io/noob/) · [arXiv](https://arxiv.org/abs/2606.30937) · [PDF](https://arxiv.org/pdf/2606.30937)

### TODO List
- [x] Public code release

## Installation

```bash
conda create -n obs python=3.10 -y
conda activate obs
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
pip install openmim setuptools==69.5.1
mim install mmengine mmcv==2.1.0 mmdet==3.2.0 mmdet3d==1.4.0
export PYTHONPATH=$PWD:$PYTHONPATH
```

On a Compute Canada cluster, install with `PYTHONPATH` and `PIP_CONFIG_FILE`
unset — the module stack's `_manylinux.py` shim and pip config otherwise reject
every PyPI wheel. `env.sh` activates the resulting environment.

## Data

Prepare SemanticKITTI and nuScenes following
[`docs/DATA_PREPARE.md`](docs/DATA_PREPARE.md), then place the released FRNet
checkpoints in `checkpoints/`:

```
checkpoints/frnet-semantickitti_seg.pth
checkpoints/frnet-nuscenes_seg.pth
```

They are available from the
[FRNet release](https://drive.google.com/drive/folders/173ZIzO7HOSE2JQ7lz_Ikk4O85Mau68el?usp=sharing).

## Usage

### Stage 1 — prompt pre-training

Trains the prompt adapters on the source training split with the backbone
frozen (10 epochs, Adam 1e-4).

```bash
python train.py configs/frnet_obs/frnet-obs-prompt-pretrain_semantickitti.py
python train.py configs/frnet_obs/frnet-obs-prompt-pretrain_nuscenes.py
```

Multi-GPU: `bash dist_train.sh <config> <num_gpus>`.

### Stage 2 — online test-time adaptation

Streams the test set once, taking one prompt update per frame (Adam 5e-4);
backbone and classifier stay frozen.

```bash
python test.py configs/frnet_obs/frnet-obs-tta_semantickitti.py \
       work_dirs/frnet-obs-prompt-pretrain_semantickitti/best_miou.pth
```

Alongside mIoU, the run reports `adapt_time_per_frame` and
`trainable_param_ratio`.

### Cross-dataset

```bash
python test.py configs/frnet_obs/cross/frnet-obs-tta_kitti2nuscenes.py \
       work_dirs/frnet-obs-prompt-pretrain_semantickitti/best_miou.pth
```

Predictions and ground truth are mapped into a 12-class shared space
(`frnet/datasets/label_space.py`) for scoring; adaptation itself stays
unsupervised in the source label space.

## Configs

| Experiment | Config |
|---|---|
| In-domain TTA | `configs/frnet_obs/frnet-obs-tta_{semantickitti,nuscenes}.py` |
| Cross-dataset (ours) | `configs/frnet_obs/cross/frnet-obs-tta_{kitti2nuscenes,nuscenes2kitti}.py` |
| Cross-dataset (source / Tent) | `configs/frnet_obs/cross/frnet-{source,tent}_*.py` |
| VPT baseline | `configs/frnet_obs/ablation/tableIV-vpt_*.py` |
| Component ablation | `configs/frnet_obs/ablation/tableV-{a,b,c,d,e}_*.py` |

Main options, all set under `model` / `test_cfg` in a config:

| Option | Default | Effect |
|---|---|---|
| `backbone.adapter_type` | `'obs'` | `'vpt'` for the VPT baseline, `'none'` for plain FRNet |
| `backbone.use_observability` | `True` | `False` removes the gate from the prompt residual |
| `backbone.prompt_embed_dims` / `prompt_size` | `24` / `4` | Adapter width and number of prompt tokens |
| `proto_cfg` | `beta=0.01, tau_p=0.6` | `None` disables temporal prototype alignment |
| `obs_in_loss` | `True` | `False` drops `o_r` from the memory update and the loss |
| `test_cfg.steps_per_frame` | `1` | `0` evaluates without online updates |

## Citation

```bibtex
@article{jiang2026noob,
    title = {No Adaptation Without Observation: Observability-Constrained
             Test-Time Prompt Tuning for LiDAR Semantic Segmentation},
    author = {Jiang, Linlian and Ju, Wentao and Pinon, Sadman Rakib and
              Xian, Jianwei and Chi, Zhixiang and Zuo, Xinxin and Wang, Yang},
    journal = {arXiv preprint arXiv:2606.30937},
    year = {2026}
}
```

## Acknowledgements

This codebase is built on [FRNet](https://github.com/Xiangxu-0103/FRNet) and
[MMDetection3D](https://github.com/open-mmlab/mmdetection3d), and is released
under the [Apache 2.0 license](LICENSE).
