"""Online test-time adaptation loop: one pass, in order, no labels."""

import time
from typing import Dict, List, Optional, Sequence, Union

import torch
from mmengine.evaluator import Evaluator
from mmengine.model.wrappers import is_model_wrapper
from mmengine.runner.amp import autocast
from mmengine.runner.base_loop import BaseLoop
from torch.utils.data import DataLoader

try:
    from mmdet3d.registry import LOOPS
except ImportError:  # pragma: no cover
    from mmengine.registry import LOOPS


@LOOPS.register_module()
class OnlineTTALoop(BaseLoop):
    """Adapt the prompt adapters frame by frame while streaming the test set.

    Args:
        dataloader: the test stream. Must not shuffle -- the prototype memory
            requires acquisition order.
        evaluator: metric(s) to accumulate over the stream.
        optimizer (dict): torch optimizer config for the prompt adapters.
        steps_per_frame (int): gradient steps taken per incoming frame.
        predict_after_update (bool): if False (default) the prediction comes
            from the same forward that produced the loss (one forward + one
            backward per frame). If True, a second no-grad forward is scored.
        reset_memory (bool): clear the prototype memory before the stream.
        fp16 (bool): run the forward passes under autocast.
        log_interval (int): frames between progress logs.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 optimizer: Optional[Dict] = None,
                 steps_per_frame: int = 1,
                 predict_after_update: bool = False,
                 reset_memory: bool = True,
                 fp16: bool = False,
                 log_interval: int = 50) -> None:
        super().__init__(runner, dataloader)

        if isinstance(evaluator, (dict, list)):
            self.evaluator = runner.build_evaluator(evaluator)
        else:
            self.evaluator = evaluator
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo

        self.optimizer_cfg = optimizer or dict(type='Adam', lr=5e-4)
        self.steps_per_frame = steps_per_frame
        self.predict_after_update = predict_after_update
        self.reset_memory = reset_memory
        self.fp16 = fp16
        self.log_interval = log_interval
        self.frame_times: List[float] = []

    def _unwrapped_model(self):
        model = self.runner.model
        return model.module if is_model_wrapper(model) else model

    def _build_optimizer(self, model) -> Optional[torch.optim.Optimizer]:
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return None
        cfg = dict(self.optimizer_cfg)
        opt_type = cfg.pop('type')
        opt_cls = getattr(torch.optim, opt_type)
        return opt_cls(params, **cfg)

    def run(self) -> dict:
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')

        model = self._unwrapped_model()
        model.train()  # frozen submodules are held in eval by FRNetObs.train
        if self.reset_memory and hasattr(model, 'reset_adaptation'):
            model.reset_adaptation()

        optimizer = self._build_optimizer(model)
        stats = model.param_stats() if hasattr(model, 'param_stats') else {}
        if stats:
            self.runner.logger.info(
                f'online TTA: {stats["trainable"]/1e6:.4f} M trainable / '
                f'{stats["total"]/1e6:.2f} M total '
                f'({100 * stats["trainable_ratio"]:.2f}%)')

        self.frame_times = []
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch, model, optimizer)
            if self.log_interval and (idx + 1) % self.log_interval == 0:
                self.runner.logger.info(
                    f'online TTA [{idx + 1}/{len(self.dataloader)}] '
                    f'adapt {1000 * self._mean_time():.1f} ms/frame')

        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        if self.frame_times:
            metrics['adapt_time_per_frame'] = self._mean_time()
        if stats:
            metrics['trainable_param_ratio'] = stats['trainable_ratio']
        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics

    def _mean_time(self) -> float:
        # skip the first few frames: they carry CUDA/cuDNN warm-up
        warm = min(5, max(len(self.frame_times) - 1, 0))
        vals = self.frame_times[warm:] or self.frame_times
        return sum(vals) / len(vals)

    def run_iter(self, idx: int, data_batch: Sequence[dict], model,
                 optimizer) -> None:
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        data = model.data_preprocessor(data_batch, False)
        inputs, samples = data['inputs'], data['data_samples']

        if self.steps_per_frame == 0:
            # frozen-prompt evaluation pass (Table-V rows with no TTA loss)
            with torch.no_grad(), autocast(enabled=self.fp16):
                _, voxel_dict = model.adapt(inputs, samples,
                                            compute_loss=False)
        else:
            voxel_dict = None
            for _ in range(self.steps_per_frame):
                with autocast(enabled=self.fp16):
                    loss, voxel_dict = model.adapt(inputs, samples)
                if loss is not None and optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

        if self.predict_after_update and self.steps_per_frame > 0:
            with torch.no_grad(), autocast(enabled=self.fp16):
                _, voxel_dict = model.adapt(
                    inputs, samples, compute_loss=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.frame_times.append(time.perf_counter() - t0)

        with torch.no_grad():
            outputs = model.predict_from_voxel_dict(voxel_dict, samples)

        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
