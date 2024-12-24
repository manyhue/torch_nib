

from torcheval.metrics.metric import Metric
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    TypeVar,
    Union,
)
import torch
# pyre-fixme[24]: Generic type `Metric` expects 1 type parameter.
TSelf = TypeVar("TSelf", bound="Metric")
TComputeReturn = TypeVar("TComputeReturn")
# pyre-ignore[33]: Flexible key data type for dictionary
TState = Union[torch.Tensor, List[torch.Tensor], Dict[Any, torch.Tensor], int, float]

# operator.add, operator.truediv?
class MeanMetric(Metric):
    def __init__(
        self,
        label,
        statistic: Callable[[object, object], object],
        transform: Callable[[object, object], object] = lambda total, count: total
        / count,
        reduce: Callable[[object, object], object] = lambda total, stat: total + stat,
        device: Optional[torch.device] = None,
    ):
        self.save_attr()
        self.reset()

    @torch.inference_mode()
    def compute(self):
        return self.transform(self._total, self._count)

    @torch.inference_mode()
    def update(self, *args):
        self._total = self.reduce(self._total, self.statistic(*args))
        self._count += 1

    @torch.inference_mode()
    def reset(self):
        self._total = 0
        self._count = 0

    @torch.inference_mode()
    def merge_state(self: TSelf, metrics: Iterable[TSelf]) -> TSelf:
        for metric in metrics:
            self._total = self.reduce(self._total, metric._total)
            self._count += metric._count


class CatMetric(Metric):
    def __init__(
        self,
        label,
        compute=lambda *args: args,
        update=None,
        num_outs=None,
        device=torch.device("cpu"),
        **kwargs,
    ):
        self.label = label
        self._compute = compute
        self._kwargs = kwargs
        self._update = None
        if callable(update):
            assert isinstance(num_outs, int)
            self._update = update
            self.num_outs = num_outs
        else:
            self.num_outs = 2
        self._device = device

        self.reset()

    @torch.inference_mode()
    def compute(self):
        return self._compute(
            *(a.cpu().numpy() for a in self._store),
            **self._kwargs,
        )

    @torch.inference_mode()
    def update(self, *args):
        args = tuple(a.to(self._device) for a in args)
        args = self._update(*args) if callable(self._update) else args

        for idx, tensor in enumerate(self._store):
            new = args[idx]
            self._store[idx] = torch.cat(
                [tensor, new.flatten() if new.dim() == 0 else new], dim=0
            )

    @torch.inference_mode()
    def reset(self):
        self._store = [
            torch.empty(0, dtype=torch.float32, device=self._device)
            for _ in range(self.num_outs)
        ]

    def to(self, device):
        self._store = [u.to(device) for u in self._store]
        self._device = device

    @torch.inference_mode()
    def merge_state(self: TSelf, metrics: Iterable[TSelf]) -> TSelf:
        for metric in metrics:
            for idx, tensor in enumerate(self._store):
                self._store[idx] = torch.cat([tensor, metric._store[idx]], dim=0)


def from_te(torcheval_metric, label, **kwargs):
    class _subclass(torcheval_metric):
        def update(self, *args, **kwargs):
            super().update(*args, **kwargs)

    c = _subclass(**kwargs)
    c.label = label

    return c
