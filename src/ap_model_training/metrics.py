from __future__ import annotations
import typing
from statistics import mean
from dataclasses import dataclass, field, fields, InitVar, asdict

import torch
from torchmetrics.segmentation import DiceScore, MeanIoU
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassF1Score,
    MulticlassRecall,
)

if typing.TYPE_CHECKING:
    from collections.abc import Collection, Sequence, Iterable

@dataclass
class MetricsOutput:
    weights: InitVar[Sequence[float] | None]
    device: InitVar[torch.device | None]

    loss: float

    iou: torch.Tensor
    mean_iou: float = field(init=False)
    weighted_average_iou: float | None = field(init=False, default=None)

    dice: torch.Tensor
    mean_dice: float = field(init=False)
    weighted_average_dice: float | None = field(init=False, default=None)

    f1: torch.Tensor
    mean_f1: float = field(init=False)
    weighted_average_f1: float | None = field(init=False, default=None)

    accuracy: torch.Tensor
    mean_accuracy: float = field(init=False)
    weighted_average_accuracy: float | None = field(init=False, default=None)

    precision: torch.Tensor
    mean_precision: float = field(init=False)
    weighted_average_precision: float | None = field(init=False, default=None)

    recall: torch.Tensor
    mean_recall: float = field(init=False)
    weighted_average_recall: float | None = field(init=False, default=None)

    def __post_init__(
        self, weights: Sequence[float] | None, device: torch.device | None
    ) -> None:
        if weights is None or device is None:
            weights_tensor = None
        else:
            weights_tensor = torch.Tensor(weights).to(device)

        for f in fields(self):
            if not f.init or f.name == "loss":
                continue
            value = getattr(self, f.name)
            mean = torch.nanmean(value)
            object.__setattr__(self, f"mean_{f.name}", mean.item())
            if weights_tensor is not None:
                weighted_mean = torch.nansum(
                    torch.mul(value, weights_tensor)
                ) / torch.sum(weights_tensor)
                object.__setattr__(
                    self, f"weighted_average_{f.name}", weighted_mean.item()
                )

    def to_dict(
        self,
        split_labels: bool = False,
        labels: Collection[str] | None = None,
        prefix: str | None = None,
    ) -> dict[str, typing.Any]:
        d = asdict(self)
        if split_labels:
            if labels is None:
                raise ValueError("No labels have been supplied")
            for k in tuple(d.keys()):
                v = d[k]
                if isinstance(v, torch.Tensor) and len(v) > 1:
                    v = d.pop(k).numpy(force=True).tolist()
                    if len(v) != len(labels):
                        raise ValueError(
                            "An incorrect number of labels have been supplied"
                        )
                    for i, label in enumerate(labels):
                        d[f"{label}_{k}"] = float(v[i])
        if prefix is not None:
            for k in tuple(d.keys()):
                d[f"{prefix}_{k}"] = d.pop(k)
        return d


@dataclass
class Metrics:
    device: InitVar[torch.device]
    num_classes: InitVar[int]
    iou: MeanIoU = field(init=False)
    dice: DiceScore = field(init=False)
    f1: MulticlassF1Score = field(init=False)
    accuracy: MulticlassAccuracy = field(init=False)
    precision: MulticlassPrecision = field(init=False)
    recall: MulticlassRecall = field(init=False)

    def __post_init__(self, device: torch.device, num_classes: int) -> None:
        self.iou = MeanIoU(
            num_classes=num_classes, per_class=True, input_format="index"
        ).to(device)
        self.dice = DiceScore(
            num_classes=num_classes, average="none", input_format="index"
        ).to(device)
        self.f1 = MulticlassF1Score(num_classes=num_classes, average="none").to(device)
        self.accuracy = MulticlassAccuracy(num_classes=num_classes, average="none").to(
            device
        )
        self.precision = MulticlassPrecision(
            num_classes=num_classes, average="none"
        ).to(device)
        self.recall = MulticlassRecall(num_classes=num_classes, average="none").to(
            device
        )

    def get_step_metrics(
        self,
        loss: float,
        y: torch.Tensor,
        y_pred: torch.Tensor,
        weights: Sequence[float] | None = None,
        device: torch.device | None = None,
    ) -> MetricsOutput:
        kwargs = {
            f.name: getattr(self, f.name).forward(y_pred, y) for f in fields(self)
        }
        return MetricsOutput(
            weights=weights,
            device=device,
            loss=loss,
            **kwargs,
        )

    def get_epoch_metrics(
        self,
        step_losses: Iterable[float],
        weights: Sequence[float] | None = None,
        device: torch.device | None = None,
    ) -> MetricsOutput:
        kwargs = {f.name: getattr(self, f.name).compute() for f in fields(self)}
        return MetricsOutput(
            weights=weights,
            device=device,
            loss=float(mean(step_losses)),
            **kwargs,
        )

    def reset(self) -> None:
        for f in fields(self):
            getattr(self, f.name).reset()
