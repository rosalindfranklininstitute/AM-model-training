from __future__ import annotations
import typing
from dataclasses import dataclass, field, fields, InitVar, asdict

import numpy as np
import torch
from torchmetrics.segmentation import DiceScore, MeanIoU
from torchmetrics.classification import MulticlassConfusionMatrix

if typing.TYPE_CHECKING:
    from collections.abc import Collection, Sequence, Iterable


def get_metrics_from_confusion_matrix(
    confusion_matrix: torch.Tensor | None,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[None, None, None, None]
):
    if confusion_matrix is None:
        return None, None, None, None
    tp = confusion_matrix.diag()
    fp = confusion_matrix.sum(0) - tp
    fn = confusion_matrix.sum(1) - tp

    accuracy = tp / (tp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return (
        torch.nan_to_num(f1),
        torch.nan_to_num(accuracy),
        torch.nan_to_num(precision),
        torch.nan_to_num(recall),
    )


@dataclass
class MetricsOutput:
    device: InitVar[torch.device | None]
    loss: float
    weights: Sequence[float] | None | None = None

    confusion_matrix: torch.Tensor | None = None

    iou: torch.Tensor | None = None
    mean_iou: float = field(init=False)
    weighted_average_iou: float | None = field(init=False, default=None)

    dice: torch.Tensor | None = None
    mean_dice: float = field(init=False)
    weighted_average_dice: float | None = field(init=False, default=None)

    # Calculated from confusion matrix (and weights):
    f1: torch.Tensor | None = field(init=False)
    mean_f1: float = field(init=False)
    weighted_average_f1: float | None = field(init=False, default=None)

    accuracy: torch.Tensor | None = field(init=False)
    mean_accuracy: float = field(init=False)
    weighted_average_accuracy: float | None = field(init=False, default=None)

    precision: torch.Tensor | None = field(init=False)
    mean_precision: float = field(init=False)
    weighted_average_precision: float | None = field(init=False, default=None)

    recall: torch.Tensor | None = field(init=False)
    mean_recall: float = field(init=False)
    weighted_average_recall: float | None = field(init=False, default=None)

    def __post_init__(self, device: torch.device | None) -> None:
        self.f1, self.accuracy, self.precision, self.recall = (
            get_metrics_from_confusion_matrix(confusion_matrix=self.confusion_matrix)
        )

        if self.weights is None or device is None:
            weights_tensor = None
        else:
            weights_tensor = torch.Tensor(self.weights).to(device)

        for f in fields(self):
            if not f.init or f.name in ("weights", "loss", "confusion_matrix"):
                continue
            value = getattr(self, f.name)
            if value is None:
                continue
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
    confusion_matrix: MulticlassConfusionMatrix = field(init=False)
    iou: MeanIoU = field(init=False)
    dice: DiceScore = field(init=False)

    def __post_init__(self, device: torch.device, num_classes: int) -> None:
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes).to(
            device
        )
        self.iou = MeanIoU(
            num_classes=num_classes, per_class=True, input_format="index"
        ).to(device)
        self.dice = DiceScore(
            num_classes=num_classes, average="none", input_format="index"
        ).to(device)

    def update(
        self,
        y: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> None:
        for f in fields(self):
            getattr(self, f.name).update(y_pred, y)

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
            device=device,
            loss=loss,
            weights=weights,
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
            device=device,
            loss=float(np.nanmean(tuple(step_losses))),
            weights=weights,
            **kwargs,
        )

    def reset(self) -> None:
        for f in fields(self):
            getattr(self, f.name).reset()
