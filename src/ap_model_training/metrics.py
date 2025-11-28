from __future__ import annotations
import typing
from dataclasses import dataclass, field, fields, InitVar, asdict
from collections.abc import Collection

import numpy as np
from sklearn.metrics import (
    jaccard_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike


def dice_score(
    y_true: NDArray[np.integer[typing.Any]], y_pred: NDArray[np.integer[typing.Any]]
) -> NDArray[np.float64]:
    smooth = 1.0
    intersection = np.sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)


@dataclass
class EpochMetrics:
    loss: float

    iou: NDArray[np.float64]
    mean_iou: float
    weighted_average_iou: float | None

    dice: NDArray[np.float64]
    mean_dice: float
    weighted_average_dice: float | None

    f1: NDArray[np.float64]
    mean_f1: float
    weighted_average_f1: float | None

    accuracy: NDArray[np.float64]
    mean_accuracy: float
    weighted_average_accuracy: float | None

    precision: NDArray[np.float64]
    mean_precision: float
    weighted_average_precision: float | None

    recall: NDArray[np.float64]
    mean_recall: float
    weighted_average_recall: float | None

    @classmethod
    def from_step_metrics(cls, step_metrics: list[StepMetrics]) -> "EpochMetrics":
        d: dict[str, NDArray[np.float64] | float | None] = {}
        for f in fields(StepMetrics):
            v = np.mean(
                tuple(
                    getattr(_, f.name)
                    for _ in step_metrics
                    if getattr(_, f.name) is not None
                )
            )
            if not isinstance(v, np.ndarray):
                if np.isnan(v):
                    v = None
                else:
                    v = v.item()
            d[f.name] = v
        return cls(**d)  # type: ignore

    def to_dict(
        self,
        split_labels: bool = False,
        labels: Collection[str] | None = None,
        prefix: str | None = None,
    ) -> dict[str, typing.Any]:
        d = asdict(self)
        if not split_labels:
            return d
        elif labels is None:
            raise ValueError("No labels have been supplied")
        for k in tuple(d.keys()):
            v = d.pop(k)
            if isinstance(v, np.ndarray):
                v = v.tolist()
                if len(v) != len(labels):
                    raise ValueError("An incorrect number of labels have been supplied")
                for i, label in enumerate(labels):
                    d[f"{label}_{k}"] = float(v[i])
        if prefix is not None:
            for k in tuple(d.keys()):
                d[f"{prefix}_{k}"] = d.pop(k)
        return d


@dataclass
class StepMetrics:
    weights: InitVar[ArrayLike | None]

    loss: float

    iou: NDArray[np.float64]
    mean_iou: float = field(init=False)
    weighted_average_iou: float | None = field(init=False, default=None)

    dice: NDArray[np.float64]
    mean_dice: float = field(init=False)
    weighted_average_dice: float | None = field(init=False, default=None)

    f1: NDArray[np.float64]
    mean_f1: float = field(init=False)
    weighted_average_f1: float | None = field(init=False, default=None)

    accuracy: NDArray[np.float64]
    mean_accuracy: float = field(init=False)
    weighted_average_accuracy: float | None = field(init=False, default=None)

    precision: NDArray[np.float64]
    mean_precision: float = field(init=False)
    weighted_average_precision: float | None = field(init=False, default=None)

    recall: NDArray[np.float64]
    mean_recall: float = field(init=False)
    weighted_average_recall: float | None = field(init=False, default=None)

    def __post_init__(self, weights: ArrayLike | None):
        for f in fields(self):
            if not f.init:
                continue
            value = getattr(self, f.name)
            mean = np.mean(value)
            object.__setattr__(self, f"mean_{f.name}", mean.item())
            if weights is not None:
                weighted_mean = np.sum(np.multiply(value, weights), axis=1) / np.sum(
                    weights
                )
                object.__setattr__(self, f"mean_{f.name}", weighted_mean.item())

    @classmethod
    def calculate_metrics(
        cls,
        loss: float,
        y: NDArray[np.integer[typing.Any]],
        y_pred: NDArray[np.integer[typing.Any]],
        weights: ArrayLike | None,
    ) -> "StepMetrics":
        flattened_y = y.swapaxes(0, 1).reshape((y.shape[1], -1))
        flattened_y_pred = y_pred.swapaxes(0, 1).reshape((y.shape[1], -1))

        iou = np.asarray(
            jaccard_score(y_true=flattened_y, y_pred=flattened_y_pred, average=None)
        )
        dice = np.asarray(dice_score(y_true=flattened_y, y_pred=flattened_y_pred))
        f1 = np.asarray(
            f1_score(y_true=flattened_y, y_pred=flattened_y_pred, average=None)
        )
        accuracy = np.asarray(
            accuracy_score(y_true=flattened_y, y_pred=flattened_y_pred)
        )
        precision = np.asarray(
            precision_score(y_true=flattened_y, y_pred=flattened_y_pred, average=None)
        )
        recall = np.asarray(
            recall_score(y_true=flattened_y, y_pred=flattened_y_pred, average=None)
        )
        return cls(
            weights=weights,
            loss=loss,
            iou=iou,
            dice=dice,
            f1=f1,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
        )

    def to_dict(
        self,
        split_labels: bool = False,
        labels: Collection[str] | None = None,
        prefix: str | None = None,
    ) -> dict[str, typing.Any]:
        d = asdict(self)
        if not split_labels:
            return d
        elif labels is None:
            raise ValueError("No labels have been supplied")
        for k in tuple(d.keys()):
            v = d.pop(k)
            if isinstance(v, np.ndarray):
                v = v.tolist()
                if len(v) != len(labels):
                    raise ValueError("An incorrect number of labels have been supplied")
                for i, label in enumerate(labels):
                    d[f"{label}_{k}"] = float(v[i])
        if prefix is not None:
            for k in tuple(d.keys()):
                d[f"{prefix}_{k}"] = d.pop(k)
        return d
