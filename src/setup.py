from __future__ import annotations
import logging
import typing
from dataclasses import dataclass, field, InitVar, asdict

import numpy as np
import pandas as pd

import torch
# from ignite import metrics as ignite_metrics

from monai.utils.misc import first
from monai import data, transforms, losses, optimizers, metrics
# from monai.handlers import (
#     from_engine,
# )

from augmentations import get_transform_list
from utils import MONAI_KEYS


if typing.TYPE_CHECKING:
    from os import PathLike
    from collections.abc import Callable
    from matplotlib.axes import Axes
    from numpy.typing import NDArray


_logger = logging.getLogger("adaptive_milling_training")


def create_datasets(
    input_data: NDArray[np.str_] | pd.DataFrame,
    image_size: int,
    label_count: int,
    validation_split: float = 0.2,
    *,
    label_changes: list[tuple[int, int]] = [],
    **transform_kwargs: typing.Any,
) -> tuple[data.Dataset, data.Dataset]:
    if isinstance(input_data, np.ndarray):
        datalist = [
            {MONAI_KEYS.IMAGE: _[0], MONAI_KEYS.LABEL: _[1]} for _ in input_data
        ]
    elif isinstance(input_data, pd.DataFrame):
        datalist = input_data.to_dict(orient="records")
    else:
        raise TypeError(f"Unsupported data type '{type(data)}'")
    train, validate = data.partition_dataset(
        datalist,
        ratios=(1 - validation_split, validation_split),
        num_partitions=2,
        shuffle=True,
    )

    return (
        data.Dataset(
            data=train,
            transform=transforms.Compose(
                get_transform_list(
                    image_size,
                    label_count=label_count,
                    training=True,
                    label_changes=label_changes,
                    **transform_kwargs,
                )
            ),
        ),
        data.Dataset(
            data=validate,
            transform=transforms.Compose(
                get_transform_list(
                    image_size,
                    label_count=label_count,
                    training=False,
                    label_changes=label_changes,
                    **transform_kwargs,
                )
            ),
        ),
    )


def get_device(cpu_only: bool = False) -> torch.device:
    return torch.device("cuda" if not cpu_only and torch.cuda.is_available() else "cpu")


@dataclass
class TrainingParameters:
    num_classes: int
    input_image_shape: tuple[int, int]
    learning_rate: float
    best_metric: str
    max_epochs: int
    model_path: str | PathLike[str]
    train_patience: int
    val_patience: int
    total_training_data: int
    training_batch_size: int
    total_validation_data: int
    validation_batch_size: int
    key_train_metrics: list[str]
    key_val_metrics: list[str]
    frozen_epochs: int = 0
    loss_weights: tuple[float, ...] | None = None
    current_metrics: dict[str, dict[str, float]] = field(init=False)
    best_metrics: dict[str, dict[str, float]] = field(init=False)

    def __post_init__(self):
        self.current_metrics = {"train": {}, "val": {}}
        self.best_metrics = {"train": {}, "val": {}}

    def update_metrics(
        self,
        epoch: int,
        metrics_dict: dict[str, float],
        stage: typing.Literal["train", "val"],
    ) -> bool:
        is_best = False
        metrics_dict["epoch"] = epoch

        self.current_metrics[stage] = metrics_dict
        if (
            not self.best_metrics[stage]
            or self.current_metrics[stage][self.best_metric]
            > self.best_metrics[stage][self.best_metric]
        ):
            is_best = True
            _logger.info("New best %s epoch found", stage)
            self.best_metrics = self.current_metrics

        logged_metrics = self.current_metrics[stage].copy()
        best_epoch = logged_metrics.pop("epoch")

        _logger.info(
            "Current epoch: %i, stage: %s\n%s\nBest %s %s: %.4f at epoch %i",
            epoch + 1,
            stage,
            "\n".join(
                (f"{name}: {value:.4f}" for name, value in logged_metrics.items())
            ),
            stage,
            self.best_metric,
            logged_metrics[self.best_metric],
            best_epoch,
        )
        return is_best

    def asdict(self, include_metrics: bool = False) -> dict[str, typing.Any]:
        d = asdict(self)
        if not include_metrics:
            del d["current_metrics"]
            del d["best_metrics"]
        return d


@dataclass
class TrainingObjects:
    training_data: data.Dataset
    validation_data: data.Dataset
    device: torch.DeviceObjType
    model: torch.nn.Module
    loss_function: losses._Loss
    optimizer: torch.optim.Optimizer
    grad_scaler: torch.GradScaler
    train_metrics: dict[str, metrics.Metric]
    val_metrics: dict[str, metrics.Metric]
    post_train_transform: transforms.Transform | Callable = lambda x: x
    post_val_transform: transforms.Transform | Callable = lambda x: x
    training_data_workers: InitVar[int] = 8
    validation_data_workers: InitVar[int] = 4
    training_batch_size: InitVar[int] = 4
    validation_batch_size: InitVar[int] = 1
    training_dataloader: data.Dataloader = field(init=False)
    validation_dataloader: data.Dataloader = field(init=False)
    num_training_data: int = field(init=False)
    num_validation_data: int = field(init=False)
    check_loaders: InitVar[bool] = True
    lr_scheduler_class: InitVar[type[torch._LRScheduler]] = (
        optimizers.WarmupCosineSchedule
    )
    lr_scheduler: torch._LRScheduler = field(init=False)
    lr_scheduler_kwargs: InitVar[dict[str, typing.Any] | None] = None

    def __post_init__(
        self,
        training_data_workers: int,
        validation_data_workers: int,
        training_batch_size: int,
        validation_batch_size: int,
        check_loaders: bool,
        lr_scheduler_class: type[torch._LRScheduler],
        lr_scheduler_kwargs: dict[str, typing.Any] | None,
    ) -> None:
        self.num_training_data = len(self.training_data)

        self.num_validation_data = len(self.validation_data)

        if lr_scheduler_kwargs is None:
            lr_scheduler_kwargs = {}

        self.lr_scheduler = lr_scheduler_class(
            optimizer=self.optimizer, **lr_scheduler_kwargs
        )

        self.training_dataloader, self.validation_dataloader = load_data(
            self,
            training_batch_size=training_batch_size,
            validation_batch_size=validation_batch_size,
            check_data_loads=check_loaders,
            training_workers=training_data_workers,
            validation_workers=validation_data_workers,
        )


def setup_training_objects(
    device: torch.DeviceObjType,
    training_data: data.Dataset,
    validation_data: data.Dataset,
    num_classes: int,
    model: torch.nn.Module,
    loss_function: losses._Loss,
    learning_rate: float = 1e-4,
    **kwargs: typing.Any,
) -> TrainingObjects:
    train_metrics = {
        "mean_iou": metrics.MeanIoU(
            include_background=True,
            reduction="mean",
        ),
        "mean_dice": metrics.DiceMetric(
            include_background=True,
            reduction="mean",
        ),
    }

    val_metrics = {
        "mean_iou": metrics.MeanIoU(
            include_background=True,
            reduction="mean",
        ),
        "mean_dice": metrics.DiceMetric(
            include_background=True,
            reduction="mean",
        ),
    }

    post_train_transform = transforms.Compose(
        [
            transforms.EnsureType(),
            transforms.Activations(softmax=True),
            transforms.AsDiscrete(
                argmax=True,
                to_onehot=num_classes,
            ),
        ]
    )

    post_val_transform = transforms.Compose(
        [
            transforms.EnsureType(),
            transforms.Activations(softmax=True),
            transforms.AsDiscrete(
                argmax=True,
                to_onehot=num_classes,
            ),
        ]
    )

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    grad_scaler = torch.GradScaler(device=device)

    return TrainingObjects(
        training_data,
        validation_data,
        device=device,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        grad_scaler=grad_scaler,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        post_train_transform=post_train_transform,
        post_val_transform=post_val_transform,
        **kwargs,
    )


def load_data(
    training_objects: TrainingObjects,
    training_batch_size: int,
    validation_batch_size: int,
    check_data_loads: bool = True,
    training_workers: int = 4,
    validation_workers: int = 1,
) -> tuple[data.DataLoader, data.DataLoader]:
    pin_memory = training_objects.device.type == "cuda"

    if check_data_loads:
        # Check data loads
        check_loader = data.DataLoader(
            training_objects.training_data,
            batch_size=10,
            num_workers=2,
            pin_memory=pin_memory,
        )
        first_batch = first(check_loader)
        assert first_batch is not None, "DataLoader check failed"

    training_dataloader = data.DataLoader(
        training_objects.training_data,
        batch_size=training_batch_size,
        shuffle=True,
        num_workers=training_workers,
        pin_memory=pin_memory,
    )

    validation_dataloader = data.DataLoader(
        training_objects.validation_data,
        batch_size=validation_batch_size,
        shuffle=True,
        num_workers=validation_workers,
        pin_memory=pin_memory,
    )
    return training_dataloader, validation_dataloader


def find_learning_rate(
    ax: Axes,
    training_objects: TrainingObjects,
    lower_learning_rate: float = 1e-6,
    upper_learning_rate: float = 1e-2,
    iterations: int = 20,
) -> None:
    lr_finder = optimizers.LearningRateFinder(
        model=training_objects.model,
        optimizer=training_objects.optimizer,
        criterion=training_objects.loss_function,
        device=training_objects.device,
    )
    lr_finder.range_test(
        training_objects.training_dataloader,
        training_objects.validation_dataloader,
        start_lr=lower_learning_rate,
        end_lr=upper_learning_rate,
        num_iter=iterations,
    )
    # for grad, loss in zip(*lr_finder.get_lrs_and_losses())
    #     print(f"Gradient, loss: {grad}, {loss}")
    sg, sg_loss = lr_finder.get_steepest_gradient()
    msg = f"Steepest gradient: {sg:2e}, loss: {sg_loss:2e}"
    print(msg)
    _ = lr_finder.plot(ax=ax)
    ax.set_title(msg)
