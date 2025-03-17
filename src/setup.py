from __future__ import annotations
import logging
import typing
from dataclasses import dataclass, field, InitVar, asdict

import numpy as np
import pandas as pd

import torch
# from ignite import metrics as ignite_metrics

from monai.utils.misc import first
from monai import data, transforms, losses, optimizers, metrics, inferers
# from monai.handlers import (
#     from_engine,
# )

from augmentations import get_transform_list
from utils import MONAI_KEYS


if typing.TYPE_CHECKING:
    from os import PathLike
    from collections.abc import Callable, Sequence
    from matplotlib.axes import Axes
    from numpy.typing import NDArray


_logger = logging.getLogger("adaptive_milling_training")


def create_datasets(
    input_data: NDArray[np.str_] | pd.DataFrame,
    image_size: int,
    label_count: int,
    validation_split: float = 0.2,
    foreground_labels: Sequence[int] | None = None,
    *,
    label_changes: list[tuple[int, int]] = [],
    dataset_type: data.Dataset = data.Dataset,
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
        dataset_type(
            data=train,
            transform=transforms.Compose(
                get_transform_list(
                    image_size,
                    label_count=label_count,
                    training=True,
                    label_changes=label_changes,
                    foreground_labels=foreground_labels,
                    **transform_kwargs,
                )
            ),
        ),
        dataset_type(
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


def get_device(cpu_only: bool = False, gpu: int | None = None) -> torch.device:
    gpu_str = "cuda"
    if gpu is not None:
        gpu_str += f":{gpu}"
    return torch.device(
        gpu_str if not cpu_only and torch.cuda.is_available() else "cpu"
    )


@dataclass
class TrainingParameters:
    num_classes: int
    label_names: tuple[str, ...]
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
    foreground_labels: tuple[int, ...] = field(default_factory=tuple)
    loss_weights: tuple[float, ...] | None = None
    current_metrics: dict[str, dict[str, float]] = field(init=False)
    best_metrics: dict[str, dict[str, float]] = field(init=False)

    def __post_init__(self):
        self.current_metrics = {"train": {}, "val": {}}
        self.best_metrics = {"train": {}, "val": {}}

    def update_metrics(
        self,
        epoch: int,
        metrics_dict: dict[str, float | NDArray[typing.Any]],
        stage: typing.Literal["train", "val"],
    ) -> bool:
        is_best = False
        metrics_dict["epoch"] = epoch

        self.current_metrics[stage] = metrics_dict
        if not self.best_metrics[stage] or np.mean(
            self.current_metrics[stage][self.best_metric]
        ) > np.mean(self.best_metrics[stage][self.best_metric]):
            is_best = True
            _logger.info("New best %s epoch found", stage)
            self.best_metrics[stage] = self.current_metrics[stage]

        logged_metrics = self.current_metrics[stage].copy()
        best_epoch = logged_metrics.pop("epoch")

        _logger.info(
            "Current epoch: %i, stage: %s\n%s\nBest %s %s: %.4f at epoch %i",
            epoch + 1,
            stage,
            "\n".join(
                (
                    f"{name}_mean: {np.mean(values):.4f}"
                    for name, values in logged_metrics.items()
                )
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
    device: torch.device
    model: torch.nn.Module
    loss_function: losses._Loss
    optimizer: torch.optim.Optimizer
    train_metrics: dict[str, metrics.Metric]
    val_metrics: dict[str, metrics.Metric]
    grad_scaler: torch.GradScaler | None = None
    post_train_transform: transforms.Transform | Callable = lambda x: x
    post_train_label_transform: transforms.Transform | Callable = lambda x: x
    post_val_transform: transforms.Transform | Callable = lambda x: x
    post_val_label_transform: transforms.Transform | Callable = lambda x: x
    training_inferer: inferers.Inferer = field(default_factory=inferers.SimpleInferer)
    validation_inferer: inferers.Inferer = field(default_factory=inferers.SimpleInferer)
    training_dataloader: data.Dataloader = field(init=False)
    validation_dataloader: data.Dataloader = field(init=False)
    lr_scheduler: torch._LRScheduler | None = field(init=False)
    # InitVars:
    training_data_workers: InitVar[int] = 8
    validation_data_workers: InitVar[int] = 4
    training_batch_size: InitVar[int] = 4
    validation_batch_size: InitVar[int] = 1
    check_loaders: InitVar[bool] = True
    lr_scheduler_class: InitVar[type[torch._LRScheduler] | None] = (
        optimizers.WarmupCosineSchedule
    )
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
        if lr_scheduler_kwargs is None:
            lr_scheduler_kwargs = {}

        if lr_scheduler_class is None:
            self.lr_scheduler = None
        else:
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

    def asdict(self) -> dict[str, typing.Any]:
        return asdict(self)


def setup_training_objects(
    device: torch.device,
    training_data: data.Dataset,
    validation_data: data.Dataset,
    num_classes: int,
    model: torch.nn.Module,
    loss_function: losses._Loss,
    learning_rate: float = 1e-4,
    include_background: bool = False,
    **kwargs: typing.Any,
) -> TrainingObjects:
    train_metrics = {
        "mean_iou": metrics.MeanIoU(
            include_background=include_background,
            reduction="mean",
        ),
        "mean_dice": metrics.DiceMetric(
            include_background=include_background,
            reduction="mean_batch",
        ),
    }

    val_metrics = {
        "mean_iou": metrics.MeanIoU(
            include_background=include_background,
            reduction="mean_batch",
        ),
        "mean_dice": metrics.DiceMetric(
            include_background=include_background,
            reduction="mean_batch",
        ),
    }

    labels_to_keep = tuple(range(1 - int(include_background), num_classes))

    post_train_transform = transforms.Compose(
        [
            # transforms.Activations(softmax=True),
            transforms.AsDiscrete(
                argmax=True,
                to_onehot=num_classes,
                # dim=1,
                # keepdim=True,
                # dtype=torch.long,
            ),
            transforms.LabelToMask(labels_to_keep),
            # transforms.EnsureType(dtype=torch.long),
        ]
    )

    post_train_label_transform = transforms.Compose(
        [
            transforms.AsDiscrete(
                # argmax=True,
                to_onehot=num_classes,
                # dim=1,
                # keepdim=True,
                # dtype=torch.long,
            ),
            transforms.LabelToMask(labels_to_keep),
            # transforms.EnsureType(dtype=torch.long),
        ]
    )

    post_val_transform = transforms.Compose(
        [
            # transforms.Activations(softmax=True),
            # ArgMax(dim=1),
            transforms.AsDiscrete(
                argmax=True,
                to_onehot=num_classes,
            ),
            transforms.LabelToMask(labels_to_keep),
        ]
    )

    post_val_label_transform = transforms.Compose(
        [
            # transforms.Activations(softmax=True),
            transforms.AsDiscrete(
                # argmax=True,
                to_onehot=num_classes,
            ),
            transforms.LabelToMask(labels_to_keep),
        ]
    )

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    grad_scaler = torch.GradScaler(device=device)
    # grad_scaler = None

    training_batch_size = 6
    validation_batch_size = 3

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
        post_train_label_transform=post_train_label_transform,
        post_val_transform=post_val_transform,
        post_val_label_transform=post_val_label_transform,
        training_batch_size=training_batch_size,
        validation_batch_size=validation_batch_size,
        training_data_workers=training_batch_size * 2,
        validation_data_workers=validation_batch_size * 2,
        check_loaders=False,
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
        persistent_workers=True,  # Avoids issues when also submitting images via MLFlow
    )

    validation_dataloader = data.DataLoader(
        training_objects.validation_data,
        batch_size=validation_batch_size,
        shuffle=True,
        num_workers=validation_workers,
        pin_memory=pin_memory,
        persistent_workers=True,  # Avoids issues when also submitting images via MLFlow
    )
    return training_dataloader, validation_dataloader


def find_learning_rate(
    ax: Axes,
    training_objects: TrainingObjects,
    lower_learning_rate: float = 1e-7,
    upper_learning_rate: float = 1e-2,
    iterations: int = 20,
    amp: bool = True,
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
        amp=amp,
    )
    # for grad, loss in zip(*lr_finder.get_lrs_and_losses())
    #     print(f"Gradient, loss: {grad}, {loss}")
    sg, sg_loss = lr_finder.get_steepest_gradient()
    msg = f"Steepest gradient: {sg:2e}, loss: {sg_loss:2e}"
    print(msg)
    _ = lr_finder.plot(ax=ax)
    ax.set_title(msg)
