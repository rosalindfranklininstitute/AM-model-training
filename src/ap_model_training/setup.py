from __future__ import annotations
import logging
import typing
from dataclasses import dataclass, field, InitVar, asdict
from collections.abc import Sequence

import numpy as np
import pandas as pd

import torch
from torch.amp.grad_scaler import GradScaler

from monai.utils.misc import first
from monai import data, transforms, losses, optimizers, inferers

from ap_model_training.schedulers import lr_scheduler_creation_functions
from ap_model_training.augmentations import get_transform_list
from ap_model_training.utils import MONAI_KEYS
from ap_model_training.metrics import Metrics


if typing.TYPE_CHECKING:
    from os import PathLike
    from collections.abc import Callable, Mapping
    from matplotlib.axes import Axes
    from numpy.typing import NDArray


_logger = logging.getLogger("adaptive_milling_training")


def create_dataset(
    input_data: NDArray[np.str_] | pd.DataFrame | Sequence,
    image_size: int,
    augmentations: bool,
    *,
    pad: bool = True,
    rgb: bool = True,
    dataset_type: type[data.Dataset] = data.Dataset,
    **transform_kwargs: typing.Any,
) -> data.Dataset:
    datalist: Sequence
    if isinstance(input_data, np.ndarray):
        datalist = [
            {MONAI_KEYS.IMAGE: _[0], MONAI_KEYS.LABEL: _[1]} for _ in input_data
        ]
    elif isinstance(input_data, pd.DataFrame):
        datalist = input_data.to_dict(orient="records")
    elif isinstance(input_data, Sequence):
        datalist = input_data
    else:
        raise TypeError(f"Unsupported data type '{type(data)}'")
    return dataset_type(
        data=datalist,
        transform=transforms.Compose(
            get_transform_list(
                image_size=image_size,
                augmentations=augmentations,
                pad=pad,
                rgb=rgb,
                **transform_kwargs,
            )
        ),
    )


def create_datasets(
    input_data: NDArray[np.str_] | pd.DataFrame,
    image_size: int,
    validation_split: float = 0.2,
    *,
    pad: bool = True,
    rgb: bool = True,
    dataset_type: type[data.Dataset] = data.Dataset,
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
        seed=42,
        shuffle=False,
    )

    return (
        create_dataset(
            input_data=train,
            image_size=image_size,
            augmentations=True,
            pad=pad,
            rgb=rgb,
            dataset_type=dataset_type,
            **transform_kwargs,
        ),
        create_dataset(
            input_data=validate,
            image_size=image_size,
            augmentations=False,
            pad=pad,
            rgb=rgb,
            dataset_type=dataset_type,
            **transform_kwargs,
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
    num_channels: int
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
    include_background: bool = False
    val_interval: int = 2

    def __post_init__(self):
        self.current_metrics = {"train": {}, "val": {}}
        self.best_metrics = {"train": {}, "val": {}}

    def update_metrics(
        self,
        epoch: int,
        metrics: Mapping[str, float],
        stage: typing.Literal["train", "val"],
    ) -> bool:
        is_best = False
        metrics_dict = dict(metrics)
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
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    train_metrics: Metrics
    val_metrics: Metrics
    grad_scaler: GradScaler | None = None
    post_train_transform: transforms.Transform | Callable = lambda x: x
    post_val_transform: transforms.Transform | Callable = lambda x: x
    training_inferer: inferers.Inferer = field(default_factory=inferers.SimpleInferer)
    validation_inferer: inferers.Inferer = field(default_factory=inferers.SimpleInferer)
    training_dataloader: data.dataloader.DataLoader = field(init=False)
    validation_dataloader: data.dataloader.DataLoader = field(init=False)
    # InitVars:
    training_data_workers: InitVar[int] = 4
    validation_data_workers: InitVar[int] = 4
    training_batch_size: InitVar[int] = 2
    validation_batch_size: InitVar[int] = 2
    check_loaders: InitVar[bool] = True

    def __post_init__(
        self,
        training_data_workers: int,
        validation_data_workers: int,
        training_batch_size: int,
        validation_batch_size: int,
        check_loaders: bool,
    ) -> None:
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
    training_batch_size: int,
    validation_batch_size: int,
    learning_rate: float = 1e-4,
    lr_scheduler_name: str = "onecyclelr",
    lr_scheduler_kwargs: dict[str, typing.Any] | None = None,
    **kwargs: typing.Any,
) -> TrainingObjects:
    if lr_scheduler_kwargs is None:
        lr_scheduler_kwargs = {}

    train_metrics = Metrics(device=device, num_classes=num_classes)

    val_metrics = Metrics(device=device, num_classes=num_classes)

    post_train_transform = transforms.Compose(
        [
            transforms.AsDiscrete(
                argmax=True,
                dtype=torch.long,
            ),
        ]
    )

    post_val_transform = transforms.Compose(
        [
            transforms.AsDiscrete(
                argmax=True,
                dtype=torch.long,
            ),
        ]
    )

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    lr_scheduler = lr_scheduler_creation_functions[lr_scheduler_name](
        optimizer, **lr_scheduler_kwargs
    )

    grad_scaler = GradScaler(device=device.type)
    # grad_scaler = None

    return TrainingObjects(
        training_data,
        validation_data,
        device=device,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        grad_scaler=grad_scaler,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        post_train_transform=post_train_transform,
        post_val_transform=post_val_transform,
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
) -> tuple[data.dataloader.DataLoader, data.dataloader.DataLoader]:
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

    training_dataloader = data.dataloader.DataLoader(
        training_objects.training_data,
        batch_size=training_batch_size,
        shuffle=True,
        num_workers=training_workers,
        pin_memory=pin_memory,
        persistent_workers=True,  # Avoids issues when also submitting images via MLFlow
    )

    validation_dataloader = data.dataloader.DataLoader(
        training_objects.validation_data,
        batch_size=validation_batch_size,
        shuffle=False,
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
    )
    # for grad, loss in zip(*lr_finder.get_lrs_and_losses())
    #     print(f"Gradient, loss: {grad}, {loss}")
    sg, sg_loss = lr_finder.get_steepest_gradient()
    msg = f"Steepest gradient: {sg:2e}, loss: {sg_loss:2e}"
    print(msg)
    _ = lr_finder.plot(ax=ax)
    ax.set_title(msg)
