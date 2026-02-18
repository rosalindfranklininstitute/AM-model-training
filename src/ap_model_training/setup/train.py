from __future__ import annotations
import logging
import typing
from dataclasses import dataclass, field, InitVar, asdict

import numpy as np

import torch
from torch.amp.grad_scaler import GradScaler

from monai.utils.misc import first
from monai import data, transforms, inferers

from ap_model_training.schedulers import lr_scheduler_creation_functions
from ap_model_training.metrics import Metrics

if typing.TYPE_CHECKING:
    from os import PathLike
    from collections.abc import Callable, Mapping

_logger = logging.getLogger(__name__)


@dataclass
class TrainingParameters:
    output_path: str | PathLike[str]
    run_id: str
    model_name: str
    num_classes: int
    num_channels: int
    label_names: tuple[str, ...]
    input_image_shape: tuple[int, int]
    learning_rate: float
    best_metric: str
    max_epochs: int
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
    seed: int = 42

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
    loss_function: torch.nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    train_metrics: Metrics
    val_metrics: Metrics
    grad_scaler: GradScaler | None = None
    post_transform: transforms.Transform | Callable = lambda x: x
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
        self.training_dataloader, self.validation_dataloader = self.load_data(
            check_data_loads=check_loaders,
            training_workers=training_data_workers,
            training_batch_size=training_batch_size,
            validation_batch_size=validation_batch_size,
            validation_workers=validation_data_workers,
        )

    def asdict(self) -> dict[str, typing.Any]:
        return asdict(self)

    def load_data(
        self,
        training_workers: int,
        validation_workers: int,
        training_batch_size: int,
        validation_batch_size: int,
        check_data_loads: bool = True,
    ) -> tuple[data.dataloader.DataLoader, data.dataloader.DataLoader]:
        pin_memory = self.device.type == "cuda"

        if check_data_loads:
            # Check data loads
            check_loader = data.DataLoader(
                self.training_data,
                batch_size=10,
                num_workers=2,
                pin_memory=pin_memory,
            )
            first_batch = first(check_loader)
            assert first_batch is not None, "DataLoader check failed"

        training_dataloader = data.dataloader.DataLoader(
            self.training_data,
            batch_size=training_batch_size,
            shuffle=True,
            num_workers=training_workers,
            pin_memory=pin_memory,
            persistent_workers=True,  # Avoids issues when also submitting images via MLFlow
        )

        validation_dataloader = data.dataloader.DataLoader(
            self.validation_data,
            batch_size=validation_batch_size,
            shuffle=False,
            num_workers=validation_workers,
            pin_memory=pin_memory,
            persistent_workers=True,  # Avoids issues when also submitting images via MLFlow
        )
        return training_dataloader, validation_dataloader


def setup_training_objects(
    device: torch.device,
    training_data: data.Dataset,
    validation_data: data.Dataset,
    num_classes: int,
    model: torch.nn.Module,
    loss_function: torch.nn.Module,
    training_batch_size: int,
    validation_batch_size: int,
    num_training_workers: int | None = None,
    num_validation_workers: int | None = None,
    learning_rate: float = 1e-4,
    lr_scheduler_name: str = "onecyclelr",
    lr_scheduler_kwargs: dict[str, typing.Any] | None = None,
    **kwargs: typing.Any,
) -> TrainingObjects:
    if lr_scheduler_kwargs is None:
        lr_scheduler_kwargs = {}

    train_metrics = Metrics(device=device, num_classes=num_classes)

    val_metrics = Metrics(device=device, num_classes=num_classes)

    post_transform = transforms.Compose(
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

    if num_training_workers is None:
        num_training_workers = training_batch_size * 4
    if num_validation_workers is None:
        num_validation_workers = validation_batch_size * 4

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
        post_transform=post_transform,
        training_data_workers=num_training_workers,
        validation_data_workers=num_validation_workers,
        training_batch_size=training_batch_size,
        validation_batch_size=validation_batch_size,
        check_loaders=False,
        **kwargs,
    )
