from __future__ import annotations
import logging
import typing
from dataclasses import dataclass, field, asdict

import numpy as np

import torch
from torch.amp.grad_scaler import GradScaler

from monai import data, transforms, inferers

from ap_model_training.schedulers import lr_scheduler_creation_functions
from ap_model_training.metrics import Metrics
from ap_model_training.setup.stage import StageObjects
from ap_model_training.setup.abstract import (
    _AbstractLossObjects,
    _AbstractModelObjects,
    _AbstractTrainObjects,
)

if typing.TYPE_CHECKING:
    from os import PathLike
    from collections.abc import Mapping

_logger = logging.getLogger(__name__)


@dataclass
class TrainingParameters:
    output_path: str | PathLike[str]
    run_id: str
    model_name: str
    num_classes: int
    num_channels: int
    label_names: tuple[str, ...]
    input_image_shape: int
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
    loss_weights: tuple[float, ...] | None = None
    current_metrics: dict[str, dict[str, float]] = field(init=False)
    best_metrics: dict[str, dict[str, float]] = field(init=False)
    include_background: bool = False
    val_interval: int = 2
    seed: int = 42

    def __post_init__(self):
        self.output_path = str(self.output_path)  # Ensure JSON serializable
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
            epoch,
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
class TrainingObjects(
    _AbstractLossObjects, _AbstractTrainObjects, _AbstractModelObjects
):
    training: StageObjects
    validation: StageObjects

    def asdict(self) -> dict[str, typing.Any]:
        return asdict(self)


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

    training_objects = StageObjects(
        device=device,
        data=training_data,
        inferer=inferers.SimpleInferer(),
        post_transform=post_transform,
        data_workers=num_training_workers,
        batch_size=training_batch_size,
        shuffle=True,
        check_loaders=False,
        metrics=train_metrics,
    )
    validation_objects = StageObjects(
        device=device,
        data=validation_data,
        inferer=inferers.SimpleInferer(),
        post_transform=post_transform,
        data_workers=num_validation_workers,
        batch_size=validation_batch_size,
        shuffle=False,
        check_loaders=False,
        metrics=val_metrics,
    )

    return TrainingObjects(
        device=device,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        grad_scaler=grad_scaler,
        loss_function=loss_function,
        training=training_objects,
        validation=validation_objects,
    )
