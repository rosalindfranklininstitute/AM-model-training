from __future__ import annotations
import logging
import typing
from dataclasses import dataclass, field, asdict

import torch

from monai import transforms, inferers

from ap_model_training.metrics import Metrics
from ap_model_training.setup.stage import StageObjects
from ap_model_training.setup.abstract import (
    _AbstractLossObjects,
    _AbstractModelObjects,
)
from ap_model_training._version import __version__

if typing.TYPE_CHECKING:
    from os import PathLike
    from collections.abc import Mapping

    from monai import data

_logger = logging.getLogger(__name__)


@dataclass
class EvaluationParameters:
    output_path: str | PathLike[str]
    run_id: str
    weights_file: str | PathLike[str]
    num_classes: int
    num_channels: int
    label_names: tuple[str, ...]
    input_image_shape: int
    total_data: int
    batch_size: int
    key_metrics: list[str]
    loss_weights: tuple[float, ...] | None = None
    metrics: dict[str, float] = field(init=False)
    include_background: bool = False
    package_version: str = field(init=False)

    def __post_init__(self):
        self.output_path = str(self.output_path)  # Ensure JSON serializable
        self.weights_file = str(self.weights_file)
        self.package_version = __version__

    def update_metrics(
        self,
        metrics: Mapping[str, float],
    ) -> None:
        self.metrics = dict(metrics)

    def asdict(self, include_metrics: bool = False) -> dict[str, typing.Any]:
        d = asdict(self)
        if not include_metrics:
            del d["metrics"]
        return d


@dataclass
class EvaluationObjects(_AbstractLossObjects, _AbstractModelObjects):
    validation: StageObjects

    def asdict(self) -> dict[str, typing.Any]:
        return asdict(self)


def setup_evaluation_objects(
    device: torch.device,
    data: data.Dataset,
    num_classes: int,
    model: torch.nn.Module,
    loss_function: torch.nn.Module,
    batch_size: int,
    num_workers: int | None = None,
) -> EvaluationObjects:
    metrics = Metrics(device=device, num_classes=num_classes)

    post_transform = transforms.Compose(
        [
            transforms.AsDiscrete(
                argmax=True,
                dtype=torch.long,
            ),
        ]
    )

    model = model.to(device)

    if num_workers is None:
        num_workers = batch_size * 4
    validation_objects = StageObjects(
        metrics=metrics,
        data=data,
        device=device,
        data_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        check_loaders=False,
        inferer=inferers.SimpleInferer(),
        post_transform=post_transform,
    )

    return EvaluationObjects(
        device=device,
        model=model,
        loss_function=loss_function,
        validation=validation_objects,
    )
