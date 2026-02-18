from __future__ import annotations
import logging
import typing
from dataclasses import dataclass, field, InitVar, asdict

import torch

from monai.utils.misc import first
from monai import data, transforms, inferers

from ap_model_training.metrics import Metrics

if typing.TYPE_CHECKING:
    from os import PathLike
    from collections.abc import Callable, Mapping

_logger = logging.getLogger(__name__)


@dataclass
class EvaluationParameters:
    output_path: str | PathLike[str]
    run_id: str
    weights_file: str | PathLike[str]
    num_classes: int
    num_channels: int
    label_names: tuple[str, ...]
    input_image_shape: tuple[int, int]
    total_data: int
    batch_size: int
    key_metrics: list[str]
    foreground_labels: tuple[int, ...] = field(default_factory=tuple)
    loss_weights: tuple[float, ...] | None = None
    metrics: dict[str, float] = field(init=False)
    include_background: bool = False
    seed: int = 42

    def update_metrics(
        self,
        epoch: int,
        metrics: Mapping[str, float],
    ) -> None:
        metrics_dict = dict(metrics)
        metrics_dict["epoch"] = epoch
        self.metrics = metrics_dict

    def asdict(self) -> dict[str, typing.Any]:
        return asdict(self)


@dataclass
class EvaluationObjects:
    data: data.Dataset
    device: torch.device
    model: torch.nn.Module
    loss_function: torch.nn.Module
    metrics: Metrics
    post_transform: transforms.Transform | Callable = lambda x: x
    inferer: inferers.Inferer = field(default_factory=inferers.SimpleInferer)
    dataloader: data.dataloader.DataLoader = field(init=False)
    # InitVars:
    data_workers: InitVar[int] = 4
    batch_size: InitVar[int] = 1
    check_loaders: InitVar[bool] = True

    def __post_init__(
        self,
        data_workers: int,
        batch_size: int,
        check_loaders: bool,
    ) -> None:
        self.training_dataloader, self.validation_dataloader = self.load_data(
            workers=data_workers,
            batch_size=batch_size,
            check_data_loads=check_loaders,
        )

    def asdict(self) -> dict[str, typing.Any]:
        return asdict(self)

    def load_data(
        self,
        workers: int,
        batch_size: int,
        check_data_loads: bool = True,
    ) -> data.dataloader.DataLoader:
        pin_memory = self.device.type == "cuda"

        if check_data_loads:
            # Check data loads
            check_loader = data.DataLoader(
                self.data,
                batch_size=10,
                num_workers=2,
                pin_memory=pin_memory,
            )
            first_batch = first(check_loader)
            assert first_batch is not None, "DataLoader check failed"

        return data.dataloader.DataLoader(
            self.data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=pin_memory,
            persistent_workers=True,  # Avoids issues when also submitting images via MLFlow
        )


def setup_evaulation_objects(
    device: torch.device,
    data: data.Dataset,
    num_classes: int,
    model: torch.nn.Module,
    loss_function: torch.nn.Module,
    batch_size: int,
    num_workers: int | None = None,
    **kwargs: typing.Any,
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

    return EvaluationObjects(
        data,
        device=device,
        model=model,
        loss_function=loss_function,
        metrics=metrics,
        post_transform=post_transform,
        data_workers=num_workers,
        batch_size=batch_size,
        check_loaders=False,
        **kwargs,
    )
