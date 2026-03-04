from __future__ import annotations
import logging
import typing
from dataclasses import dataclass, asdict

import torch

from monai import transforms, inferers

from ap_model_training.setup.abstract import (
    _AbstractInferenceObjects,
    _AbstractModelObjects,
)

if typing.TYPE_CHECKING:
    from os import PathLike

    from monai import data

_logger = logging.getLogger(__name__)


@dataclass
class InferenceParameters:
    output_path: str | PathLike[str]
    run_id: str
    weights_file: str | PathLike[str]
    num_classes: int
    num_channels: int
    label_names: tuple[str, ...]
    input_image_shape: int
    total_data: int
    batch_size: int
    include_background: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        self.output_path = str(self.output_path)  # Ensure JSON serializable
        self.weights_file = str(self.weights_file)  # Ensure JSON serializable

    def asdict(self) -> dict[str, typing.Any]:
        return asdict(self)


@dataclass
class InferenceObjects(_AbstractInferenceObjects, _AbstractModelObjects):
    device: torch.device

    def asdict(self) -> dict[str, typing.Any]:
        return asdict(self)


def setup_inference_objects(
    device: torch.device,
    data: data.Dataset,
    model: torch.nn.Module,
    batch_size: int,
    num_workers: int | None = None,
) -> InferenceObjects:

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
    return InferenceObjects(
        device=device,
        model=model,
        data=data,
        data_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        check_loaders=False,
        inferer=inferers.SimpleInferer(),
        post_transform=post_transform,
    )
