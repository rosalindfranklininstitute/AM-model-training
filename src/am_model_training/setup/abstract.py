from __future__ import annotations
import logging
import typing
from abc import ABC
from dataclasses import dataclass, field, InitVar, asdict

import torch
from torch.amp.grad_scaler import GradScaler

from monai.utils.misc import first
from monai import data, transforms, inferers

from am_model_training.metrics import Metrics

if typing.TYPE_CHECKING:
    from collections.abc import Callable

_logger = logging.getLogger(__name__)


@dataclass
class _AbstractModelObjects(ABC):
    device: torch.device
    model: torch.nn.Module


@dataclass
class DataObjects:
    data: data.Dataset

    # InitVars:
    device: InitVar[torch.device]
    data_workers: InitVar[int]
    batch_size: InitVar[int]
    shuffle: InitVar[bool]
    check_loaders: InitVar[bool]

    dataloader: data.DataLoader = field(init=False)


@dataclass
class _AbstractInferenceObjects:
    data: data.Dataset

    # InitVars:
    device: InitVar[torch.device]
    data_workers: InitVar[int]
    batch_size: InitVar[int]
    shuffle: InitVar[bool]
    check_loaders: InitVar[bool]

    inferer: inferers.Inferer = field(default_factory=inferers.SimpleInferer)
    post_transform: transforms.Transform | Callable = lambda x: x
    dataloader: data.DataLoader = field(init=False)

    def __post_init__(
        self,
        device: torch.device,
        data_workers: int,
        batch_size: int,
        shuffle: bool,
        check_loaders: bool,
    ) -> None:
        self.dataloader = self.load_data(
            device=device,
            workers=data_workers,
            batch_size=batch_size,
            shuffle=shuffle,
            check_data_loads=check_loaders,
        )

    def asdict(self) -> dict[str, typing.Any]:
        return asdict(self)

    def load_data(
        self,
        device: torch.device,
        workers: int,
        batch_size: int,
        shuffle: bool = False,
        check_data_loads: bool = True,
    ) -> data.DataLoader:
        pin_memory = device.type == "cuda"

        if check_data_loads:
            # Check data loads
            check_loader = data.DataLoader(
                self.data,
                batch_size=2,
                num_workers=2,
                pin_memory=pin_memory,
            )
            first_batch = first(check_loader)
            assert first_batch is not None, "DataLoader check failed"

        return data.DataLoader(
            self.data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=pin_memory,
            persistent_workers=True,  # Avoids issues when also submitting images via MLFlow
        )


@dataclass
class _AbstractTrainObjects(ABC):
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    grad_scaler: GradScaler


@dataclass
class _AbstractLossObjects(ABC):
    loss_function: torch.nn.Module


@dataclass
class _AbstractMetricsObjects(ABC):
    metrics: Metrics
