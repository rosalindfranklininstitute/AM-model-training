from __future__ import annotations
import logging
import typing
from collections.abc import Sequence

import numpy as np
import pandas as pd

import torch

from monai import data, transforms
from am_model_training.augmentations import get_transform_list
from am_model_training.utils import MONAI_KEYS

from am_model_training.setup.train import (
    TrainingObjects,
    TrainingParameters,
    setup_training_objects,
)
from am_model_training.setup.eval import (
    EvaluationObjects,
    EvaluationParameters,
    setup_evaluation_objects,
)
from am_model_training.setup.stage import StageObjects
from am_model_training.setup.inference import (
    InferenceObjects,
    InferenceParameters,
    setup_inference_objects,
)

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray

_logger = logging.getLogger(__name__)

__all__ = [
    "create_dataset",
    "create_datasets",
    "get_device",
    "TrainingObjects",
    "TrainingParameters",
    "setup_training_objects",
    "EvaluationObjects",
    "EvaluationParameters",
    "setup_evaluation_objects",
    "StageObjects",
    "InferenceObjects",
    "InferenceParameters",
    "setup_inference_objects",
]


def create_dataset(
    input_data: NDArray[np.str_] | pd.DataFrame | Sequence,
    image_size: int,
    augmentations: bool,
    *,
    pad: bool = True,
    rgb: bool = True,
    dataset_type: type[data.Dataset] = data.Dataset,
    **dataset_kwargs: typing.Any,
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
            )
        ),
        **dataset_kwargs,
    )


def partition_datasets(
    input_data: NDArray[np.str_] | pd.DataFrame,
    validation_split: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if isinstance(input_data, np.ndarray):
        datalist = [
            {MONAI_KEYS.IMAGE: _[0], MONAI_KEYS.LABEL: _[1]} for _ in input_data
        ]
    elif isinstance(input_data, pd.DataFrame):
        datalist = input_data.to_dict(orient="records")
    else:
        raise TypeError(f"Unsupported data type '{type(data)}'")
    return data.partition_dataset(  # type: ignore
        datalist,
        ratios=(1 - validation_split, validation_split),
        num_partitions=2,
        seed=seed,
        shuffle=True,
    )


def create_datasets(
    input_data: NDArray[np.str_] | pd.DataFrame,
    image_size: int,
    validation_split: float = 0.2,
    *,
    pad: bool = True,
    rgb: bool = True,
    dataset_type: type[data.Dataset] = data.Dataset,
    seed: int = 42,
    **dataset_kwargs: typing.Any,
) -> tuple[data.Dataset, data.Dataset]:
    train, validate = partition_datasets(
        input_data=input_data, validation_split=validation_split, seed=seed
    )

    return (
        create_dataset(
            input_data=train,
            image_size=image_size,
            augmentations=True,
            pad=pad,
            rgb=rgb,
            dataset_type=dataset_type,
            **dataset_kwargs,
        ),
        create_dataset(
            input_data=validate,
            image_size=image_size,
            augmentations=False,
            pad=pad,
            rgb=rgb,
            dataset_type=dataset_type,
            **dataset_kwargs,
        ),
    )


def get_device(cpu_only: bool = False, gpu: int | None = None) -> torch.device:
    gpu_str = "cuda"
    if gpu is not None:
        gpu_str += f":{gpu}"
    return torch.device(
        gpu_str if not cpu_only and torch.cuda.is_available() else "cpu"
    )
