from __future__ import annotations

import json
import typing
from pathlib import Path

from monai.utils.enums import CommonKeys

if typing.TYPE_CHECKING:
    from os import PathLike

MONAI_KEYS = CommonKeys

# TENSORBOARD_LOG_DIR = Path.home() / "ap_model_training" / "logs" / "tensorboard"
# TENSORBOARD_LOG_DIR.mkdir(exist_ok=True)

# MONAI_LOG_DIR = Path.home() / "ap_model_training" / "logs" / "mlflow"
# MONAI_LOG_DIR.mkdir(exist_ok=True)


def get_best_epoch(
    training_directory: str | PathLike[str],
    stage: typing.Literal["train", "val"] = "val",
) -> int:
    training_parameters_path = Path(training_directory) / "training_parameters.json"

    with training_parameters_path.open() as f:
        training_parameters = dict(json.load(f))
    return int(training_parameters["best_metrics"][stage]["epoch"])


def get_weights_path_of_best_epoch(
    training_directory: str | PathLike[str],
    stage: typing.Literal["train", "val"] = "val",
) -> Path:
    training_directory = Path(training_directory)

    best_epoch = get_best_epoch(training_directory=training_directory, stage=stage)
    weights_paths = list(training_directory.glob(f"*epoch{best_epoch:03}.pth"))
    if not weights_paths:
        raise FileNotFoundError(
            f"No weights found for epoch {best_epoch} in {training_directory}"
        )
    elif len(weights_paths) > 1:
        raise ValueError(
            f"Multiple matching weights files found for epoch  {best_epoch} in {training_directory}"
        )
    return weights_paths[0]
