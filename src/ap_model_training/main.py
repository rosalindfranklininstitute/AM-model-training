from __future__ import annotations

import logging
from pathlib import Path
import typing

import mlflow
import torch

from ap_model_training import run
from ap_model_training.utils import MONAI_LOG_DIR

if typing.TYPE_CHECKING:
    from os import PathLike

_logger = logging.getLogger(__package__)
_logger.propagate = False
_logger.setLevel(logging.DEBUG)
for _handler in _logger.handlers:
    _logger.removeHandler(_handler)

_logging_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S'"
)
_stream_handler = logging.StreamHandler()
_stream_handler.setLevel(logging.WARNING)
_stream_handler.setFormatter(_logging_formatter)

_logger.addHandler(_stream_handler)


def main(
    save_directory: str | PathLike[str],
    csv: str | PathLike[str] | tuple[str | PathLike[str], str | PathLike[str]],
    max_epochs: int,
    frozen_epochs: int,
    validation_split: float = 0.15,  # Ignored if two csv files given
    weights_path: str | PathLike[str] | None = None,
    cpu_only: bool = False,
    gpu_number: int = 0,
    training_batch_size: int = 6,
    validation_batch_size: int = 1,
    seed: int = 42,
    log_mlflow: bool = False,
    mlflow_experiment_name: str = "ap_model_training",
) -> None:
    if log_mlflow:
        # Setup MLFlow
        mlflow.pytorch.autolog()
        port = 54598
        mlflow_uri = f"http://localhost:{port}"
        _logger.info("Setting up mlflow with URI '%s'", mlflow_uri)
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(mlflow_experiment_name)
        print(
            f"Run the following command to start:\n$mlflow ui --port {port}\nThen navigate to:\nhttp://127.0.0.1:{port}"
        )

        # Setup logging
        _file_handler = logging.FileHandler(MONAI_LOG_DIR / "training.log")
        _file_handler.setLevel(logging.INFO)
        _file_handler.setFormatter(_logging_formatter)
        _logger.addHandler(_file_handler)

    try:
        models_dir = Path(save_directory)
        models_dir.mkdir(exist_ok=True)

        run.run_training(
            models_dir=models_dir,
            model_name="smp_fpn",
            csv=csv,
            validation_split=validation_split,
            validation_interval=1,
            image_size=1536,
            pad_images=False,
            rgb_images=True,
            loss_name="compound_loss",
            learning_rate=1e-5,
            epochs=max_epochs,
            include_background=True,
            frozen_epochs=frozen_epochs,
            model_kwargs={
                "pretrained": False,
                "weights_file": weights_path,
            },
            loss_kwargs={
                "weights": (1.0, 4.0, 3.0, 6.0, 2.0),
                "alpha": 0.5,
            },
            cpu_only=cpu_only,
            gpu_number=gpu_number,
            lr_scheduler_name="onecyclelr",
            lr_scheduler_kwargs={
                "max_lr": 1e-3,  # onecyclelr
            },
            training_batch_size=training_batch_size,
            validation_batch_size=validation_batch_size,
            log_mlflow=log_mlflow,
            submit_training_images=False,
            seed=seed,
        )
    finally:  # noqa: E722
        try:
            with torch.no_grad():
                torch.cuda.empty_cache()
        except Exception:
            _logger.error("Failed to clear CUDA cache", exc_info=True)
        try:
            mlflow.end_run()
        except:  # noqa: E722
            _logger.error("Failed to end MLFlow run", exc_info=True)
