from __future__ import annotations

import logging
from pathlib import Path

import mlflow

import torch
import numpy as np

from ap_model_training import run
from ap_model_training.utils import MONAI_LOG_DIR

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

# Setup MLFlow
mlflow.pytorch.autolog()
port = 54598
# mlflow_uri = f"file://{MONAI_LOG_DIR}"
mlflow_uri = f"http://localhost:{port}"
_logger.info("Setting up mlflow with URI '%s'", mlflow_uri)
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("ap_model_training")
print(
    f"Run the following command to start:\n$mlflow ui --backend-store-uri {mlflow_uri} --port {port}\nThen navigate to:\nhttp://127.0.0.1:{port}"
)

# Setup logging
_file_handler = logging.FileHandler(MONAI_LOG_DIR / "training.log")
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(_logging_formatter)
_logger.addHandler(_file_handler)


# Input file paths
all_files_csv = (
    "/ceph/groups/structbio/adaptive_milling_project/2024labels_new/all_files4.csv"
)
train_csv = "/ceph/groups/structbio/adaptive_milling_project/2024labels_new/train.csv"
validate_csv = "/ceph/groups/structbio/adaptive_milling_project/2024labels_new/val.csv"


try:
    dist_matrix = np.asarray(
        [
            [0.0, 0.1, 0.1, 0.1, 0.1, 0.1],  # padding
            [0.1, 0.0, 0.5, 0.8, 0.6, 0.2],  # background
            [0.1, 0.5, 0.0, 0.8, 0.9, 1.0],  # lamella
            [0.1, 0.8, 0.8, 0.0, 0.7, 0.7],  # GIS
            [0.1, 0.6, 0.9, 0.7, 0.0, 0.9],  # crack
            [0.1, 0.2, 1.0, 0.7, 0.9, 0.0],  # void
        ],
        dtype=np.float32,
    )
    max_epochs = 100
    frozen_fraction = 0.25
    models_dir = Path.cwd() / "models"
    models_dir.mkdir(exist_ok=True)

    run.run_training(
        models_dir=models_dir,
        model_name="smp_fpn",  # "segresnet",
        # csv_path=all_files_csv,
        training_csv_path=train_csv,
        validation_csv_path=validate_csv,
        validation_split=0.15,
        validation_interval=1,
        image_size=768 * 2,
        pad_images=False,
        rgb_images=True,
        loss_name="diceceloss",
        learning_rate=1e-5,
        epochs=max_epochs,
        include_background=True,
        frozen_epochs=int(max_epochs * frozen_fraction),
        model_kwargs={"encoder_weights": "advprop", "pretrained": True},
        loss_kwargs={
            "weights": (1.0, 4.0, 3.0, 6.0, 2.0),
            "dist_matrix": dist_matrix,
        },
        # cpu_only=True,
        gpu_number=0,
        lr_scheduler_name="onecyclelr",
        lr_scheduler_kwargs={
            "max_lr": 1e-3,  # onecyclelr & cycliclr
            "base_lr": 1e-6,  # cycliclr
            "mode": "triangular2",  # cycliclr
            "step_size_up": 1000,  # cycliclr
            "step_size_down": 2000,  # cycliclr
            "warmup_steps": 5,  # warmupcosineschedule
        },
        training_batch_size=6,
        validation_batch_size=1,
        log_mlflow=False,
        submit_training_images=False,
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
