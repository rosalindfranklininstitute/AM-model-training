from __future__ import annotations

import gc
import sys
import logging
from pathlib import Path

import mlflow

import torch
import numpy as np

from ap_model_training import run
from ap_model_training.utils import MONAI_LOG_DIR

_logger = logging.getLogger("adaptive_milling_training")
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
mlflow_uri = f"file://{MONAI_LOG_DIR}"
_logger.info("Setting up mlflow with URI '%s'", mlflow_uri)
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("MONAI adaptive milling")
print(
    f"Run the following command to start:\n$mlflow ui --backend-store-uri {mlflow_uri} --port {port}\nThen navigate to:\nhttp://127.0.0.1:{port}"
)

# Setup logging
_file_handler = logging.FileHandler(MONAI_LOG_DIR / "training.log")
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(_logging_formatter)
_logger.addHandler(_file_handler)


# Input file paths
test_csv = "/ceph/groups/structbio/adaptive_milling_project/2024labels_new/test.csv"
all_files_csv = (
    "/ceph/groups/structbio/adaptive_milling_project/2024labels_new/all_files.csv"
)

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
    max_epochs = 120
    frozen_fraction = 0.2
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    run.run_training(
        # model_save_path = (
        #     Path.cwd().parent
        #     / "models"
        #     / "250317_234235_all_files_smp_efficientnet_b4_unetplusplus.pth"
        # )
        # mlflow_run_id = "f6453abe173949bf8564c868fa1a0528"
        # run.submit_validation_for_mlflow_run(
        #     mlflow_run_id,
        #     model_save_path,
        #     116,
        all_files_csv,
        models_dir=models_dir,
        model_name="smp_efficientnet_b4_unet",  # "segresnet",
        loss_name="diceloss",
        learning_rate=1e-2,
        epochs=max_epochs,
        image_size=768 * 2,
        frozen_epochs=int(max_epochs * frozen_fraction),
        model_kwargs={"encoder_weights": "advprop", "pretrained": True},
        loss_kwargs={
            "weights": (0.0, 1.0, 4.0, 3.5, 5.0, 2.0),
            "dist_matrix": dist_matrix,
        },
        gpu_number=2,
        lr_scheduler_name="cycliclr",
        lr_scheduler_kwargs={
            "max_lr": 1e-2,  # onecyclelr & cycliclr
            "base_lr": 1e-6,  # cycliclr
            "mode": "triangular2",  # cycliclr
            "step_size_up": 1000,  # cycliclr
            "step_size_down": 2000,  # cycliclr
            "warmup_steps": 5,  # warmupcosineschedule
        },
    )
finally:  # noqa: E722
    with torch.no_grad():
        torch.cuda.empty_cache()
    try:
        mlflow.end_run()
    except:  # noqa: E722
        _logger.error("Failed to end MLFlow run", exc_info=True)
