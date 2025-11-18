from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np

import lr
from losses import loss_creation_functions

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S'"
    )
)
_logger = logging.getLogger("adaptive_milling_training")
for _ in _logger.handlers:
    _logger.removeHandler(_)
_logger.setLevel(logging.DEBUG)
_logger.addHandler(_handler)


test_csv = "/ceph/groups/structbio/adaptive_milling_project/2024labels_new/test.csv"
all_files_csv = (
    "/ceph/groups/structbio/adaptive_milling_project/2024labels_new/all_files.csv"
)


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

lr_dir = Path(__file__).parent.parent / "learning_rate"
lr_dir.mkdir(exist_ok=True)

for loss_name in loss_creation_functions.keys():
    try:
        lr.plot_learning_rates(
            all_files_csv,
            output_dir=lr_dir,
            cpu_only=False,
            models_to_ignore=["segresnetds2", "segresnetvae"],
            loss_name=loss_name,
            iterations=100,
            image_size=768,
            # model_kwargs={"encoder_weights": "advprop", "pretrained": True},
            loss_kwargs={
                "weights": (1.0, 4.0, 3.0, 6.0, 2.0),
                "dist_matrix": dist_matrix,
            },
            gpu_number=1,
        )
    except Exception:
        _logger.error(
            "Failed to plot learning rates for loss %s", loss_name, exc_info=True
        )
