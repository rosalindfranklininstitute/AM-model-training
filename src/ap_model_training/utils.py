from __future__ import annotations
from pathlib import Path

from monai.utils.enums import CommonKeys


MONAI_KEYS = CommonKeys

TENSORBOARD_LOG_DIR = Path.home() / "ap_model_training" / "logs" / "tensorboard"
TENSORBOARD_LOG_DIR.mkdir(exist_ok=True)

MONAI_LOG_DIR = Path.home() / "ap_model_training" / "logs" / "mlflow"
MONAI_LOG_DIR.mkdir(exist_ok=True)
