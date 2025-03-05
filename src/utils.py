from pathlib import Path

from monai.utils.enums import CommonKeys


MONAI_KEYS = CommonKeys

TENSORBOARD_LOG_DIR = (Path(__file__).parent.parent / "logs" / "tensorboard").absolute()
TENSORBOARD_LOG_DIR.mkdir(exist_ok=True)

MONAI_LOG_DIR = (Path.home() / "logs" / "mlflow").absolute()
MONAI_LOG_DIR.mkdir(exist_ok=True)
