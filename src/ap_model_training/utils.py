from __future__ import annotations
from pathlib import Path
import typing

import pandas as pd

from monai.utils.enums import CommonKeys

if typing.TYPE_CHECKING:
    from os import PathLike


MONAI_KEYS = CommonKeys

# TENSORBOARD_LOG_DIR = Path.home() / "ap_model_training" / "logs" / "tensorboard"
# TENSORBOARD_LOG_DIR.mkdir(exist_ok=True)

# MONAI_LOG_DIR = Path.home() / "ap_model_training" / "logs" / "mlflow"
# MONAI_LOG_DIR.mkdir(exist_ok=True)


def combine_csvs(
    *csvs: str | PathLike[str], output_path: str | PathLike[str]
) -> str | PathLike[str]:
    if len(csvs) == 1:
        return csvs[0]
    dfs: list[pd.DataFrame] = [pd.read_csv(csv, index_col=0, header=0) for csv in csvs]
    merged_df = pd.concat(dfs, axis=0, ignore_index=True)
    merged_df.to_csv(output_path, index=False)
    return output_path
