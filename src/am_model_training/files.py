from __future__ import annotations

import os
import typing
from pathlib import Path

import numpy as np
import pandas as pd

from am_model_training.utils import MONAI_KEYS

if typing.TYPE_CHECKING:
    from os import PathLike

    from numpy.typing import NDArray

__all__ = ["paths_dataframe_from_csv", "paths_array_from_csv"]


def paths_dataframe_from_csv(
    fp: str | PathLike[str], include_labels: bool = True
) -> pd.DataFrame:
    fp = Path(fp).absolute()
    df = pd.read_csv(
        fp,
        names=[MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL],
        skipinitialspace=True,
        dtype=str,
    )
    base = fp.parent

    first_row = df.iloc[0, :]
    if first_row[MONAI_KEYS.IMAGE] == MONAI_KEYS.IMAGE and (
        first_row[MONAI_KEYS.LABEL] == MONAI_KEYS.LABEL or not include_labels
    ):
        df = df.iloc[1:].reset_index(drop=True)

    if not include_labels:
        df.drop(columns=[MONAI_KEYS.LABEL], inplace=True)

    # Make paths absolute relative to the folder containing the csv file
    df = df.map(lambda _: f"{base / _}" if not os.path.isabs(_) else _)

    return df


def paths_array_from_csv(fp: str | PathLike[str]) -> NDArray[np.str_]:
    return np.asarray(paths_dataframe_from_csv(fp), dtype=np.str_)


def combine_csvs(
    *csvs: str | PathLike[str], output_path: str | PathLike[str]
) -> str | PathLike[str]:
    if len(csvs) == 1:
        return csvs[0]
    dfs: list[pd.DataFrame] = [paths_dataframe_from_csv(csv) for csv in csvs]
    merged_df = pd.concat(dfs, axis=0, ignore_index=True)
    merged_df.to_csv(output_path, index=False)
    return output_path
