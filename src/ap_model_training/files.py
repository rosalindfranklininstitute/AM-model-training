from __future__ import annotations
import os
import typing
from pathlib import Path

import pandas as pd
import numpy as np

from ap_model_training.utils import MONAI_KEYS

if typing.TYPE_CHECKING:
    from os import PathLike
    from numpy.typing import NDArray

__all__ = ["paths_dataframe_from_csv", "paths_array_from_csv"]


def paths_dataframe_from_csv(fp: str | PathLike[str]) -> pd.DataFrame:
    fp = Path(fp).absolute()
    # TODO: check whether this trims off the first line when no header is given.
    df = pd.read_csv(
        fp,
        header=0,
        names=[MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL],
        skipinitialspace=True,
        dtype=str,
    )
    base = fp.parent

    df = df.map(lambda _: f"{base / _}" if not os.path.isabs(_) else _)

    return df


def paths_array_from_csv(fp: str | PathLike[str]) -> NDArray[np.str_]:
    return np.asarray(paths_dataframe_from_csv(fp), dtype=np.str_)


def combine_csvs(
    *csvs: str | PathLike[str], output_path: str | PathLike[str]
) -> str | PathLike[str]:
    if len(csvs) == 1:
        return csvs[0]
    dfs: list[pd.DataFrame] = [
        pd.read_csv(
            csv,
            index_col=0,
            header=0,
            names=[MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL],
            skipinitialspace=True,
        )
        for csv in csvs
    ]
    merged_df = pd.concat(dfs, axis=0, ignore_index=True)
    merged_df.to_csv(output_path, index=False)
    return output_path
