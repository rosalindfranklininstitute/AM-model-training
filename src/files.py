from __future__ import annotations
import typing
from pathlib import Path

import pandas as pd

if typing.TYPE_CHECKING:
    from os import PathLike

__all__ = ["paths_from_csv"]


def paths_from_csv(fp: str | PathLike[str]) -> pd.DataFrame[Path]:
    return pd.read_csv(
        fp, header=None, names=["sem", "labels"], skipinitialspace=True, dtype=str
    )
