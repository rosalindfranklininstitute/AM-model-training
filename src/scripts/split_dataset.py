from __future__ import annotations
import typing
from pathlib import Path

import pandas as pd

from ap_model_training.files import paths_dataframe_from_csv
from ap_model_training.setup import partition_datasets

if typing.TYPE_CHECKING:
    from os import PathLike


def run(csv: str | PathLike[str], validation_split: float = 0.2, seed: int = 42) -> None:
    csv = Path(csv).resolve()
    if not csv.is_file():
        raise FileNotFoundError(csv)
    train, validation = partition_datasets(paths_dataframe_from_csv(csv), validation_split=validation_split, seed=seed)

    pd.DataFrame(train).to_csv(csv.with_stem(f"{csv.stem}_train_split"), index=False)
    pd.DataFrame(validation).to_csv(csv.with_stem(f"{csv.stem}_val_split"), index=False)

if __name__ == "__main__":
    csv = Path()

    run(csv=csv, validation_split=0.15, seed=42)
