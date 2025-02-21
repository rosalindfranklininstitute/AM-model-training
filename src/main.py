from __future__ import annotations
import sys
import typing
from datetime import datetime
from pathlib import Path

import mlflow

from files import paths_from_csv
from setup import dataframe_to_datasets
from train import train

if typing.TYPE_CHECKING:
    from os import PathLike

mlflow.pytorch.autolog()


def main(csv_path: str | PathLike[str]) -> None:
    csv_path = Path(csv_path)
    assert csv_path.is_file()

    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = (
        models_dir / f"{datetime.now().strftime('%y%m%d_%H%M%S')}_{csv_path.stem}.pth"
    )

    df = paths_from_csv(csv_path)

    training_data, validation_data = dataframe_to_datasets(df, validation_split=0.2)
    with mlflow.start_run(description=model_path.stem):
        train(
            training_data,
            validation_data,
            label_count=5,
            model_path=model_path,
            epochs=30,
        )


if __name__ == "__main__":
    main(sys.argv[1])
