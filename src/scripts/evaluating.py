from __future__ import annotations
from pathlib import Path
import typing

from ap_model_training.main import evaluate
from ap_model_training.utils import combine_csvs

if typing.TYPE_CHECKING:
    from os import PathLike

if __name__ == "__main__":
    weights_path = Path()  # The path to the existing model weights
    output_path = Path()  # The path where outputs will be saved
    csv_path = output_path / "combined.csv"  # Make sure you don't overwrite this

    # Put list of paths here
    retrain_csvs: list[str | PathLike[str]] = []

    csv = combine_csvs(*retrain_csvs[:], output_path=csv_path)

    evaluate(
        output_path=output_path,
        csv=csv,
        weights_path=weights_path,
        cpu_only=False,
        gpu_number=0,  # Sets which GPU will be used (if cpu_only=False)
        batch_size=1,
        log_mlflow=False,
        mlflow_experiment_name="ap_model_evaluating",
    )
