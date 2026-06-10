from __future__ import annotations
from pathlib import Path
import typing

from am_model_training.main import train
from am_model_training.files import combine_csvs

if typing.TYPE_CHECKING:
    from os import PathLike

if __name__ == "__main__":
    weights_path = Path()  # The path to the existing model weights
    output_path = Path()  # The path where models will be saved
    csv_path = output_path / "combined.csv"  # Make sure you don't overwrite this

    # Put list of paths here, one per lamella
    retrain_csvs: list[str | PathLike[str]] = []

    # Use this to determine which lamelae will be used for training (useful to figure out how many lamellae we need to get a good model)
    csv = combine_csvs(*retrain_csvs[:], output_path=csv_path)

    train(
        output_path=output_path,
        csv=csv,
        weights_path=weights_path,
        max_epochs=20,
        frozen_epochs=0,
        validation_split=0.15,  # Ignored if csv is passed as a tuple of training and validation csvs
        cpu_only=False,
        gpu_number=0,  # Sets which GPU will be used (if cpu_only=False)
        training_batch_size=6,
        validation_batch_size=1,
        save_all_models=True,
        initial_learning_rate=2e-6,
        max_learning_rate=3e-4,
        log_mlflow=False,
        mlflow_experiment_name="ap_model_retraining",
    )
