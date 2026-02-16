from __future__ import annotations
from pathlib import Path

from ap_model_training.main import main

if __name__ == "__main__":
    # Input file paths
    all_files_csv = (
        "/ceph/groups/structbio/adaptive_milling_project/2024labels_new/all_files4.csv"
    )
    train_csv = "/ceph/users/tpr78264/code/adaptive_milling_monai/training_data.csv"
    validate_csv = (
        "/ceph/users/tpr78264/code/adaptive_milling_monai/validation_data.csv"
    )

    save_directory = Path.cwd() / "models"
    save_directory.mkdir(exist_ok=True)

    main(
        save_directory=save_directory,
        csv=(train_csv, validate_csv),
        max_epochs=100,
        frozen_epochs=25,
        validation_split=0.15,  # Ignored if csv is passed as a tuple of training and validation csvs
        cpu_only=True,
        gpu_number=0,
        training_batch_size=6,
        validation_batch_size=1,
        log_mlflow=False,
        mlflow_experiment_name="ap_model_training",
    )
