from __future__ import annotations

from pathlib import Path

from am_model_training.main import train

if __name__ == "__main__":
    # Input file paths
    train_csv = Path()
    validate_csv = Path()

    output_path = Path.cwd() / "models"
    output_path.mkdir(exist_ok=True)

    # Should be a single csv path or a tuple of training and validation csv paths
    # If a single csv path is given, the validation split will be applied.
    csv = (train_csv, validate_csv)

    train(
        output_path=output_path,
        csv=csv,
        max_epochs=100,
        frozen_epochs=25,
        validation_split=0.15,  # Ignored if csv is passed as a tuple of training and validation csvs
        cpu_only=False,
        gpu_number=0,  # Sets which GPU will be used (if cpu_only=False)
        training_batch_size=6,
        validation_batch_size=1,
        save_all_models=False,
        log_mlflow=False,
        mlflow_experiment_name="am_model_training",
    )
