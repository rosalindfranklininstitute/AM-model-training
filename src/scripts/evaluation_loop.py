from __future__ import annotations
from pathlib import Path
import json

from ap_model_training.main import evaluate

if __name__ == "__main__":
    min_epoch = 0
    max_epoch = 99
    epoch_step = 1
    weights_directory = Path()  # The directory containing the existing model weights
    output_path = Path()  # The path where outputs will be saved
    csv = Path()  # Path to csv containing data paths

    all_evaluation_metrics = []

    for epoch in range(min_epoch, max_epoch + 1, epoch_step):
        weights_path = weights_directory / f"cryo_sem_epoch_{epoch}.pth"  # Old style
        # weights_path = list(weights_directory.glob(f"*epoch{epoch:03}.pth"))[0]  # New style
        if weights_path.is_file():
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
            metrics_path = output_path / f"smp_fpn_{weights_path.stem}_{csv.stem}_eval_metrics.json"
            if metrics_path.is_file():
                with metrics_path.open() as f:
                    all_evaluation_metrics.append(json.load(f))

    all_metrics_path = output_path / f"smp_fpn_{weights_directory.stem}_{csv.stem}_combined_metrics.json"
    with all_metrics_path.open("w+") as f:
        json.dump(all_evaluation_metrics, f)
