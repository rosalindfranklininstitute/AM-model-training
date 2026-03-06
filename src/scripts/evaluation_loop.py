from __future__ import annotations
import re
import json
import typing
from pathlib import Path

from tqdm import tqdm

from ap_model_training.main import evaluate

if typing.TYPE_CHECKING:
    from os import PathLike

weights_epoch_pattern = re.compile(r".*epoch_?(\d{0,3}).pth", re.I)


def get_epoch_from_path(path: Path) -> int:
    m = weights_epoch_pattern.match(path.name)
    if m is None:
        raise ValueError("Regex failed")
    return int(m.group(1))


def eval_for_multiple_epochs(
    output_path: str | PathLike[str],
    csv: str | PathLike[str],
    weights_directory: str | PathLike[str],
    cpu_only: bool = False,
    gpu_number: int = 0,
    batch_size: int = 1,
) -> Path:
    output_path = Path(output_path)
    csv = Path(csv)
    weights_directory = Path(weights_directory)

    weights_paths = sorted(
        ((get_epoch_from_path(path), path) for path in weights_directory.glob("*.pth"))
    )
    all_evaluation_metrics = []
    for epoch, weights_path in tqdm(
        weights_paths,
        desc=f"Evaluation with {csv.name}",
        total=len(weights_paths),
        unit="epoch",
        leave=True,
    ):
        if weights_path.is_file():
            evaluate(
                output_path=output_path,
                csv=csv,
                weights_path=weights_path,
                cpu_only=cpu_only,
                gpu_number=gpu_number,
                batch_size=batch_size,
                log_mlflow=False,
                mlflow_experiment_name="ap_model_evaluating",
            )
            metrics_path = tuple(
                (output_path / f"evaluation_{weights_path.stem}_{csv.stem}").glob(
                    "*_metrics.json"
                )
            )[0]
            if metrics_path.is_file():
                with metrics_path.open() as f:
                    all_evaluation_metrics.append(json.load(f))

    all_metrics_path = (
        output_path
        / f"smp_fpn_{weights_directory.stem}_{csv.stem}_combined_metrics.json"
    )
    with all_metrics_path.open("w+") as f:
        json.dump(all_evaluation_metrics, f, indent=4)
    return all_metrics_path


if __name__ == "__main__":
    weights_directory = Path()  # The directory containing the existing model weights
    output_path = (
        Path.cwd() / f"eval_{weights_directory.name}"
    )  # The path where outputs will be saved
    output_path.mkdir()

    train_csv = Path()
    validate_csv = Path()

    csvs = [
        train_csv,
        validate_csv,
    ]  # Path to csv containing data paths, each will get a separate output

    for csv in csvs:
        eval_for_multiple_epochs(
            output_path=output_path,
            csv=csv,
            weights_directory=weights_directory,
            cpu_only=False,
            gpu_number=0,  # Sets which GPU will be used (if cpu_only=False)
            batch_size=6,
        )
