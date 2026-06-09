from __future__ import annotations
import argparse
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--csv",
        type=Path,
        action="append",
        dest="csv_path",
        required=True,
        help="Path to a csv file listing paths to image-segmentation pairs. If this argument is given multiple times the files will be combined.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        dest="output_directory",
        required=True,
        help="Path to directory where models will be saved. Directory will be created if it doesn't already exist.",
    )
    parser.add_argument(
        "-v",
        "--validation-csv",
        type=Path,
        action="append",
        dest="validation_path",
        required=False,
        help="Path to a csv file listing paths to image-segmentation pairs to be used for validation. This will override the --validation-split argument if specified. If this argument is given multiple times the files will be combined.",
    )
    parser.add_argument(
        "-s",
        "--validation-split",
        default=0.2,
        type=float,
        dest="validation_split",
        required=False,
        help="The validation split to use. This will be ignored if --validation-csv is given. (default: %(default)s)",
    )
    parser.add_argument(
        "-e",
        "--max-epochs",
        default=100,
        type=int,
        dest="max_epochs",
        required=False,
        help="The maximum number of epochs to refine for. (default: %(default)s)",
    )
    parser.add_argument(
        "--frozen",
        default=25,
        type=int,
        dest="frozen_epochs",
        required=False,
        help="The number of epochs before the encoder is unfrozen. (default: %(default)s)",
    )
    parser.add_argument(
        "-tb",
        "--training-batch",
        default=6,
        type=int,
        dest="training_batch_size",
        required=False,
        help="The training batch size. Larger numbers will run faster, smaller will require less memory. (default: %(default)s)",
    )
    parser.add_argument(
        "-vb",
        "--validation-batch",
        default=1,
        type=int,
        dest="validation_batch_size",
        required=False,
        help="The validation batch size. Larger numbers will run faster, smaller will require less memory. (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        dest="gpu_number",
        required=False,
        help="Number of the GPU to use. (default: %(default)s)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        dest="cpu_only",
        required=False,
        help="Use the CPU only, ignoring the --gpu setting. This will be extremely slow, so is not recommended.",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        dest="save_all",
        required=False,
        help="Save model weights from all epochs. Otherwise, just the models that improve on previous epochs.",
    )

    return parser


def parse_arguments(parser: argparse.ArgumentParser) -> None:
    namespace = parser.parse_args()

    from ap_model_training.main import train
    from ap_model_training.files import combine_csvs

    namespace.output_path.mkdir(exist_ok=True)

    csv = combine_csvs(
        *namespace.csv_path, output_path=namespace.output_path / "combined.csv"
    )
    if namespace.validation_paths:
        val_csv = combine_csvs(
            *namespace.validation_path,
            output_path=namespace.output_path / "combined_val.csv",
        )
        csv = (csv, val_csv)

    train(
        output_path=namespace.output_path,
        csv=csv,
        max_epochs=namespace.max_epochs,
        frozen_epochs=namespace.frozen_epochs,
        validation_split=namespace.validation_split,
        cpu_only=namespace.cpu_only,
        gpu_number=namespace.gpu_number,
        training_batch_size=namespace.training_batch_size,
        validation_batch_size=namespace.validation_batch_size,
        initial_learning_rate=2e-6,
        max_learning_rate=3e-4,
        save_all_models=namespace.save_all_models,
        mlflow_experiment_name="ap_model_training",
    )


def main():
    parse_arguments(create_parser())


if __name__ == "__main__":
    main()
