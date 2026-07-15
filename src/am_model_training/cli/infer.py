from __future__ import annotations
import argparse
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI for inferring images using an Adaptive Milling model"
    )

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
        "-w",
        "--weights",
        type=Path,
        dest="weights_path",
        required=True,
        help="Path to the weights file that will be refined.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=1,
        type=int,
        dest="batch_size",
        required=False,
        help="The batch size. Larger numbers will run faster, smaller will require less memory. (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        dest="gpu_number",
        required=False,
        help="Number of the GPU that will be used. (default: %(default)s)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        dest="cpu_only",
        required=False,
        help="Use the CPU only, ignoring the --gpu setting. This will be extremely slow, so is not recommended.",
    )

    return parser


def parse_arguments(parser: argparse.ArgumentParser) -> None:
    namespace = parser.parse_args()

    from am_model_training.main import infer
    from am_model_training.files import combine_csvs

    namespace.output_path.mkdir(exist_ok=True)

    csv = combine_csvs(
        *namespace.csv_path, output_path=namespace.output_path / "combined.csv"
    )

    infer(
        output_path=namespace.output_path,
        csv=csv,
        weights_path=namespace.weights_path,
        cpu_only=namespace.cpu_only,
        gpu_number=namespace.gpu_number,  # Sets which GPU will be used (if cpu_only=False)
        batch_size=namespace.batch_size,
    )


def main():
    parse_arguments(create_parser())


if __name__ == "__main__":
    main()
