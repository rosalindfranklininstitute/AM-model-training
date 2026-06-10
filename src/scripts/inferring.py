from __future__ import annotations
import typing
from pathlib import Path

from am_model_training.main import infer
from am_model_training.files import combine_csvs

if typing.TYPE_CHECKING:
    from os import PathLike


if __name__ == "__main__":
    weights_path = Path()  # The path to the existing model weights
    output_path = Path()  # The path where outputs will be saved

    # Put list of paths here
    image_csvs: list[str | PathLike[str]] = []

    csv = combine_csvs(*image_csvs[:], output_path=output_path / "combined.csv")

    infer(
        output_path=output_path,
        csv=csv,
        weights_path=weights_path,
        cpu_only=False,
        gpu_number=0,  # Sets which GPU will be used (if cpu_only=False)
        batch_size=6,
    )
