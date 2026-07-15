from __future__ import annotations
import logging
import json
from pathlib import Path
from importlib.metadata import distributions

import tifffile
import numpy as np

import torch
from torch.amp import autocast

try:
    from IPython import get_ipython

    ip = get_ipython()
    if ip is None:
        from tqdm import tqdm
    else:
        from tqdm.notebook import tqdm

except ImportError:
    from tqdm import tqdm

from monai.data import decollate_batch

from am_model_training.utils import MONAI_KEYS
from am_model_training.setup import InferenceObjects, InferenceParameters

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_logger = logging.getLogger(__name__)


def get_requirements() -> list[str]:
    requirements: list[str] = []
    for dist in distributions():
        name = dist.metadata["Name"]
        version = dist.version
        requirements.append(f"{name}=={version}")
    return requirements


PIP_REQUIREMENTS = get_requirements()


def run(
    inference_objects: InferenceObjects,
    inference_parameters: InferenceParameters,
) -> None:
    # Resolve ensures that the comparison later works as expected
    output_path = Path(inference_parameters.output_path).resolve()
    inference_length = int(
        np.ceil(inference_parameters.total_data / inference_parameters.batch_size)
    )

    # Get image paths in the order that the sampler will index them
    input_paths = [
        Path(inference_objects.data.data[_][MONAI_KEYS.IMAGE]).resolve()
        for _ in inference_objects.dataloader.sampler
    ]

    with (output_path / "inference_parameters.json").open("w+") as f:
        json.dump(inference_parameters.asdict(), f, indent=4)

    with torch.no_grad():
        for step, batch_data in tqdm(
            enumerate(inference_objects.dataloader, 1),
            desc=f"{Path(inference_parameters.weights_file).name} inference",
            total=inference_length,
            unit="step",
            leave=False,
        ):
            inference_objects.dataloader.sampler
            with autocast(inference_objects.device.type):
                batch_data[MONAI_KEYS.IMAGE]
                images = batch_data[MONAI_KEYS.IMAGE].to(inference_objects.device)
                outputs = inference_objects.inferer(images, inference_objects.model)
            images = [
                inference_objects.post_transform(_)
                for _ in decollate_batch(outputs)  # type: ignore
            ]

            index = (step - 1) * inference_parameters.batch_size
            for image, path in zip(images, input_paths[index : index + len(images)]):
                save_path = output_path / f"{path.stem}.tif"
                if save_path == path:
                    _logger.warning(
                        "The output path %s is the same as the input path, so '_label' will be appended to the file name",
                        save_path,
                    )
                    # Avoid overwriting inputs
                    save_path = output_path / f"{path.stem}_label.tif"

                _logger.info("Saving to %s", save_path)
                with tifffile.TiffWriter(save_path) as tiff:
                    tiff.write(
                        image.numpy(force=True)[0, :, :].astype(np.uint8),
                        photometric=tifffile.PHOTOMETRIC.MINISBLACK,
                        dtype="uint8",
                        compression=tifffile.COMPRESSION.LZW,
                    )
