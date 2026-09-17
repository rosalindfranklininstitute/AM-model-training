from __future__ import annotations

import logging
import time
import typing
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from torch.amp import autocast
from torch.nn.functional import one_hot
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks

try:
    from IPython import get_ipython

    ip = get_ipython()
    if ip is None:
        from tqdm import tqdm
    else:
        from tqdm.notebook import tqdm

except ImportError:
    from tqdm import tqdm

from monai.data import Dataset, decollate_batch
from monai.transforms import Compose

from am_model_training.utils import MONAI_KEYS

if typing.TYPE_CHECKING:
    from am_model_training.setup import TrainingObjects, TrainingParameters

_logger = logging.getLogger(__name__)

_tab10 = plt.get_cmap("tab10")


def submit_validation_images_to_mflow(
    training_objects: TrainingObjects,
    training_parameters: TrainingParameters,
    epoch: int,
) -> None:
    # Log images using the best val epoch model
    model_path = Path(training_parameters.output_path) / training_parameters.model_name
    training_objects.model.load_state_dict(
        state_dict=torch.load(
            model_path.with_stem(f"{model_path.stem}_epoch{epoch:03}"),
            map_location=training_objects.device,
        )
    )
    training_objects.model.eval()
    epoch_len = int(
        np.ceil(
            training_parameters.total_validation_data
            / training_parameters.validation_batch_size
        )
    )
    with torch.no_grad():
        for data in tqdm(
            training_objects.validation.dataloader,
            desc="Submitting validation images from best validation epoch to MLFlow",
            total=epoch_len,
            unit="step",
            leave=False,
        ):
            images, labels = (
                data[MONAI_KEYS.IMAGE].to(training_objects.device),
                data[MONAI_KEYS.LABEL].to(training_objects.device),
            )
            with autocast(training_objects.device.type):
                outputs = training_objects.validation.inferer(
                    images, training_objects.model
                )

            outputs = torch.stack(
                [
                    training_objects.validation.post_transform(_)
                    for _ in decollate_batch(outputs)  # type: ignore
                ]
            )

            submit_images_to_mlflow(
                images,
                labels,
                outputs,
                step=epoch,
                num_classes=training_parameters.num_classes,
                timestamp=int(time.time()),
                separate_background=training_parameters.include_background,
            )
            del images
            del labels
            del outputs

        mlflow.flush_artifact_async_logging()
        mlflow.flush_async_logging()


def submit_images_to_mlflow(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    step: int,
    max_dims: tuple[int, int] = (512, 512),
    num_classes: int = 5,
    timestamp: int | None = None,
    log_unlabelled: bool = False,
    separate_background: bool = False,
) -> None:
    _logger.debug("Sumbitting images to MLFlow")
    colours: list[tuple[int, int, int] | str] = [
        tuple((np.asarray(_tab10(_)[:3]) * 255).astype(int).tolist())
        for _ in range(num_classes)
    ]
    for img, label, pred in zip(images, labels, predictions):
        img = img.to("cpu", copy=True)
        label = (
            one_hot(torch.squeeze(label), num_classes=num_classes)
            .to("cpu", torch.bool, copy=True)
            .permute(2, 0, 1)
        )
        pred = (
            one_hot(torch.squeeze(pred), num_classes=num_classes)
            .to("cpu", torch.bool, copy=True)
            .permute(2, 0, 1)
        )
        # Images must be handled last as full size RGB required for draw_segmentation_masks
        if len(img.shape) == 2 or img.shape[0] == 1:
            rgb_img = img.repeat((3, 1, 1))
        else:
            rgb_img = img
        pred = to_pil_image(
            draw_segmentation_masks(
                rgb_img, pred[1 if separate_background else 0 :, ...], colors=colours
            )
        )
        pred.thumbnail(max_dims)

        label = to_pil_image(
            draw_segmentation_masks(
                rgb_img, label[1 if separate_background else 0 :, ...], colors=colours
            )
        )
        label.thumbnail(max_dims)
        del rgb_img

        if log_unlabelled:
            img = to_pil_image(img)
            img.thumbnail(max_dims)

            mlflow.log_image(
                img,
                step=step,
                key=MONAI_KEYS.IMAGE,
                timestamp=timestamp,
                synchronous=False,
            )
        else:
            del img

        # Log thumbnailed PIL images as this is quicker to display and more space efficient
        mlflow.log_image(
            label,
            step=step,
            key=MONAI_KEYS.LABEL,
            timestamp=timestamp,
            synchronous=False,
        )
        mlflow.log_image(
            pred,
            step=step,
            key=MONAI_KEYS.PRED,
            timestamp=timestamp,
            synchronous=False,
        )
    _logger.debug("Submitted images to MLFlow")


def log_training_objects_to_mlflow(training_objects: TrainingObjects) -> None:
    def log_types(name, obj) -> None:
        if isinstance(obj, Dataset):
            mlflow.log_param(
                f"{name}_transforms",
                tuple(
                    f"{_.__class__.__name__}({_.__dict__})"
                    for _ in obj.transform.transforms
                ),
            )
        elif isinstance(obj, Compose):
            mlflow.log_param(
                name,
                tuple(
                    f"{_.__class__.__name__}({{k: v for k, v in _.__dict__.items() if not k.startswith('_')}})"
                    for _ in obj.transforms
                ),
            )
        elif hasattr(obj, "__dict__"):
            mlflow.log_param(
                name,
                f"{obj.__class__.__name__}({obj.__dict__})",
            )
        else:
            mlflow.log_param(
                name,
                str(obj),
            )

    for name, obj in training_objects.asdict().items():
        if name == "model":
            continue
        elif name == "optimizer":
            mlflow.log_param(
                name,
                str(obj),
            )
            continue
        try:
            if name == "training":
                for sub_name, sub_obj in obj.items():
                    log_types(f"training_{sub_name}", sub_obj)
            elif name == "validation":
                for sub_name, sub_obj in obj.items():
                    log_types(f"validation_{sub_name}", sub_obj)
            else:
                log_types(name, obj)
        except Exception:
            _logger.warning("Failed to log '%s'", name, exc_info=True)
