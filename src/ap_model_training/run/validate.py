from __future__ import annotations
import logging
import typing
from importlib.metadata import distributions

import numpy as np

import torch
from torch.amp import autocast
import mlflow

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

from ap_model_training.utils import MONAI_KEYS
from ap_model_training.metrics import MetricsOutput, Metrics
from ap_model_training.run.utils import get_metrics_to_log, get_model_artifact_path


if typing.TYPE_CHECKING:
    from os import PathLike
    from ap_model_training.setup import TrainingObjects, TrainingParameters

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_logger = logging.getLogger("adaptive_milling_training")

def get_requirements() -> list[str]:
    requirements: list[str] = []
    for dist in distributions():
        name = dist.metadata["Name"]
        version = dist.version
        requirements.append(f"{name}=={version}")
    return requirements


PIP_REQUIREMENTS = get_requirements()

def _validate_step(
    step: int,
    images: torch.Tensor,
    labels: torch.Tensor,
    metrics: Metrics,
    training_objects: TrainingObjects,
    training_parameters: TrainingParameters,
    log_mlflow: bool = False,
) -> float:
    with autocast(training_objects.device.type):
        outputs = training_objects.validation_inferer(images, training_objects.model)
        loss = training_objects.loss_function(outputs, labels)

    loss_value = loss.item()
    if log_mlflow:
        mlflow.log_metric("val_loss", loss_value, step=step)

    del images

    _logger.debug("Updating metrics")
    metrics.update(
        y=labels,
        y_pred=torch.stack(
            [
                training_objects.post_val_transform(_)
                for _ in decollate_batch(outputs)  # type: ignore
            ],
        ),
    )
    return loss_value


def validate(
    training_objects: TrainingObjects,
    training_parameters: TrainingParameters,
    epoch: int,
    model_path: str | PathLike[str],
    model_signature: mlflow.models.ModelSignature | None = None,
    log_mlflow: bool = False,
) -> tuple[MetricsOutput, bool]:
    epoch_len = int(
        np.ceil(
            training_parameters.total_validation_data
            / training_parameters.validation_batch_size
        )
    )

    metrics = training_objects.val_metrics
    metrics.reset()
    training_objects.model.eval()

    loss_list: list[float] = []
    with torch.no_grad():
        for step, batch_data in tqdm(
            enumerate(training_objects.validation_dataloader, 1),
            desc=f"Epoch {epoch + 1} validation",
            total=epoch_len,
            unit="step",
            leave=False,
        ):
            with torch.device(training_objects.device):
                step_loss = _validate_step(
                    step=epoch_len * epoch + step,
                    images=batch_data[MONAI_KEYS.IMAGE].to(training_objects.device),
                    labels=batch_data[MONAI_KEYS.LABEL].to(training_objects.device),
                    metrics=metrics,
                    training_objects=training_objects,
                    training_parameters=training_parameters,
                    log_mlflow=log_mlflow,
                )
            loss_list.append(step_loss)

            del batch_data

        epoch_metrics = metrics.get_epoch_metrics(
            step_losses=loss_list,
            weights=training_parameters.loss_weights,
            device=training_objects.device,
        )
        metrics.reset()
        metrics_to_log, best_epoch = get_metrics_to_log(
            epoch + 1,
            "val",
            training_parameters,
            **epoch_metrics.to_dict(
                split_labels=True,
                labels=training_parameters.label_names,
                prefix="epoch",
            ),
        )

        if best_epoch:
            torch.save(training_objects.model.state_dict(), model_path)
            tqdm.write(f"Model saved: {str(model_path)}")
            if model_signature is not None:
                mlflow.pytorch.log_model(
                    training_objects.model,
                    name=get_model_artifact_path(epoch + 1),
                    signature=model_signature,
                    pip_requirements=PIP_REQUIREMENTS,
                )
            _logger.info(f"Saved new best metric model: {model_path}")

        if log_mlflow:
            mlflow.log_metrics(
                metrics_to_log,
                step=epoch + 1,
            )

    return epoch_metrics, best_epoch
