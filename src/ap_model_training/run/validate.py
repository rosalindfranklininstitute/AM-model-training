from __future__ import annotations
import logging
import json
import typing
from pathlib import Path

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
from ap_model_training.run.utils import (
    get_mean_of_key_metrics,
    get_metrics_to_log,
)

if typing.TYPE_CHECKING:
    from os import PathLike
    from ap_model_training.setup import (
        TrainingObjects,
        TrainingParameters,
        EvaluationObjects,
        EvaluationParameters,
    )

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_logger = logging.getLogger(__name__)


def evaluate(
    evaluation_objects: EvaluationObjects,
    evaluation_parameters: EvaluationParameters,
    log_mlflow: bool = False,
) -> MetricsOutput:
    output_path = Path(evaluation_parameters.output_path)

    validation_length = int(
        np.ceil(evaluation_parameters.total_data / evaluation_parameters.batch_size)
    )

    metrics = evaluation_objects.metrics
    metrics.reset()
    evaluation_objects.model.eval()

    loss_list: list[float] = []
    with torch.no_grad():
        for step, batch_data in tqdm(
            enumerate(evaluation_objects.dataloader, 1),
            desc=f"{Path(evaluation_parameters.weights_file).name} validation",
            total=validation_length,
            unit="step",
            leave=False,
        ):
            with torch.device(evaluation_objects.device):
                step_loss = _validate_step(
                    step=step,
                    images=batch_data[MONAI_KEYS.IMAGE].to(evaluation_objects.device),
                    labels=batch_data[MONAI_KEYS.LABEL].to(evaluation_objects.device),
                    metrics=metrics,
                    training_objects=evaluation_objects,
                    log_mlflow=log_mlflow,
                )
            loss_list.append(step_loss)

            del batch_data

        eval_metrics = metrics.get_epoch_metrics(
            step_losses=loss_list,
            weights=evaluation_parameters.loss_weights,
            device=evaluation_objects.device,
        )

    eval_metrics_dict = eval_metrics.to_dict()

    get_mean_of_key_metrics(
        metrics_dict=eval_metrics_dict,
        key_metrics=evaluation_parameters.key_metrics,
    )

    evaluation_parameters.update_metrics(metrics=eval_metrics_dict)

    metrics.reset()

    if log_mlflow:
        metrics_to_log = get_metrics_to_log(
            "eval",
            evaluation_parameters.metrics,
            evaluation_parameters.label_names,
        )

        mlflow.log_metrics(
            metrics_to_log,
            step=0,
        )

    with (output_path / f"{evaluation_parameters.run_id}_eval_metrics.json").open(
        "w+"
    ) as f:
        json.dump(eval_metrics_dict, f)

    with (output_path / "eval_parameters.json").open("w+") as f:
        json.dump(evaluation_parameters.asdict(include_metrics=True), f, indent=4)

    return eval_metrics


def _validate_step(
    step: int,
    images: torch.Tensor,
    labels: torch.Tensor,
    metrics: Metrics,
    training_objects: TrainingObjects | EvaluationObjects,
    log_mlflow: bool = False,
) -> float:
    with autocast(training_objects.device.type):
        outputs = training_objects.validation_inferer(images, training_objects.model)
        loss = training_objects.loss_function(outputs, labels)

    loss_value = loss.item()

    if np.isnan(loss_value):
        _logger.warning("Loss for training step %i is NaN", step)
        tqdm.write(f"Loss for training step {step} is NaN")
    elif log_mlflow:
        mlflow.log_metric("val_loss", loss_value, step=step)

    del images

    _logger.debug("Updating metrics")
    metrics.update(
        y=labels,
        y_pred=torch.stack(
            [
                training_objects.post_transform(_)
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
                    log_mlflow=log_mlflow,
                )
            loss_list.append(step_loss)

            del batch_data

        epoch_metrics = metrics.get_epoch_metrics(
            step_losses=loss_list,
            weights=training_parameters.loss_weights,
            device=training_objects.device,
        )

    epoch_metrics_dict = epoch_metrics.to_dict(prefix="epoch")

    get_mean_of_key_metrics(
        metrics_dict=epoch_metrics_dict,
        key_metrics=training_parameters.key_val_metrics,
    )

    stage = "val"
    best_epoch = training_parameters.update_metrics(
        epoch=epoch + 1, metrics=epoch_metrics_dict, stage=stage
    )

    metrics.reset()

    if log_mlflow:
        metrics_to_log = get_metrics_to_log(
            stage,
            training_parameters.current_metrics[stage],
            training_parameters.label_names,
        )

        mlflow.log_metrics(
            metrics_to_log,
            step=epoch + 1,
        )

    return epoch_metrics, best_epoch
