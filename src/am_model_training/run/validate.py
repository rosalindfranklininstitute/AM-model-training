from __future__ import annotations

import json
import logging
import typing
from pathlib import Path

import mlflow
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

from am_model_training.metrics import Metrics, MetricsOutput
from am_model_training.run.utils import (
    get_mean_of_key_metrics,
    get_metrics_to_log,
)
from am_model_training.utils import MONAI_KEYS

if typing.TYPE_CHECKING:
    from am_model_training.setup import (
        EvaluationObjects,
        EvaluationParameters,
        StageObjects,
        TrainingObjects,
        TrainingParameters,
    )
    from am_model_training.setup.abstract import _AbstractModelObjects

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

    metrics = evaluation_objects.validation.metrics
    metrics.reset()
    evaluation_objects.model.eval()

    loss_list: list[float] = []
    with torch.no_grad():
        for step, batch_data in tqdm(
            enumerate(evaluation_objects.validation.dataloader, 1),
            desc=f"{Path(evaluation_parameters.weights_file).name} validation",
            total=validation_length,
            unit="step",
            leave=False,
        ):
            with torch.device(evaluation_objects.device):
                images = batch_data[MONAI_KEYS.IMAGE].to(evaluation_objects.device)
                labels = batch_data[MONAI_KEYS.LABEL].to(evaluation_objects.device)
                step_loss = _validate_step(
                    step=step,
                    images=images,
                    labels=labels,
                    metrics=metrics,
                    loss_function=evaluation_objects.loss_function,
                    model_objects=evaluation_objects,
                    stage_objects=evaluation_objects.validation,
                    log_mlflow=log_mlflow,
                )
            loss_list.append(step_loss)

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
        json.dump(eval_metrics_dict, f, indent=4)

    with (output_path / "eval_parameters.json").open("w+") as f:
        json.dump(evaluation_parameters.asdict(include_metrics=True), f, indent=4)

    return eval_metrics


def _validate_step(
    step: int,
    images: torch.Tensor,
    labels: torch.Tensor,
    metrics: Metrics,
    loss_function: torch.nn.Module,
    model_objects: _AbstractModelObjects,
    stage_objects: StageObjects,
    log_mlflow: bool = False,
) -> float:
    with autocast(model_objects.device.type):
        outputs = stage_objects.inferer(images, model_objects.model)
        loss = loss_function(outputs, labels)

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
                stage_objects.post_transform(_)
                for _ in decollate_batch(outputs)  # type: ignore
            ],
        ),
    )
    return loss_value


def validate(
    training_objects: TrainingObjects,
    training_parameters: TrainingParameters,
    epoch: int,
    log_mlflow: bool = False,
) -> tuple[MetricsOutput, bool]:
    epoch_len = int(
        np.ceil(
            training_parameters.total_validation_data
            / training_parameters.validation_batch_size
        )
    )

    metrics = training_objects.validation.metrics
    metrics.reset()
    training_objects.model.eval()

    loss_list: list[float] = []
    with torch.no_grad():
        for step, batch_data in tqdm(
            enumerate(training_objects.validation.dataloader, 1),
            desc=f"Epoch {epoch} validation",
            total=epoch_len,
            unit="step",
            leave=False,
        ):
            with torch.device(training_objects.device):
                images = batch_data[MONAI_KEYS.IMAGE].to(training_objects.device)
                labels = batch_data[MONAI_KEYS.LABEL].to(training_objects.device)
                step_loss = _validate_step(
                    step=epoch_len * (epoch - 1) + step,
                    images=images,
                    labels=labels,
                    metrics=metrics,
                    loss_function=training_objects.loss_function,
                    model_objects=training_objects,
                    stage_objects=training_objects.validation,
                    log_mlflow=log_mlflow,
                )
            loss_list.append(step_loss)

        epoch_metrics = metrics.get_epoch_metrics(
            step_losses=loss_list,
            weights=training_parameters.loss_weights,
            device=training_objects.device,
        )

    epoch_metrics_dict = epoch_metrics.to_dict()

    get_mean_of_key_metrics(
        metrics_dict=epoch_metrics_dict,
        key_metrics=training_parameters.key_val_metrics,
    )

    best_epoch = training_parameters.update_metrics(
        epoch=epoch, metrics=epoch_metrics_dict, stage="val"
    )

    metrics.reset()

    if log_mlflow:
        metrics_to_log = get_metrics_to_log(
            "val",
            training_parameters.current_metrics["val"],
            training_parameters.label_names,
        )

        mlflow.log_metrics(
            metrics_to_log,
            step=epoch,
        )

    return epoch_metrics, best_epoch
