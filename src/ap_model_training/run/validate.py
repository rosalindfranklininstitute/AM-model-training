from __future__ import annotations
import logging
import typing
from pathlib import Path
from contextlib import nullcontext
from importlib.metadata import distributions

import pandas as pd
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
    clear_memory,
    get_metrics_to_log,
    get_model_artifact_path,
)
from ap_model_training.run.mlflow import (
    log_training_objects_to_mlflow,
    submit_validation_images_to_mflow,
)

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


def run(
    training_objects: TrainingObjects,
    training_parameters: TrainingParameters,
    log_mlflow: bool = False,
) -> None:
    model_path = Path(training_parameters.model_path)

    pd.DataFrame(training_objects.training_data.data).to_csv(
        model_path.with_name(f"{model_path.stem}_train_data.csv")
    )
    pd.DataFrame(training_objects.validation_data.data).to_csv(
        model_path.with_name(f"{model_path.stem}_val_data.csv")
    )

    with mlflow.start_run() if log_mlflow else nullcontext():
        if log_mlflow:
            # log model to mlflow
            input_array = np.random.uniform(
                size=(
                    1,
                    training_parameters.num_channels,
                    *training_parameters.input_image_shape,
                )
            ).astype(np.float32)
            model_signature = mlflow.models.infer_signature(
                input_array,
                training_objects.model(
                    torch.from_numpy(input_array).to(training_objects.device)
                ).numpy(force=True),
            )
            del input_array

            mlflow.log_params(training_parameters.asdict(include_metrics=False))

            log_training_objects_to_mlflow(training_objects)
        else:
            model_signature = None

        if training_parameters.frozen_epochs > 0:
            # Freeze model if some initial epochs will be frozen
            for param in training_objects.model.encoder.parameters():  # type: ignore
                param.requires_grad = False

        val_epoch_metrics_dict: dict[int, MetricsOutput] = {}
        epoch: int = 1
        for epoch in tqdm(
            range(training_parameters.max_epochs),
            desc="Training progress",
            unit="epoch",
            total=training_parameters.max_epochs,
            initial=1,
        ):
            val_epoch_metrics = None

            val_epoch_metrics, _ = validate(
                training_objects,
                training_parameters,
                epoch=epoch,
                model_path=model_path.with_stem(
                    f"{model_path.stem}_epoch{epoch + 1:03}"
                ),
                model_signature=model_signature,
                log_mlflow=log_mlflow,
            )

            val_epoch_metrics_dict[epoch] = val_epoch_metrics

            epoch_info_str = f"Epoch {epoch + 1}/{training_parameters.max_epochs}, Val Loss: {val_epoch_metrics.loss:.4f}"

            tqdm.write(epoch_info_str)

            pd.DataFrame(
                [
                    {
                        "epoch": epoch,
                        **val_epoch_metrics_dict[epoch].to_dict(
                            split_labels=True,
                            labels=training_parameters.label_names,
                        ),
                    }
                    for epoch in sorted(val_epoch_metrics_dict)
                ]
            ).to_csv(model_path.with_name(f"{model_path.stem}_val_metrics.csv"))

        pd.DataFrame(
            [
                {
                    "epoch": epoch,
                    **val_epoch_metrics_dict[epoch].to_dict(
                        split_labels=True,
                        labels=training_parameters.label_names,
                    ),
                }
                for epoch in sorted(val_epoch_metrics_dict)
            ]
        ).to_csv(model_path.with_name(f"{model_path.stem}_val_metrics.csv"))

        clear_memory()

        if log_mlflow:
            mlflow.log_param(
                "best_train_metrics",
                training_parameters.best_metrics["train"],
            )
            mlflow.log_param(
                "best_val_metrics",
                training_parameters.best_metrics["val"],
            )
            best_val_epoch = int(training_parameters.best_metrics["val"]["epoch"])
            submit_validation_images_to_mflow(
                training_objects=training_objects,
                training_parameters=training_parameters,
                epoch=best_val_epoch,
            )


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
