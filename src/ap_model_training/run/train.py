from __future__ import annotations
import logging
import typing
import time
import json
from pathlib import Path
from contextlib import nullcontext

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
from ap_model_training.run.utils import clear_memory, get_metrics_to_log
from ap_model_training.run.mlflow import (
    log_training_objects_to_mlflow,
    submit_images_to_mlflow,
    submit_validation_images_to_mflow,
)
from ap_model_training.run.validate import validate

if typing.TYPE_CHECKING:
    from ap_model_training.setup import TrainingObjects, TrainingParameters

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_logger = logging.getLogger("adaptive_milling_training")


class EarlyStopper:
    """Adapted from ignite.handlers.early_stopping.EarlyStopping"""

    def __init__(
        self,
        patience: int,
        min_delta: float = 0,
        cumulative_delta: bool = False,
    ) -> None:
        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        if min_delta < 0.0:
            raise ValueError("Argument min_delta should not be a negative number.")

        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta
        self.counter = 0
        self.best_score: float | None = None

    def stop_early(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.min_delta:
            if not self.cumulative_delta and score > self.best_score:
                self.best_score = score
            self.counter += 1
            _logger.debug("EarlyStopper: %i / %i", self.counter, self.patience)
            if self.counter >= self.patience:
                _logger.info("EarlyStopper: stopping training")
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False


def run(
    training_objects: TrainingObjects,
    training_parameters: TrainingParameters,
    submit_training_images: bool = False,
    log_mlflow: bool = False,
) -> None:
    model_path = Path(training_parameters.model_path)

    train_stopper = EarlyStopper(patience=training_parameters.train_patience)
    val_stopper = EarlyStopper(patience=training_parameters.val_patience)

    pd.DataFrame(training_objects.training_data.data).to_csv(
        model_path.with_name(f"{model_path.stem}_train_data.csv"), index=False
    )
    pd.DataFrame(training_objects.validation_data.data).to_csv(
        model_path.with_name(f"{model_path.stem}_val_data.csv"), index=False
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

        train_epoch_metrics_dict: dict[int, MetricsOutput] = {}
        val_epoch_metrics_dict: dict[int, MetricsOutput] = {}
        epoch: int = 1
        for epoch in tqdm(
            range(training_parameters.max_epochs),
            desc="Training progress",
            unit="epoch",
            total=training_parameters.max_epochs,
            initial=1,
        ):
            train_epoch_metrics = None
            val_epoch_metrics = None
            # print("-" * 10)
            # print(f"epoch {epoch + 1}/{training_parameters.max_epochs}")
            train_epoch_metrics, best_train_epoch = train(
                training_objects,
                training_parameters,
                epoch=epoch,
                submit_images=submit_training_images,
                log_mlflow=log_mlflow,
            )
            train_epoch_metrics_dict[epoch] = train_epoch_metrics

            pd.DataFrame(
                [
                    {
                        "epoch": epoch,
                        **train_epoch_metrics_dict[epoch].to_dict(
                            split_labels=True,
                            labels=training_parameters.label_names,
                        ),
                    }
                    for epoch in sorted(train_epoch_metrics_dict)
                ]
            ).to_csv(model_path.with_name(f"{model_path.stem}_train_metrics.csv"))

            if epoch > 0 and epoch == training_parameters.frozen_epochs:
                # Unfreeze (no need if it wasn't frozen)
                for param in training_objects.model.encoder.parameters():  # type: ignore
                    param.requires_grad = True

            epoch_info_str = f"Epoch {epoch + 1}/{training_parameters.max_epochs}, Train Loss: {train_epoch_metrics.loss:.4f}"
            if (epoch + 1) % training_parameters.val_interval == 0 or best_train_epoch:
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

                epoch_info_str += f", Val Loss: {val_epoch_metrics.loss:.4f}"

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

            if val_epoch_metrics is not None:
                if training_parameters.frozen_epochs < epoch and val_stopper.stop_early(
                    training_parameters.current_metrics["val"][
                        training_parameters.best_metric
                    ]
                ):
                    tqdm.write(
                        f"Stopped early after epoch {epoch + 1} due to val stopper"
                    )
                    _logger.info(
                        "Stopped early after epoch %i due to val stopper", epoch + 1
                    )
                    break

            if training_parameters.frozen_epochs < epoch and train_stopper.stop_early(
                training_parameters.current_metrics["train"][
                    training_parameters.best_metric
                ]
            ):
                if val_epoch_metrics is None:
                    _logger.info(
                        "Starting final validation loop, as train stopper has been triggered but no validation has been run this epoch"
                    )
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
                tqdm.write(
                    f"Stopped early after epoch {epoch + 1} due to train stopper"
                )
                _logger.info(
                    "Stopped early after epoch %i due to train stopper",
                    epoch + 1,
                )
                break

        print(
            f"train completed, best metric '{training_parameters.best_metric}': {training_parameters.best_metrics['val'][training_parameters.best_metric]:.4f} at epoch {training_parameters.best_metrics['val']['epoch']}"
        )
        pd.DataFrame(
            [
                {
                    "epoch": epoch,
                    **train_epoch_metrics_dict[epoch].to_dict(
                        split_labels=True,
                        labels=training_parameters.label_names,
                    ),
                }
                for epoch in sorted(train_epoch_metrics_dict)
            ]
        ).to_csv(model_path.with_name(f"{model_path.stem}_train_metrics.csv"))

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

        with model_path.with_name("training_parameters.json").open("w+") as f:
            json.dump(training_parameters.asdict(include_metrics=True), f)

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


def _train_step(
    step: int,
    images: torch.Tensor,
    labels: torch.Tensor,
    metrics: Metrics,
    training_objects: TrainingObjects,
    training_parameters: TrainingParameters,
    log_mlflow: bool = False,
    submit_images: bool = False,
) -> float:
    images.requires_grad_()
    training_objects.optimizer.zero_grad()
    with autocast(training_objects.device.type):
        outputs = training_objects.training_inferer(images, training_objects.model)
        loss = training_objects.loss_function(outputs, labels)

    skip_lr_scheduler = False
    if training_objects.grad_scaler is not None:
        training_objects.grad_scaler.scale(loss).backward()
        training_objects.grad_scaler.step(training_objects.optimizer)
        scale = training_objects.grad_scaler.get_scale()
        training_objects.grad_scaler.update()
        skip_lr_scheduler = scale > training_objects.grad_scaler.get_scale()
    else:
        loss.backward()
        training_objects.optimizer.step()

    if training_objects.lr_scheduler is not None:
        if log_mlflow:
            mlflow.log_metrics(
                {
                    f"learning_rate_{i}": _
                    for i, _ in enumerate(training_objects.lr_scheduler.get_last_lr())
                },
                step=step,
            )

        if not skip_lr_scheduler:
            training_objects.lr_scheduler.step()

    # Calculate metrics and log progress for this step
    # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
    loss_value = loss.item()
    if np.isnan(loss_value):
        _logger.warning("Loss for training step %i is NaN", step)
        tqdm.write(f"Loss for training step {step} is NaN")

    outputs = torch.stack(
        [
            training_objects.post_train_transform(_)
            for _ in decollate_batch(outputs)  # type: ignore
        ]
    )
    if log_mlflow:
        if not np.isnan(loss_value):
            mlflow.log_metric("train_loss", loss_value, step=step)
        if submit_images:
            submit_images_to_mlflow(
                images,
                labels,
                outputs,
                step=step,
                num_classes=training_parameters.num_classes,
                timestamp=int(time.time()),
                separate_background=False,
            )
    del images
    _logger.debug("Updating metrics")
    metrics.update(
        y=labels,
        y_pred=outputs,
    )
    return loss_value


def train(
    training_objects: TrainingObjects,
    training_parameters: TrainingParameters,
    epoch: int,
    log_mlflow: bool = False,
    submit_images: bool = False,
) -> tuple[MetricsOutput, bool]:
    metrics = training_objects.train_metrics
    metrics.reset()
    training_objects.model.train()
    epoch_len = int(
        np.ceil(
            training_parameters.total_training_data
            / training_parameters.training_batch_size
        )
    )
    loss_list: list[float] = []
    for step, batch_data in tqdm(
        enumerate(training_objects.training_dataloader, 1),
        desc=f"Epoch {epoch + 1} training",
        total=epoch_len,
        unit="step",
        leave=False,
    ):
        with torch.device(training_objects.device):
            step_loss = _train_step(
                step=epoch_len * epoch + step,
                images=batch_data[MONAI_KEYS.IMAGE].to(training_objects.device),
                labels=batch_data[MONAI_KEYS.LABEL].to(training_objects.device),
                metrics=metrics,
                training_objects=training_objects,
                training_parameters=training_parameters,
                log_mlflow=log_mlflow,
                submit_images=submit_images,
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
        "train",
        training_parameters,
        **epoch_metrics.to_dict(
            split_labels=True,
            labels=training_parameters.label_names,
            prefix="epoch",
        ),
    )

    if log_mlflow:
        mlflow.log_metrics(
            metrics_to_log,
            step=epoch + 1,
        )
    return epoch_metrics, best_epoch
