from __future__ import annotations
import logging
import typing
from pathlib import Path

import numpy as np

import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid, draw_segmentation_masks
import mlflow

from monai.inferers import sliding_window_inference

from utils import MONAI_KEYS

if typing.TYPE_CHECKING:
    from os import PathLike
    from collections.abc import Callable
    from setup import TrainingObjects, TrainingParameters


_logger = logging.getLogger("adaptive_milling_training")


class EarlyStopper:
    """Adapted from ignite.handlers.early_stopping.EarlyStopping"""

    def __init__(
        self,
        patience: int,
        min_delta: float = 0,
        cumulative_delta: bool = False,
    ):
        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        if min_delta < 0.0:
            raise ValueError("Argument min_delta should not be a negative number.")

        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta
        self.counter = 0
        self.best_score: float | None = None

    def stop_early(self, score: float) -> None:
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


def calculate_batch_metrics(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    metrics_dict: dict[str, Callable],
) -> None:
    for metric_fn in metrics_dict.values():
        metric_fn(y_pred=outputs, y=labels)


def run(
    training_objects: TrainingObjects,
    training_parameters: TrainingParameters,
) -> None:
    model_path = Path(training_parameters.model_path)

    train_stopper = EarlyStopper(patience=training_parameters.train_patience)
    val_stopper = EarlyStopper(patience=training_parameters.val_patience)

    # start a typical PyTorch training
    val_interval: int = 2

    with mlflow.start_run():
        # log model to mlflow
        input_array = np.random.uniform(
            size=(1, 1, *training_parameters.input_image_shape)
        ).astype(np.float32)
        model_signature = mlflow.models.infer_signature(
            input_array,
            training_objects.model(
                torch.from_numpy(input_array).to(training_objects.device)
            )
            .detach()
            .to("cpu")
            .numpy(),
        )

        init_params_dict = training_parameters.asdict(include_metrics=False)
        mlflow.log_params(init_params_dict)

        if training_parameters.frozen_epochs > 0:
            # Freeze model if some initial epochs will be frozen
            for param in training_objects.model.parameters():
                param.requires_grad = False

        for epoch in range(training_parameters.max_epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{training_parameters.max_epochs}")
            train(training_objects, training_parameters, epoch=epoch)

            if epoch > 0 and epoch == training_parameters.frozen_epochs:
                # Unfreeze (no need if it wasn't frozen)
                for param in training_objects.model.parameters():
                    param.requires_grad = True

            if (epoch + 1) % val_interval == 0:
                validate(
                    training_objects,
                    training_parameters,
                    epoch=epoch,
                    model_path=model_path,
                    model_signature=model_signature,
                )
                if train_stopper.stop_early(
                    training_parameters.current_metrics["train"][
                        training_parameters.best_metric
                    ]
                ):
                    _logger.info(
                        "Stopped early after epoch %i due to train stopper",
                        epoch + 1,
                    )
                    break
                if val_stopper.stop_early(
                    training_parameters.current_metrics["val"][
                        training_parameters.best_metric
                    ]
                ):
                    _logger.info(
                        "Stopped early after epoch %i due to val stopper", epoch + 1
                    )
                    break

        print(
            f"train completed, best metric '{training_parameters.best_metric}': {training_parameters.best_metrics['val'][training_parameters.best_metric]:.4f} at epoch {training_parameters.best_metrics['val']['epoch']}"
        )
        mlflow.log_param(
            "best_train_metrics",
            training_parameters.best_metrics["train"][training_parameters.best_metric],
        )
        mlflow.log_param(
            "best_val_metrics",
            training_parameters.best_metrics["val"][training_parameters.best_metric],
        )


def train(
    training_objects: TrainingObjects,
    training_parameters: TrainingParameters,
    epoch: int,
) -> None:
    training_objects.model.train()
    epoch_loss = 0
    for step, batch_data in enumerate(training_objects.training_dataloader, 1):
        batch_data
        inputs, labels = (
            batch_data[MONAI_KEYS.IMAGE].to(training_objects.device),
            batch_data[MONAI_KEYS.LABEL].to(training_objects.device),
        )
        training_objects.optimizer.zero_grad()

        with torch.autocast(training_objects.device.type):
            outputs = training_objects.model(inputs)

            # Apply post_train_transform to each image (decollate causes issues for some metrics)
            for i in range(outputs.size(0)):
                outputs[i] = training_objects.post_train_transform(outputs[i])

            loss = training_objects.loss_function(outputs, labels)

        training_objects.grad_scaler.scale(loss).backward()
        training_objects.grad_scaler.step(training_objects.optimizer)
        training_objects.grad_scaler.update()
        training_objects.lr_scheduler.step()

        # Calculate metrics and log progress for this step
        epoch_loss += loss.item()
        epoch_len = (
            training_parameters.total_training_data
            // training_parameters.training_batch_size
        )
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        mlflow.log_metric("train_loss", loss.item(), step=epoch_len * epoch + step)
        calculate_batch_metrics(
            outputs,
            labels,
            metrics_dict=training_objects.train_metrics,
        )

    # Calculate metrics and log progress for this epoch
    epoch_loss /= step

    epoch_metrics: dict[str, float] = {}
    for metric_name, metric_fn in training_objects.train_metrics.items():
        epoch_metrics[f"{metric_name}"] = metric_fn.aggregate().item()
        metric_fn.reset()

    epoch_metrics["epoch_loss"] = epoch_loss

    # Calculate mean metric
    _ = tuple(epoch_metrics[k] for k in training_parameters.key_train_metrics)
    epoch_metrics["mean_of_key_metrics"] = sum(_) / len(_)

    training_parameters.update_metrics(
        epoch=epoch + 1, metrics_dict=epoch_metrics, stage="train"
    )

    _logger.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    metrics_to_log = {
        f"train_{k}": v for k, v in training_parameters.current_metrics["train"].items()
    }
    del metrics_to_log["train_epoch"]
    mlflow.log_metrics(
        metrics_to_log,
        step=epoch + 1,
    )


def validate(
    training_objects: TrainingObjects,
    training_parameters: TrainingParameters,
    epoch: int,
    model_path: str | PathLike[str],
    model_signature: mlflow.models.ModelSignature | None = None,
) -> None:
    training_objects.model.eval()
    epoch_loss = 0
    with torch.no_grad():
        val_outputs: tuple[typing.Any] | None = None
        for step, val_data in enumerate(training_objects.validation_dataloader, 1):
            val_images, val_labels = (
                val_data[MONAI_KEYS.IMAGE].to(training_objects.device),
                val_data[MONAI_KEYS.LABEL].to(training_objects.device),
            )
            roi_size = (96, 96)
            sw_batch_size = 4
            with torch.autocast(training_objects.device.type):
                val_outputs = sliding_window_inference(  # type: ignore[assignment]
                    val_images, roi_size, sw_batch_size, training_objects.model
                )

                if val_outputs is None:
                    raise TypeError("val_outputs is None")

                # Apply post_train_transform to each image (decollate causes issues for some metrics)
                for i in range(val_outputs.size(0)):
                    val_outputs[i] = training_objects.post_val_transform(val_outputs[i])

                epoch_loss += training_objects.loss_function(
                    val_outputs, val_labels
                ).item()

            # Calculate metrics for this step
            calculate_batch_metrics(
                val_outputs,
                val_labels,
                metrics_dict=training_objects.val_metrics,
            )

        # Calculate metrics and log progress for this epoch
        epoch_loss /= step

        epoch_metrics: dict[str, float] = {}
        for (
            metric_name,
            metric_fn,
        ) in training_objects.val_metrics.items():
            epoch_metrics[f"{metric_name}"] = metric_fn.aggregate().item()
            metric_fn.reset()

        epoch_metrics["epoch_loss"] = epoch_loss

        # Calculate mean metric
        _ = tuple(epoch_metrics[k] for k in training_parameters.key_val_metrics)
        epoch_metrics["mean_of_key_metrics"] = sum(_) / len(_)

        if training_parameters.update_metrics(
            epoch=epoch + 1, metrics_dict=epoch_metrics, stage="val"
        ):
            torch.save(training_objects.model.state_dict(), model_path)
            mlflow.pytorch.log_model(
                training_objects.model,
                "model",
                signature=model_signature,
                pip_requirements=[
                    f"-r {Path(__file__).absolute().parent.parent / 'requirements.txt'}"
                ],
            )
            _logger.info(f"Saved new best metric model: {model_path}")

    metrics_to_log = {
        f"val_{k}": v for k, v in training_parameters.current_metrics["val"].items()
    }
    del metrics_to_log["val_epoch"]
    mlflow.log_metrics(
        metrics_to_log,
        step=epoch + 1,
    )

    submit_images_to_mlflow(val_images, val_labels, val_outputs, step=epoch + 1)


def submit_images_to_mlflow(
    images: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor, step: int
) -> None:
    images = make_grid(images)
    labels = make_grid(labels.to(torch.bool))
    predictions = make_grid(predictions.to(torch.bool))

    labels = to_pil_image(draw_segmentation_masks(images, labels))
    predictions = to_pil_image(draw_segmentation_masks(images, predictions))
    images = to_pil_image(images)

    mlflow.log_image(
        images,
        step=step,
        key=MONAI_KEYS.IMAGE,
    )
    mlflow.log_image(
        labels,
        step=step,
        key=MONAI_KEYS.LABEL,
    )
    mlflow.log_image(
        predictions,
        step=step,
        key=MONAI_KEYS.PRED,
    )
