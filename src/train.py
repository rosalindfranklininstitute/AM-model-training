from __future__ import annotations
import logging
import typing
import time
from pathlib import Path

import numpy as np

import torch
from torch.nn.functional import one_hot
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks
import mlflow

try:
    from IPython import get_ipython

    ip = get_ipython()
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

from monai.data import decollate_batch, Dataset
from monai.transforms import Compose
from monai.utils import set_determinism

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

    set_determinism(seed=0)

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

        mlflow.log_params(training_parameters.asdict(include_metrics=False))

        log_training_objects_to_mlflow(training_objects)

        if training_parameters.frozen_epochs > 0:
            # Freeze model if some initial epochs will be frozen
            for param in training_objects.model.encoder.parameters():
                param.requires_grad = False

        for epoch in tqdm(
            range(training_parameters.max_epochs),
            desc="Training progress",
            unit="epoch",
            total=training_parameters.max_epochs,
            initial=1,
        ):
            # print("-" * 10)
            # print(f"epoch {epoch + 1}/{training_parameters.max_epochs}")
            train(training_objects, training_parameters, epoch=epoch)

            if epoch > 0 and epoch == training_parameters.frozen_epochs:
                # Unfreeze (no need if it wasn't frozen)
                for param in training_objects.model.encoder.parameters():
                    param.requires_grad = True

            if (epoch + 1) % val_interval == 0:
                validate(
                    training_objects,
                    training_parameters,
                    epoch=epoch,
                    model_path=model_path,
                    model_signature=model_signature,
                )

                if training_parameters.frozen_epochs < epoch and val_stopper.stop_early(
                    training_parameters.current_metrics["val"][
                        training_parameters.best_metric
                    ]
                ):
                    _logger.info(
                        "Stopped early after epoch %i due to val stopper", epoch + 1
                    )
                    break

            if training_parameters.frozen_epochs < epoch and train_stopper.stop_early(
                training_parameters.current_metrics["train"][
                    training_parameters.best_metric
                ]
            ):
                _logger.info(
                    "Stopped early after epoch %i due to train stopper",
                    epoch + 1,
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
    epoch_len = int(
        np.ceil(
            training_parameters.total_training_data
            / training_parameters.training_batch_size
        )
    )

    for step, batch_data in tqdm(
        enumerate(training_objects.training_dataloader, 1),
        desc=f"Epoch {epoch + 1} training",
        total=epoch_len,
        unit="step",
        leave=False,
    ):
        images, labels = (
            batch_data[MONAI_KEYS.IMAGE].to(training_objects.device),
            batch_data[MONAI_KEYS.LABEL].to(training_objects.device),
        )
        training_objects.optimizer.zero_grad()
        with torch.autocast(training_objects.device.type):
            outputs = training_objects.training_inferer(images, training_objects.model)
            loss = training_objects.loss_function(outputs, labels)

        outputs = [
            training_objects.post_train_transform(_) for _ in decollate_batch(outputs)
        ]
        labels = [
            training_objects.post_train_label_transform(_)
            for _ in decollate_batch(labels)
        ]

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
            mlflow.log_metrics(
                {
                    f"learning_rate_{i}": _
                    for i, _ in enumerate(training_objects.lr_scheduler.get_last_lr())
                },
                step=epoch_len * epoch + step,
            )

            if not skip_lr_scheduler:
                training_objects.lr_scheduler.step()

        # Calculate metrics and log progress for this step
        epoch_loss += loss.item()
        # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
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
        epoch_metrics[f"{metric_name}"] = metric_fn.aggregate().numpy(force=True)
        metric_fn.reset()

    epoch_metrics["epoch_loss"] = epoch_loss

    # Calculate mean metric
    _ = np.asarray(
        tuple(epoch_metrics[k] for k in training_parameters.key_train_metrics)
    )
    epoch_metrics["mean_of_key_metrics"] = _.mean()

    training_parameters.update_metrics(
        epoch=epoch + 1, metrics_dict=epoch_metrics, stage="train"
    )

    _logger.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    metrics_to_log: dict[str, float] = {}
    for metric_name, values in training_parameters.current_metrics["train"].items():
        if metric_name == "epoch":
            continue
        metric_key = f"train_{metric_name}"
        if isinstance(values, np.ndarray):
            for label_name, v in zip(training_parameters.label_names, values):
                metrics_to_log[f"{metric_key}_{label_name}"] = v
            metrics_to_log[f"{metric_key}_mean"] = values.mean()
        else:
            metrics_to_log[metric_key] = values
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
    epoch_len = int(
        np.ceil(
            training_parameters.total_validation_data
            / training_parameters.validation_batch_size
        )
    )
    with torch.no_grad():
        outputs: tuple[typing.Any] | None = None
        for step, data in tqdm(
            enumerate(training_objects.validation_dataloader, 1),
            desc=f"Epoch {epoch + 1} validation",
            total=epoch_len,
            unit="step",
            leave=False,
        ):
            images, labels = (
                data[MONAI_KEYS.IMAGE].to(training_objects.device),
                data[MONAI_KEYS.LABEL].to(training_objects.device),
            )
            with torch.autocast(training_objects.device.type):
                outputs = training_objects.validation_inferer(
                    images, training_objects.model
                )
                loss = training_objects.loss_function(outputs, labels)

            mlflow.log_metric("val_loss", loss.item(), step=epoch_len * epoch + step)
            epoch_loss += loss

            outputs = [
                training_objects.post_val_transform(_) for _ in decollate_batch(outputs)
            ]

            labels = [
                training_objects.post_val_label_transform(_)
                for _ in decollate_batch(labels)
            ]

            # Calculate metrics for this step
            calculate_batch_metrics(
                outputs,
                labels,
                metrics_dict=training_objects.val_metrics,
            )

        # Calculate metrics and log progress for this epoch
        epoch_loss /= step

        epoch_metrics: dict[str, float] = {}
        for (
            metric_name,
            metric_fn,
        ) in training_objects.val_metrics.items():
            epoch_metrics[f"{metric_name}"] = metric_fn.aggregate().numpy(force=True)
            metric_fn.reset()

        epoch_metrics["epoch_loss"] = epoch_loss

        # Calculate mean metric
        _ = np.asarray(
            tuple(epoch_metrics[k] for k in training_parameters.key_val_metrics)
        )
        epoch_metrics["mean_of_key_metrics"] = _.mean()

        if training_parameters.update_metrics(
            epoch=epoch + 1, metrics_dict=epoch_metrics, stage="val"
        ):
            torch.save(training_objects.model.state_dict(), model_path)
            mlflow.pytorch.log_model(
                training_objects.model,
                f"epoch_{epoch + 1}_model",
                signature=model_signature,
                pip_requirements=[
                    f"-r {Path(__file__).absolute().parent.parent / 'requirements.txt'}"
                ],
            )
            _logger.info(f"Saved new best metric model: {model_path}")

            for step, data in tqdm(
                enumerate(training_objects.validation_dataloader, 1),
                desc=f"Submitting epoch {epoch + 1} validation images to MLFlow",
                total=epoch_len,
                unit="step",
                leave=False,
            ):
                images, labels = (
                    data[MONAI_KEYS.IMAGE].to(training_objects.device),
                    data[MONAI_KEYS.LABEL].to(training_objects.device),
                )
                with torch.autocast(training_objects.device.type):
                    outputs = training_objects.validation_inferer(
                        images, training_objects.model
                    )

                outputs = [
                    training_objects.post_val_transform(_)
                    for _ in decollate_batch(outputs)
                ]

                labels = [
                    training_objects.post_val_label_transform(_)
                    for _ in decollate_batch(labels)
                ]

                submit_images_to_mlflow(
                    images,
                    labels,
                    outputs,
                    step=epoch + 1,
                    onehot=None,
                    include_background=True,
                    swap_xy=False,
                )

    metrics_to_log: dict[str, float] = {}
    for metric_name, values in training_parameters.current_metrics["val"].items():
        if metric_name == "epoch":
            continue
        metric_key = f"val_{metric_name}"
        if isinstance(values, np.ndarray):
            for label_name, v in zip(training_parameters.label_names, values):
                metrics_to_log[f"{metric_key}_{label_name}"] = v
            metrics_to_log[f"{metric_key}_mean"] = values.mean()
        else:
            metrics_to_log[metric_key] = values
    mlflow.log_metrics(
        metrics_to_log,
        step=epoch + 1,
    )


def submit_images_to_mlflow(
    images: torch.Tensor | list[torch.Tensor],
    labels: torch.Tensor | list[torch.Tensor],
    predictions: torch.Tensor | list[torch.Tensor],
    step: int,
    max_dims: tuple[int, int] = (512, 512),
    swap_xy: bool = True,
    onehot: int | None = None,
    include_background: bool = False,
) -> None:
    _logger.debug("Sumbitting images to MLFlow")
    colours = ["gray", "orange", "green", "red", "yellow"]

    # Use the same timestamp for each set of submissions
    timestamp = time.time()

    if onehot is not None:
        labels = one_hot(labels.to(torch.long), onehot).squeeze(1).permute((0, 3, 1, 2))
        predictions = one_hot(predictions.to(torch.long), onehot).permute((0, 3, 1, 2))

    if swap_xy:
        images = images.permute((0, 1, 3, 2))
        labels = labels.permute((0, 1, 3, 2))
        predictions = predictions.permute((0, 1, 3, 2))

    if not include_background:
        labels = labels[:, 1:, :, :]
        predictions = predictions[:, 1:, :, :]

    for img, label, pred in zip(images, labels, predictions):
        img = img.to("cpu", copy=True)
        label = label.to("cpu", torch.bool, copy=True)
        pred = pred.to("cpu", torch.bool, copy=True)
        rgb_img = img.repeat((3, 1, 1))
        pred = to_pil_image(draw_segmentation_masks(rgb_img, pred, colors=colours))
        pred.thumbnail(max_dims)

        label = to_pil_image(draw_segmentation_masks(rgb_img, label, colors=colours))
        label.thumbnail(max_dims)
        del rgb_img

        # Images must be handled last as full size RGB required for draw_segmentation_masks
        img = to_pil_image(img)
        img.thumbnail(max_dims)

        # Log thumbnailed PIL images as this is quicker to display and more space efficient
        mlflow.log_image(
            img,
            step=step,
            key=MONAI_KEYS.IMAGE,
            timestamp=timestamp,
            synchronous=False,
        )
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
    mlflow.flush_artifact_async_logging()
    _logger.debug("Submitted images to MLFlow")


def log_training_objects_to_mlflow(training_objects: TrainingObjects) -> None:
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
                        f"{_.__class__.__name__}({dict(((k, v) for k, v in _.__dict__.items() if not k.startswith('_')))})"
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

        except Exception:
            _logger.warning("Failed to log '%s'", name, exc_info=True)
