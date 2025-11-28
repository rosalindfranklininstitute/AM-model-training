from __future__ import annotations
import logging
import typing
import time
import gc
import pandas as pd
from importlib.metadata import distributions
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
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

from ap_model_training.utils import MONAI_KEYS
from ap_model_training.metrics import StepMetrics, EpochMetrics

if typing.TYPE_CHECKING:
    from os import PathLike
    from ap_model_training.setup import TrainingObjects, TrainingParameters


_logger = logging.getLogger("adaptive_milling_training")


def get_requirements() -> list[str]:
    requirements: list[str] = []
    for dist in distributions():
        name = dist.metadata["Name"]
        version = dist.version
        requirements.append(f"{name}=={version}")
    return requirements


pip_requirements = get_requirements()
_tab10 = plt.get_cmap("tab10")


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
) -> None:
    model_path = Path(training_parameters.model_path)

    train_stopper = EarlyStopper(patience=training_parameters.train_patience)
    val_stopper = EarlyStopper(patience=training_parameters.val_patience)

    # Fix determinism for consistent results
    set_determinism(seed=42)

    with mlflow.start_run():
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

        if training_parameters.frozen_epochs > 0:
            # Freeze model if some initial epochs will be frozen
            for param in training_objects.model.encoder.parameters():
                param.requires_grad = False

        train_epoch_metrics_list: list[EpochMetrics] = []
        val_epoch_metrics_list: list[EpochMetrics] = []

        for epoch in tqdm(
            range(training_parameters.max_epochs),
            desc="Training progress",
            unit="epoch",
            total=training_parameters.max_epochs,
            initial=1,
        ):
            # print("-" * 10)
            # print(f"epoch {epoch + 1}/{training_parameters.max_epochs}")
            train_epoch_metrics, _, _ = train(
                training_objects, training_parameters, epoch=epoch
            )
            train_epoch_metrics_list.append(train_epoch_metrics)

            clear_memory()

            if epoch > 0 and epoch == training_parameters.frozen_epochs:
                # Unfreeze (no need if it wasn't frozen)
                for param in training_objects.model.encoder.parameters():
                    param.requires_grad = True

            if (epoch + 1) % training_parameters.val_interval == 0:
                val_epoch_metrics, _, best_val_epoch = validate(
                    training_objects,
                    training_parameters,
                    epoch=epoch,
                    model_path=model_path.with_suffix(
                        f"_epoch{epoch:03}{model_path.suffix}"
                    ),
                    model_signature=model_signature,
                )

                val_epoch_metrics_list.append(val_epoch_metrics)
                clear_memory()

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

        clear_memory()

        print(
            f"train completed, best metric '{training_parameters.best_metric}': {training_parameters.best_metrics['val'][training_parameters.best_metric]:.4f} at epoch {training_parameters.best_metrics['val']['epoch']}"
        )
        mlflow.log_param(
            "best_train_metrics",
            training_parameters.best_metrics["train"],
        )
        mlflow.log_param(
            "best_val_metrics",
            training_parameters.best_metrics["val"],
        )
        pd.DataFrame(
            [
                _.to_dict(
                    split_labels=True,
                    labels=training_parameters.label_names,
                )
                for _ in train_epoch_metrics_list
            ]
        ).to_csv(model_path.with_suffix("_train_metrics.csv"))

        pd.DataFrame(
            [
                _.to_dict(
                    split_labels=True,
                    labels=training_parameters.label_names,
                )
                for _ in val_epoch_metrics_list
            ]
        ).to_csv(model_path.with_suffix("_val_metrics.csv"))

        best_val_epoch = int(training_parameters.best_metrics["val"]["epoch"])
        submit_validation_images_to_mflow(
            training_objects=training_objects,
            training_parameters=training_parameters,
            epoch=best_val_epoch,
        )


def submit_validation_images_to_mflow(
    training_objects: TrainingObjects,
    training_parameters: TrainingParameters,
    epoch: int,
) -> None:
    # Log images using the best val epoch model
    training_objects.model.load_state_dict(
        state_dict=torch.load(
            training_parameters.model_path, map_location=training_objects.device
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
            training_objects.validation_dataloader,
            desc="Submitting validation images from best validation epoch to MLFlow",
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
                training_objects.post_val_transform(_) for _ in decollate_batch(outputs)
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
                timestamp=int(time.time()),
                separate_background=training_parameters.include_background,
            )
            del images
            del labels
            del outputs
            clear_memory()

        mlflow.flush_artifact_async_logging()
        mlflow.flush_async_logging()


def train(
    training_objects: TrainingObjects,
    training_parameters: TrainingParameters,
    epoch: int,
) -> tuple[EpochMetrics, list[StepMetrics], bool]:
    training_objects.model.train()
    epoch_len = int(
        np.ceil(
            training_parameters.total_training_data
            / training_parameters.training_batch_size
        )
    )
    step_metrics_list: list[StepMetrics] = []
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
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

        del images

        step_metrics = StepMetrics.calculate_metrics(
            loss=loss.item(),  # type: ignore
            y=np.asarray(
                [
                    training_objects.post_train_label_transform(_).detach().cpu()
                    for _ in decollate_batch(labels)
                ]
            ),
            y_pred=np.asarray(
                [
                    training_objects.post_train_transform(_).detach().cpu()
                    for _ in decollate_batch(outputs)
                ]
            ),
            weights=training_parameters.loss_weights,
        )
        step_metrics_list.append(step_metrics)

        mlflow.log_metrics(
            step_metrics.to_dict(
                split_labels=True,
                labels=training_parameters.label_names,
                prefix="train",
            ),
            step=epoch_len * epoch + step,
        )
        del loss

        del outputs
        del labels

        clear_memory()

    epoch_metrics = EpochMetrics.from_step_metrics(step_metrics_list)

    metrics_to_log, best_epoch = get_metrics_to_log(
        epoch + 1,
        "train",
        training_parameters,
        **epoch_metrics.to_dict(),
    )

    mlflow.log_metrics(
        metrics_to_log,
        step=epoch + 1,
    )
    return epoch_metrics, step_metrics_list, best_epoch


def validate(
    training_objects: TrainingObjects,
    training_parameters: TrainingParameters,
    epoch: int,
    model_path: str | PathLike[str],
    model_signature: mlflow.models.ModelSignature | None = None,
) -> tuple[EpochMetrics, list[StepMetrics], bool]:
    training_objects.model.eval()
    epoch_len = int(
        np.ceil(
            training_parameters.total_validation_data
            / training_parameters.validation_batch_size
        )
    )
    step_metrics_list: list[StepMetrics] = []
    with torch.no_grad():
        step = 1
        for step, data in tqdm(
            enumerate(training_objects.validation_dataloader, 1),
            desc=f"Epoch {epoch + 1} validation",
            total=epoch_len,
            unit="step",
            leave=False,
        ):
            clear_memory()
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

            del images

            step_metrics = StepMetrics.calculate_metrics(
                loss=loss.item(),  # type: ignore
                y=np.asarray(
                    [
                        training_objects.post_train_label_transform(_).detach().cpu()
                        for _ in decollate_batch(labels)
                    ]
                ),
                y_pred=np.asarray(
                    [
                        training_objects.post_train_transform(_).detach().cpu()
                        for _ in decollate_batch(outputs)
                    ]
                ),
                weights=training_parameters.loss_weights,
            )
            step_metrics_list.append(step_metrics)

            mlflow.log_metrics(
                step_metrics.to_dict(
                    split_labels=True,
                    labels=training_parameters.label_names,
                    prefix="train",
                ),
                step=epoch_len * epoch + step,
            )
            del loss

            del outputs
            del labels
            clear_memory()

        epoch_metrics = EpochMetrics.from_step_metrics(step_metrics_list)

        metrics_to_log, best_epoch = get_metrics_to_log(
            epoch + 1,
            "val",
            training_parameters,
            **epoch_metrics.to_dict(),
        )

        if best_epoch:
            torch.save(training_objects.model.state_dict(), model_path)
            if model_signature is not None:
                mlflow.pytorch.log_model(
                    training_objects.model,
                    name=get_model_artifact_path(epoch + 1),
                    signature=model_signature,
                    pip_requirements=pip_requirements,
                )
            _logger.info(f"Saved new best metric model: {model_path}")

        mlflow.log_metrics(
            metrics_to_log,
            step=epoch + 1,
        )

    return epoch_metrics, step_metrics_list, best_epoch


def get_metrics_to_log(
    epoch: int,
    stage: typing.Literal["train", "val"],
    training_parameters: TrainingParameters,
    **epoch_metrics: float,
) -> tuple[dict[str, float], bool]:
    # Calculate mean metric
    epoch_metrics["mean_of_key_metrics"] = np.mean(
        tuple(
            epoch_metrics[k]
            for k in getattr(training_parameters, f"key_{stage}_metrics")
        )
    ).item()

    best_epoch = training_parameters.update_metrics(
        epoch=epoch, metrics=epoch_metrics, stage=stage
    )

    metrics_to_log: dict[str, float] = {}
    for metric_name, values in training_parameters.current_metrics[stage].items():
        if metric_name == "epoch":
            # Don't log epoch to MLFlow or you'll get a graph of y=x.
            continue
        metric_key = f"{stage}_{metric_name}"  # Prepend stage to log train and val metrics separately
        if isinstance(values, np.ndarray):
            # If it's a numpy array, assume it's per-class and log those separately (and the mean)
            for label_name, v in zip(training_parameters.label_names, values):
                metrics_to_log[f"{metric_key}_{label_name}"] = v
            metrics_to_log[f"{metric_key}_mean"] = values.mean()
        else:
            metrics_to_log[metric_key] = values
    return metrics_to_log, best_epoch


def submit_images_to_mlflow(
    images: torch.Tensor | list[torch.Tensor],
    labels: torch.Tensor | list[torch.Tensor],
    predictions: torch.Tensor | list[torch.Tensor],
    step: int,
    max_dims: tuple[int, int] = (512, 512),
    timestamp: int | None = None,
    log_unlabelled: bool = False,
    separate_background: bool = False,
) -> None:
    _logger.debug("Sumbitting images to MLFlow")
    colours: list[tuple[int, int, int] | str] = [
        tuple((np.asarray(_tab10(_)[:3]) * 255).astype(int).tolist()) for _ in range(5)
    ]
    if separate_background:
        colours.insert(0, "none")
    for img, label, pred in zip(images, labels, predictions):
        img = img.to("cpu", copy=True)
        label = label.to("cpu", torch.bool, copy=True)
        pred = pred.to("cpu", torch.bool, copy=True)
        # Images must be handled last as full size RGB required for draw_segmentation_masks
        rgb_img = img.repeat((3, 1, 1))
        pred = to_pil_image(draw_segmentation_masks(rgb_img, pred, colors=colours))
        pred.thumbnail(max_dims)

        label = to_pil_image(draw_segmentation_masks(rgb_img, label, colors=colours))
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


def get_model_artifact_path(epoch: int) -> str:
    return f"epoch_{epoch}_model"


def clear_memory() -> None:
    """Helps avoid the memory usage gradually growing (especially GPU)"""
    with torch.no_grad():
        gc.collect()
        torch.cuda.empty_cache()
