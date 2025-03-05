from __future__ import annotations
import logging
import typing
from pathlib import Path

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

import mlflow

from monai.inferers import sliding_window_inference
# from monai.visualize import plot_2d_or_3d_image

from utils import MONAI_KEYS, TENSORBOARD_LOG_DIR as _TENSORBOARD_LOG_DIR

if typing.TYPE_CHECKING:
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

def train(
    training_objects: TrainingObjects,
    training_parameters: TrainingParameters,
) -> None:
    model_path = Path(training_parameters.model_path)

    train_stopper = EarlyStopper(patience=training_parameters.train_patience)
    val_stopper = EarlyStopper(patience=training_parameters.val_patience)

    # start a typical PyTorch training
    val_interval: int = 2
    best_metric: float = -1
    best_metric_epoch: int = -1
    epoch_loss_values: list[float] = list()
    writer = SummaryWriter(log_dir=_TENSORBOARD_LOG_DIR)

    with mlflow.start_run():
        # log model to mlflow
        input_array = np.random.uniform(
            size=(1, 1, *training_parameters.input_image_shape)
        ).astype(np.float32)
        signature = mlflow.models.infer_signature(
            input_array,
            training_objects.model(
                torch.from_numpy(input_array).to(training_objects.device)
            )
            .detach()
            .to("cpu")
            .numpy(),
        )
        mlflow.pytorch.log_model(
            training_objects.model,
            "model",
            signature=signature,
            pip_requirements=[
                f"-r {Path(__file__).absolute().parent.parent / 'requirements.txt'}"
            ],
        )

        mlflow.log_params(training_parameters.asdict())
        for epoch in range(training_parameters.max_epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{training_parameters.max_epochs}")
            training_objects.model.train()
            epoch_loss = 0
            step: int = 0

            for batch_data in training_objects.training_dataloader:
                batch_data
                step += 1
                inputs, labels = (
                    batch_data[MONAI_KEYS.IMAGE].to(training_objects.device),
                    batch_data[MONAI_KEYS.LABEL].to(training_objects.device),
                )
                training_objects.optimizer.zero_grad()
                outputs = training_objects.model(inputs)
                for i in range(outputs.size(0)):
                    outputs[i] = training_objects.post_val_transform(outputs[i])
                loss = training_objects.loss_function(outputs, labels)
                loss.backward()
                training_objects.optimizer.step()
                epoch_loss += loss.item()
                epoch_len = (
                    training_parameters.total_training_data
                    // training_parameters.training_batch_size
                )
                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                writer.add_scalar(
                    "train_loss", loss.item(), global_step=epoch_len * epoch + step
                )
                mlflow.log_metric(
                    "train_loss", loss.item(), step=epoch_len * epoch + step
                )

                calculate_batch_metrics(
                    outputs,
                    labels,
                    metrics_dict=training_objects.key_train_metrics,

                )

                if training_objects.additional_train_metrics is not None:
                    calculate_batch_metrics(
                        outputs,
                        labels,
                        metrics_dict=training_objects.additional_train_metrics,
                    )

            epoch_key_train_metrics: dict[str, float] = {}
            for metric_name, metric_fn in training_objects.key_train_metrics.items():
                epoch_key_train_metrics[f"{metric_name}"] = metric_fn.aggregate().item()
                metric_fn.reset()

            _ = tuple(epoch_key_train_metrics.values())
            epoch_key_train_metrics["mean_of_metrics"] = sum(_) / len(
                _
            )  # get mean metric

            training_parameters.update_metrics(
                epoch=epoch + 1, metrics_dict=epoch_key_train_metrics, stage="train"
            )

            writer.add_scalars(
                "epoch_key_train_metrics",
                epoch_key_train_metrics,
                global_step=epoch + 1,
            )

            if training_objects.additional_train_metrics is not None:
                epoch_additional_train_metrics: dict[str, float] = {}
                for (
                    metric_name,
                    metric_fn,
                ) in training_objects.additional_train_metrics.items():
                    epoch_additional_train_metrics[f"{metric_name}"] = (
                        metric_fn.aggregate().item()
                    )
                    metric_fn.reset()

                writer.add_scalars(
                    "epoch_additional_train_metrics",
                    epoch_additional_train_metrics,
                    global_step=epoch + 1,
                )

            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            _logger.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                training_objects.model.eval()
                with torch.no_grad():
                    val_outputs: tuple[typing.Any] | None = None
                    for val_data in training_objects.validation_dataloader:
                        val_images, val_labels = (
                            val_data[MONAI_KEYS.IMAGE].to(training_objects.device),
                            val_data[MONAI_KEYS.LABEL].to(training_objects.device),
                        )
                        roi_size = (96, 96)
                        sw_batch_size = 4
                        val_outputs = sliding_window_inference(  # type: ignore[assignment]
                            val_images, roi_size, sw_batch_size, training_objects.model
                        )

                        for i in range(val_outputs.size(0)):
                            val_outputs[i] = training_objects.post_val_transform(
                                val_outputs[i]
                            )

                        if val_outputs is None:
                            raise TypeError("val_outputs is None")

                        calculate_batch_metrics(
                            val_outputs,
                            val_labels,
                            metrics_dict=training_objects.key_val_metrics,
                        )

                        if training_objects.additional_val_metrics is not None:
                            calculate_batch_metrics(
                                outputs,
                                labels,
                                metrics_dict=training_objects.additional_val_metrics,
                            )

                    epoch_key_val_metrics: dict[str, float] = {}
                    for (
                        metric_name,
                        metric_fn,
                    ) in training_objects.key_val_metrics.items():
                        epoch_key_val_metrics[f"{metric_name}"] = (
                            metric_fn.aggregate().item()
                        )
                        metric_fn.reset()

                    _ = tuple(epoch_key_val_metrics.values())
                    epoch_key_val_metrics["mean_of_metrics"] = sum(_) / len(
                        _
                    )  # get mean metric

                    writer.add_scalars(
                        "epoch_key_val_metrics",
                        epoch_key_val_metrics,
                        global_step=epoch + 1,
                    )

                    if training_objects.additional_val_metrics is not None:
                        epoch_additional_val_metrics: dict[str, float] = {}
                        for (
                            metric_name,
                            metric_fn,
                        ) in training_objects.additional_val_metrics.items():
                            epoch_additional_val_metrics[f"{metric_name}"] = (
                                metric_fn.aggregate().item()
                            )
                            metric_fn.reset()

                        writer.add_scalars(
                            "epoch_additional_val_metrics",
                            epoch_additional_val_metrics,
                            global_step=epoch + 1,
                        )

                    if training_parameters.update_metrics(
                        epoch + 1, epoch_key_val_metrics, stage="val"
                    ):
                        torch.save(training_objects.model.state_dict(), model_path)
                        _logger.info(f"Saved new best metric model: {model_path}")
                    writer.add_scalars(
                        "epoch_key_val_metrics", epoch_key_val_metrics, epoch + 1
                    )
                    mlflow.log_metrics(
                        training_parameters.current_metrics, step=epoch + 1
                    )
                    mlflow.log_image(val_images, step=epoch + 1, key=MONAI_KEYS.IMAGE)
                    mlflow.log_image(val_labels, step=epoch + 1, key=MONAI_KEYS.LABEL)
                    mlflow.log_image(val_outputs, step=epoch + 1, key=MONAI_KEYS.PRED)

                    # # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                    # plot_2d_or_3d_image(
                    #     val_images, epoch + 1, writer, index=0, tag=MONAI_KEYS.IMAGE
                    # )
                    # plot_2d_or_3d_image(
                    #     val_labels, epoch + 1, writer, index=0, tag=MONAI_KEYS.LABEL
                    # )
                    # plot_2d_or_3d_image(
                    #     val_outputs,  # type: ignore[arg-type]
                    #     epoch + 1,
                    #     writer,
                    #     index=0,
                    #     tag="output",
                    # )

                    if train_stopper.stop_early(
                        epoch_key_train_metrics[training_parameters.key_metric]
                    ):
                        _logger.info(
                            "Stopped early after epoch %i due to train stopper",
                            epoch + 1,
                        )
                        break
                    if val_stopper.stop_early(
                        epoch_key_val_metrics[training_parameters.key_metric]
                    ):
                        _logger.info(
                            "Stopped early after epoch %i due to val stopper", epoch + 1
                        )
                        break

        print(
            f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
        )
        mlflow.log_params(
            training_parameters.asdict()
        )  # Update to include final values
        writer.close()
