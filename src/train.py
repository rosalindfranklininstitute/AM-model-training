from __future__ import annotations
import logging
import typing
from dataclasses import dataclass, field, InitVar
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from monai.utils.misc import first
from monai import data, transforms, losses, optimizers, engines
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, MeanIoU
from monai.visualize import plot_2d_or_3d_image

if typing.TYPE_CHECKING:
    from os import PathLike
    from monai.metrics import Metric
    from matplotlib.axes import Axes


_logger = logging.getLogger("adaptive_milling_training")

def get_device(cpu_only: bool = False) -> torch.device:
    return torch.device("cuda" if not cpu_only and torch.cuda.is_available() else "cpu")


@dataclass
class TrainingObjects:
    training_data: InitVar[data.ArrayDataset]
    validation_data: InitVar[data.ArrayDataset]
    device: torch.DeviceObjType
    model: torch.nn.Module
    loss_function: losses._Loss
    optimizer: torch.optim.Optimizer
    key_train_metrics: dict[str, Metric]
    post_transform: transforms.Transform
    training_data_workers: InitVar[int] = 8
    validation_data_workers: InitVar[int] = 4
    training_batch_size: int = 4
    training_dataloader: data.Dataloader = field(init=False)
    validation_dataloader: data.Dataloader = field(init=False)
    num_training_data: int = field(init=False)
    num_validation_data: int = field(init=False)
    check_loaders: InitVar[bool] = True

    def __post_init__(
        self,
        training_data: data.ArrayDataset,
        validation_data: data.ArrayDataset,
        training_data_workers: int,
        validation_data_workers: int,
        check_loaders: bool,
    ) -> None:
        pin_memory = self.device.type == "cuda"

        if check_loaders:
            # Check data loads
            check_loader = data.DataLoader(
                training_data,
                batch_size=10,
                num_workers=2,
                pin_memory=pin_memory,
            )
            first_batch = first(check_loader)
            assert first_batch is not None, "DataLoader check failed"

        self.training_dataloader = data.DataLoader(
            training_data,
            batch_size=self.training_batch_size,
            shuffle=True,
            num_workers=training_data_workers,
            pin_memory=pin_memory,
        )

        self.validation_dataloader = data.DataLoader(
            validation_data,
            batch_size=1,
            shuffle=True,
            num_workers=validation_data_workers,
            pin_memory=pin_memory,
        )


def setup_training_objects(
    device: torch.DeviceObjType,
    training_data: data.ArrayDataset,
    validation_data: data.ArrayDataset,
    model: torch.nn.Module,
    loss_function: losses._Loss,
    learning_rate: float = 1e-4,
    **kwargs: typing.Any,
) -> TrainingObjects:
    key_train_metrics = {
        "Mean IoU": MeanIoU(include_background=True, reduction="mean"),
        "Dice": DiceMetric(include_background=True, reduction="mean"),
    }

    post_trans = transforms.Compose(
        [transforms.Activations(softmax=True), transforms.AsDiscrete(argmax=True)]
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return TrainingObjects(
        training_data,
        validation_data,
        device,
        model,
        loss_function,
        optimizer,
        key_train_metrics,
        post_trans,
        **kwargs,
    )


def find_learning_rate(
    ax: Axes,
    training_objects: TrainingObjects,
    lower_learning_rate: float = 1e-6,
    upper_learning_rate: float = 1e-2,
    iterations: int = 20,
) -> None:
    lr_finder = optimizers.LearningRateFinder(
        model=training_objects.model,
        optimizer=training_objects.optimizer,
        criterion=training_objects.loss_function,
        device=training_objects.device,
    )
    lr_finder.range_test(
        training_objects.training_dataloader,
        training_objects.validation_dataloader,
        start_lr=lower_learning_rate,
        end_lr=upper_learning_rate,
        num_iter=iterations,
    )
    # for grad, loss in zip(*lr_finder.get_lrs_and_losses())
    #     print(f"Gradient, loss: {grad}, {loss}")
    print(f"Steepest gradient, corresponding loss: {lr_finder.get_steepest_gradient()}")
    _ = lr_finder.plot(ax=ax)
    ax.set_title(
        f"Steepest gradient, corresponding loss: {lr_finder.get_steepest_gradient()}"
    )


def setup_training_engines(
    training_objects: TrainingObjects,
    model_path: str | PathLike[str],
    epochs: int = 10,
) -> tuple[engines.Trainer, engines.Evaluator]:
    model_path = Path(model_path)

    trainer = engines.SupervisedTrainer(
        training_objects.device,
        max_epochs=epochs,
        train_data_loader=training_objects.training_dataloader,
        network=training_objects.model,
        optimizer=training_objects.optimizer,
        loss_function=training_objects.loss_function,
        postprocessing=training_objects.post_transform,
        key_train_metric=training_objects.metric,
    )
    evaluator = engines.SupervisedEvaluator(
        training_objects.device,
        val_data_loader=training_objects.validation_dataloader,
        network=training_objects.model,
        postprocessing=training_objects.post_transform,
        key_val_metric=training_objects.metric,
    )
    return trainer, evaluator


def train(
    training_objects: TrainingObjects,
    model_path: str | PathLike[str],
    epochs: int = 10,
    cpu_only: bool = False,
) -> None:
    model_path = Path(model_path)

    device = get_device(cpu_only)

    # start a typical PyTorch training
    val_interval: int = 2
    best_metric: float = -1
    best_metric_epoch: int = -1
    epoch_loss_values: list[float] = list()
    metric_values: list[float] = list()
    writer = SummaryWriter()
    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        training_objects.model.train()
        epoch_loss: float = 0
        step: int = 0
        for batch_data in training_objects.training_dataloader:
            batch_data
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            training_objects.optimizer.zero_grad()
            outputs = training_objects.model(inputs)
            loss = training_objects.loss_function(outputs, labels)
            loss.backward()
            training_objects.optimizer.step()
            epoch_loss += loss.item()
            epoch_len = (
                training_objects.num_training_data
                // training_objects.training_batch_size
            )
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        _logger.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            training_objects.model.eval()
            with torch.no_grad():
                # val_images = None
                # val_labels = None
                val_outputs: tuple[typing.Any] | None = None
                for val_data in training_objects.validation_dataloader:
                    val_images, val_labels = (
                        val_data[0].to(device),
                        val_data[1].to(device),
                    )
                    roi_size = (96, 96)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(  # type: ignore[assignment]
                        val_images, roi_size, sw_batch_size, training_objects.model
                    )

                    val_outputs = tuple(
                        training_objects.post_transform(i)
                        for i in data.decollate_batch(val_outputs)
                    )
                    # compute metric for current iteration
                    training_objects.metric(y_pred=val_outputs, y=val_labels)
                if val_outputs is None:
                    raise TypeError("val_outputs is None")
                # aggregate the final mean dice result
                metric: float = training_objects.metric.aggregate().item()  # type: ignore[union-attr]
                # reset the status for next validation round
                training_objects.metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(training_objects.model.state_dict(), model_path)
                    _logger.info(f"Saved new best metric model: {model_path}")
                _logger.info(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(
                    val_outputs,  # type: ignore[arg-type]
                    epoch + 1,
                    writer,
                    index=0,
                    tag="output",
                )

    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )
    writer.close()
