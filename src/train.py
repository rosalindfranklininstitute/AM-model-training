from __future__ import annotations
import logging
import typing
from pathlib import Path

import monai.utils
import torch
from torch.utils.tensorboard import SummaryWriter

import monai
from monai import data, transforms, losses
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.visualize import plot_2d_or_3d_image


if typing.TYPE_CHECKING:
    from os import PathLike

__all__ = ["train"]


def train(
    training_data: data.ArrayDataset,
    validation_data: data.ArrayDataset,
    label_count: int,
    model_path: str | PathLike[str],
    epochs: int = 10,
) -> None:
    training_data_count = len(training_data)
    train_batch_size = 4
    check_loader = data.DataLoader(
        training_data,
        batch_size=10,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    first = monai.utils.misc.first(check_loader)
    assert first is not None, "DataLoader check failed"
    print(first[0].shape, first[1].shape)

    train_loader = data.DataLoader(
        training_data,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = data.DataLoader(
        validation_data,
        batch_size=1,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    dice_metric = DiceMetric(
        include_background=True, reduction="mean", get_not_nans=False
    )

    post_trans = transforms.Compose(
        [transforms.Activations(sigmoid=True), transforms.AsDiscrete(threshold=0.5)]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=label_count,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss_function = losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

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
        model.train()
        epoch_loss: float = 0
        step: int = 0
        for batch_data in train_loader:
            batch_data
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = training_data_count // train_batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                # val_images = None
                # val_labels = None
                val_outputs: tuple[typing.Any] | None = None
                for val_data in val_loader:
                    val_images, val_labels = (
                        val_data[0].to(device),
                        val_data[1].to(device),
                    )
                    roi_size = (96, 96)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(  # type: ignore[assignment]
                        val_images, roi_size, sw_batch_size, model
                    )

                    val_outputs = tuple(
                        post_trans(i) for i in data.decollate_batch(val_outputs)
                    )
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                if val_outputs is None:
                    raise TypeError("val_outputs is None")
                # aggregate the final mean dice result
                metric: float = dice_metric.aggregate().item()  # type: ignore[union-attr]
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), model_path)
                    print(f"Saved new best metric model: {model_path}")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(
                    val_outputs, epoch + 1, writer, index=0, tag="output"
                )

    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )
    writer.close()
