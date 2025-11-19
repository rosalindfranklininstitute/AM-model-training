from __future__ import annotations
import gc
import logging
import typing
import math
from datetime import datetime
from pathlib import Path
from os import PathLike

import torch

from monai import optimizers

import matplotlib.pyplot as plt

from ap_model_training import setup
from ap_model_training import files
from ap_model_training import models
from ap_model_training import losses

if typing.TYPE_CHECKING:
    from os import PathLike
    from matplotlib.axes import Axes
    from ap_model_training.setup import TrainingObjects


_logger = logging.getLogger("adaptive_milling_training")


def get_device(cpu_only: bool = False) -> torch.device:
    return torch.device("cuda" if not cpu_only and torch.cuda.is_available() else "cpu")


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


def plot_learning_rates(
    csv_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    cpu_only: bool = False,
    models_to_ignore: list[str] | None = None,
    loss_name: str = "diceloss",
    iterations: int = 20,
    image_size: int = 1536,
    model_kwargs: dict[str, typing.Any] | None = None,
    loss_kwargs: dict[str, typing.Any] | None = None,
    gpu_number: int | None = None,
    include_background: bool = False,
) -> None:
    csv_path = Path(csv_path).absolute()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)

    if model_kwargs is None:
        model_kwargs = {}

    if loss_kwargs is None:
        loss_kwargs = {}

    _logger.info("Starting training with '%s'", csv_path)

    num_classes = 5

    foreground_labels = (1, 2, 3)

    df = files.paths_dataframe_from_csv(csv_path)

    training_data, validation_data = setup.create_datasets(
        df,
        image_size=image_size,
        validation_split=0.2,
        # dataset_type=CacheDataset,
    )
    _logger.info("Datasets loaded")

    device = setup.get_device(cpu_only, gpu=gpu_number)

    label_names = ("padding", "background", "gis", "lamella", "crack", "vacuum")

    training_parameters = setup.TrainingParameters(
        num_classes=num_classes,
        num_channels=3,
        label_names=label_names[1 - int(include_background) :],
        input_image_shape=(image_size, image_size),
        learning_rate=1e-3,
        best_metric="mean_of_key_metrics",
        key_train_metrics=[
            # "mean_iou",
            "mean_dice",
        ],
        key_val_metrics=[
            "mean_iou",
            "mean_dice",
        ],
        max_epochs=80,
        model_path="",
        train_patience=20,
        val_patience=10,
        total_training_data=len(training_data),
        total_validation_data=len(validation_data),
        foreground_labels=foreground_labels,
        training_batch_size=6,
        validation_batch_size=3,
        loss_weights=loss_kwargs.get("weights", None),
        include_background=include_background,
    )

    model_kwargs.update(
        {
            "label_count": training_parameters.num_classes,
            # + 1,  # for background (always included in model)
            "num_channels": training_parameters.num_channels,
            "input_image_size": training_parameters.input_image_shape,
        }
    )
    loss_kwargs.update(
        {
            "num_classes": training_parameters.num_classes,
            # + 1
            # - int(include_background),
            "include_background": training_parameters.include_background,
        }
    )
    if loss_kwargs.get("weights") is not None:
        loss_kwargs["weights"] = losses.weights_to_tensor(
            loss_kwargs["weights"], device=device
        )

    models_to_test = list(models.model_creation_functions.keys())
    if models_to_ignore is not None:
        for model_to_ignore in models_to_ignore:
            try:
                models_to_test.remove(model_to_ignore)
            except ValueError:
                logging.warning("'%s' is not a valid model to remove", model_to_ignore)

    nrows = int(math.floor(math.sqrt(len(models_to_test))))
    ncols = int(math.ceil(len(models_to_test) / nrows))
    lr_fig, lr_axs = plt.subplots(
        nrows, ncols, figsize=(15 * ncols, 15 * nrows), squeeze=True
    )

    for i, lr_ax in enumerate(lr_axs.flat):
        model_name = "Unset"
        if i >= len(models_to_test):
            lr_ax.set_axis_off()
            continue
        try:
            model_name = models_to_test[i]
            model = models.model_creation_functions[model_name](**model_kwargs)

            loss_function = losses.loss_creation_functions[loss_name](**loss_kwargs)

            training_objects = setup.setup_training_objects(
                device,
                training_data=training_data,
                validation_data=validation_data,
                loss_function=loss_function,
                model=model,
                num_classes=training_parameters.num_classes,
                include_background=include_background,
            )

            find_learning_rate(
                lr_ax, training_objects=training_objects, iterations=iterations
            )
            del model
            del loss_function
            clear_memory()

        except Exception:
            logging.error("Failed to find and plot learning rate", exc_info=True)
        lr_ax.set_title(f"{model_name}\n{lr_ax.get_title()}")

        lr_fig.canvas.draw()

    lr_fig.savefig(
        output_dir
        / f"{datetime.now().strftime('%y%m%d_%H%M%S')}_{csv_path.stem}_{loss_name}.png"
    )


def clear_memory() -> None:
    """Helps avoid the memory usage gradually growing (especially GPU)"""
    with torch.no_grad():
        gc.collect()
        torch.cuda.empty_cache()
