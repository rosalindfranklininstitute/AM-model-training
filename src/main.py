import math
import logging
import typing
from datetime import datetime
from pathlib import Path
from os import PathLike

import matplotlib.pyplot as plt

import torch

import setup
import files
import train
import models
import losses

from monai.data import CacheDataset

if typing.TYPE_CHECKING:
    from os import PathLike


_logger = logging.getLogger("adaptive_milling_training")


def plot_learning_rates(
    csv_path: str | PathLike[str],
    cpu_only: bool = False,
    models_to_ignore: list[str] | None = None,
    loss_name: str = "diceloss",
    iterations: int = 20,
    image_size: int = 1536,
    model_kwargs: dict[str, typing.Any] | None = None,
    loss_kwargs: dict[str, typing.Any] | None = None,
) -> None:
    csv_path = Path(csv_path).absolute()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    if model_kwargs is None:
        model_kwargs = {}

    if loss_kwargs is None:
        loss_kwargs = {}

    _logger.info("Starting training with '%s'", csv_path)

    label_count = 5

    df = files.paths_dataframe_from_csv(csv_path)

    training_data, validation_data = setup.create_datasets(
        df,
        image_size=image_size,
        label_count=label_count,
        validation_split=0.2,
        # dataset_type=CacheDataset,
    )
    _logger.info("Datasets loaded")

    device = setup.get_device(cpu_only)

    lr_dir = Path.cwd().parent / "logs" / "learning_rate"
    lr_dir.mkdir(exist_ok=True)

    model_kwargs: dict[str, typing.Any] = {
        "label_count": 5,
        "input_image_size": (image_size, image_size),
        **model_kwargs,
    }
    loss_kwargs: dict[str, typing.Any] = {"num_classes": label_count, **loss_kwargs}

    if loss_kwargs["weights"] is not None:
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

    nrows = min(math.floor(len(models_to_test) / 3) + 1, 3)
    ncols = math.ceil(len(models_to_test) / nrows)
    lr_fig, lr_axs = plt.subplots(nrows, ncols, figsize=(15 * ncols, 15 * nrows))

    for lr_ax, model_name in zip(lr_axs.ravel(), models_to_test):
        try:
            model_kwargs: dict[str, typing.Any] = {
                "label_count": 5,
                "input_image_size": (image_size, image_size),
                **model_kwargs,
            }
            model = models.model_creation_functions[model_name](**model_kwargs)

            loss_function = losses.loss_creation_functions[loss_name](**loss_kwargs)

            training_objects = setup.setup_training_objects(
                device,
                training_data=training_data,
                validation_data=validation_data,
                loss_function=loss_function,
                model=model,
                num_classes=label_count,
                # training_data_workers=0,
                # validation_data_workers=0,
            )

            train.find_learning_rate(
                lr_ax, training_objects=training_objects, iterations=iterations
            )
        except Exception:
            logging.error("Failed to find and plot learning rate", exc_info=True)
        lr_ax.set_title(f"{model_name}\n{lr_ax.get_title()}")

        lr_fig.canvas.draw()

    lr_fig.savefig(
        lr_dir / f"{datetime.now().strftime('%y%m%d_%H%M%S')}_{csv_path.stem}.png"
    )


def run_training(
    csv_path: str | PathLike[str],
    model_name: str,
    loss_name: str = "diceloss",
    epochs: int = 30,
    learning_rate: float = 1e-4,
    cpu_only: bool = False,
    image_size: int = 1536,
    frozen_epochs: int = 0,
    model_kwargs: dict[str, typing.Any] | None = None,
    loss_kwargs: dict[str, typing.Any] | None = None,
    include_background: bool = False,
    gpu_number: int | None = None,
) -> None:
    if model_kwargs is None:
        model_kwargs = {}

    if loss_kwargs is None:
        loss_kwargs = {}

    csv_path = Path(csv_path).absolute()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    _logger.info("Starting training with '%s'", csv_path)

    label_count = 5 + 1 - int(include_background)

    df = files.paths_dataframe_from_csv(csv_path)

    training_data, validation_data = setup.create_datasets(
        df,
        image_size=image_size,
        label_count=label_count,
        validation_split=0.2,
    )
    _logger.info("Datasets loaded")

    device = setup.get_device(cpu_only, gpu=gpu_number)

    models_dir = Path.cwd().parent / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = (
        models_dir
        / f"{datetime.now().strftime('%y%m%d_%H%M%S')}_{csv_path.stem}_{model_name}.pth"
    )

    training_parameters = setup.TrainingParameters(
        num_classes=5,
        label_names=("background", "gis", "lamella", "crack", "void"),
        input_image_shape=(image_size, image_size),
        learning_rate=learning_rate,
        best_metric="mean_of_key_metrics",
        key_train_metrics=[
            # "mean_iou",
            "mean_dice",
        ],
        key_val_metrics=[
            "mean_iou",
            "mean_dice",
        ],
        max_epochs=epochs,
        model_path=model_path,
        train_patience=20,
        val_patience=10,
        total_training_data=len(training_data),
        total_validation_data=len(validation_data),
        training_batch_size=6,
        validation_batch_size=3,
        frozen_epochs=frozen_epochs,
        loss_weights=loss_kwargs.get("weights", None),
    )

    model_kwargs: dict[str, typing.Any] = {
        "label_count": label_count,
        "input_image_size": (image_size, image_size),
        **model_kwargs,
    }
    loss_kwargs: dict[str, typing.Any] = {
        "num_classes": training_parameters.num_classes,
        "include_background": include_background,
        **loss_kwargs,
    }
    if loss_kwargs["weights"] is not None:
        loss_kwargs["weights"] = losses.weights_to_tensor(
            loss_kwargs["weights"], device=device
        )

    model_creator = models.model_creation_functions[model_name]

    loss_function = losses.loss_creation_functions[loss_name](**loss_kwargs)

    _logger.info("The model will be saved as '%s'", model_path)
    _logger.info("Starting training...")

    steps_per_epoch = len(training_data) // training_parameters.training_batch_size

    model = model_creator(**model_kwargs)
    training_objects = setup.setup_training_objects(
        device,
        training_data=training_data,
        validation_data=validation_data,
        model=model,
        num_classes=label_count,
        loss_function=loss_function,
        learning_rate=learning_rate,
        lr_scheduler_class=torch.optim.lr_scheduler.CyclicLR,  # torch.optim.lr_scheduler.OneCycleLR,
        lr_scheduler_kwargs={
            "base_lr": training_parameters.learning_rate * 1e-4,
            "mode": "triangular2",
            "max_lr": training_parameters.learning_rate,
            # "epochs": training_parameters.max_epochs,
            # "steps_per_epoch": steps_per_epoch,
        },
        # lr_scheduler_kwargs={"warmup_steps": 5, "t_total": epochs},
    )
    # training_engine, evaluation_engine = train.setup_training_engines(
    #     training_objects,
    #     model_path=model_path,
    #     epochs=epochs,
    # )
    # if use_mlflow:
    #     from setup_mlflow import setup_mlflow

    #     setup_mlflow(training_engine, evaluation_engine)
    # training_engine.run()
    train.run(
        training_objects,
        training_parameters=training_parameters,
    )
