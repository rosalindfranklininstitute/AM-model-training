import logging
import typing
from datetime import datetime
from pathlib import Path
from os import PathLike

import numpy as np

import torch

import setup
import files
import train
import models
import losses

if typing.TYPE_CHECKING:
    from os import PathLike


_logger = logging.getLogger("adaptive_milling_training")


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

    num_classes = 5
    label_count = num_classes + 1 - int(include_background)
    foreground_labels = (1, 2, 3)

    df = files.paths_dataframe_from_csv(csv_path)

    training_data, validation_data = setup.create_datasets(
        df,
        image_size=image_size,
        label_count=label_count,
        validation_split=0.2,
        foreground_labels=foreground_labels,
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
        num_classes=num_classes,
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
        foreground_labels=foreground_labels,
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
    if loss_kwargs.get("weights") is not None:
        loss_kwargs["weights"] = losses.weights_to_tensor(
            loss_kwargs["weights"], device=device
        )

    model_creator = models.model_creation_functions[model_name]

    loss_function = losses.loss_creation_functions[loss_name](**loss_kwargs)

    _logger.info("The model will be saved as '%s'", model_path)
    _logger.info("Starting training...")

    steps_per_epoch = int(
        np.ceil(len(training_data) / training_parameters.training_batch_size)
    )

    model = model_creator(**model_kwargs)
    training_objects = setup.setup_training_objects(
        device,
        training_data=training_data,
        validation_data=validation_data,
        model=model,
        num_classes=label_count,
        loss_function=loss_function,
        learning_rate=training_parameters.learning_rate,
        lr_scheduler_class=torch.optim.lr_scheduler.OneCycleLR,
        # lr_scheduler_class=torch.optim.lr_scheduler.CyclicLR,
        lr_scheduler_kwargs={
            # "base_lr": training_parameters.learning_rate * 1e-4,
            # "mode": "triangular2",
            "max_lr": training_parameters.learning_rate,
            # "step_size_up": 600,
            "epochs": training_parameters.max_epochs,
            "steps_per_epoch": steps_per_epoch,
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
