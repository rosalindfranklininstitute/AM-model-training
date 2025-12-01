import logging
import typing
from datetime import datetime
from pathlib import Path
from os import PathLike

import numpy as np

from ap_model_training import setup
from ap_model_training import files
from ap_model_training import train
from ap_model_training import models
from ap_model_training import losses

import mlflow

if typing.TYPE_CHECKING:
    from os import PathLike


_logger = logging.getLogger(__package__)


def _load_csv(path: str | PathLike[str]):
    csv_path = Path(path).absolute()
    if not csv_path.is_file():
        raise FileNotFoundError(path)

    return files.paths_dataframe_from_csv(path)


def setup_training(
    model_save_path: str | PathLike[str],
    model_name: str,
    csv_path: str | PathLike[str] | None = None,
    training_csv_path: str | PathLike[str] | None = None,
    validation_csv_path: str | PathLike[str] | None = None,
    image_size: int = 1536,
    pad: bool = True,
    rgb: bool = True,
    validation_split: float = 0.15,
    validation_interval: int = 2,
    loss_name: str = "diceloss",
    epochs: int = 30,
    learning_rate: float = 1e-4,
    lr_scheduler_name: str = "onecyclelr",
    cpu_only: bool = False,
    frozen_epochs: int = 0,
    model_kwargs: dict[str, typing.Any] | None = None,
    loss_kwargs: dict[str, typing.Any] | None = None,
    lr_scheduler_kwargs: dict[str, typing.Any] | None = None,
    include_background: bool = False,
    gpu_number: int | None = None,
    training_batch_size: int = 2,
    validation_batch_size: int = 2,
) -> tuple[setup.TrainingObjects, setup.TrainingParameters]:
    if lr_scheduler_kwargs is None:
        lr_scheduler_kwargs = {}

    if model_kwargs is None:
        model_kwargs = {}

    if loss_kwargs is None:
        loss_kwargs = {}

    if csv_path is None:
        if training_csv_path is None or validation_csv_path is None:
            raise ValueError(
                "Either 'csv_path' or 'training_csv_path' and 'validation_csv_path' must be given."
            )
        training_data = setup.create_dataset(
            _load_csv(training_csv_path),
            image_size=image_size,
            augmentations=True,
            pad=pad,
            rgb=rgb,
        )
        _logger.info("Training dataset loaded from %s", training_csv_path)
        validation_data = setup.create_dataset(
            _load_csv(validation_csv_path),
            image_size=image_size,
            augmentations=False,
            pad=pad,
            rgb=rgb,
        )
        _logger.info("Validation dataset loaded from %s", validation_csv_path)
    else:
        training_data, validation_data = setup.create_datasets(
            _load_csv(csv_path),
            image_size=image_size,
            validation_split=validation_split,
            pad=pad,
            rgb=rgb,
        )
        _logger.info("Datasets loaded from %s", csv_path)

    num_classes = 5
    foreground_labels = (1, 2, 3)  # before background is added

    device = setup.get_device(cpu_only, gpu=gpu_number)

    label_names = ("padding", "background", "gis", "lamella", "crack", "vacuum")

    training_parameters = setup.TrainingParameters(
        num_classes=num_classes,
        num_channels=3,
        label_names=label_names[2 - int(pad) - int(include_background) :],
        input_image_shape=(image_size, image_size),
        learning_rate=learning_rate,
        best_metric="mean_of_key_metrics",
        key_train_metrics=[
            "epoch_weighted_average_iou",
            "epoch_weighted_average_dice",
        ],
        key_val_metrics=[
            "epoch_weighted_average_iou",
            "epoch_weighted_average_dice",
        ],
        max_epochs=epochs,
        model_path=model_save_path,
        train_patience=20,
        val_patience=10,
        total_training_data=len(training_data),
        total_validation_data=len(validation_data),
        foreground_labels=foreground_labels,
        training_batch_size=training_batch_size,
        validation_batch_size=validation_batch_size,
        frozen_epochs=frozen_epochs,
        loss_weights=loss_kwargs.get("weights", None),
        include_background=include_background,
        val_interval=validation_interval,
    )

    model_kwargs.update(
        {
            "label_count": training_parameters.num_classes,
            # + 1,  # for background (always included in model)
            "input_image_size": training_parameters.input_image_shape,
        }
    )
    loss_kwargs.update(
        {
            "num_classes": training_parameters.num_classes
            + 1
            - int(include_background),
            "include_background": training_parameters.include_background,
        }
    )
    if loss_kwargs.get("weights") is not None:
        loss_kwargs["weights"] = losses.weights_to_tensor(
            loss_kwargs["weights"], device=device
        )

    model_creator = models.model_creation_functions[model_name]

    loss_function = losses.loss_creation_functions[loss_name](**loss_kwargs)

    _logger.info("The model will be saved as '%s'", model_save_path)
    _logger.info("Starting training...")

    steps_per_epoch = int(
        np.ceil(len(training_data) / training_parameters.training_batch_size)
    )

    lr_scheduler_kwargs = {
        # "max_lr": learning_rate,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        **lr_scheduler_kwargs,
    }

    model = model_creator(**model_kwargs)
    training_objects = setup.setup_training_objects(
        device,
        training_data=training_data,
        validation_data=validation_data,
        model=model,
        num_classes=num_classes,
        loss_function=loss_function,
        learning_rate=training_parameters.learning_rate,
        lr_scheduler_name=lr_scheduler_name,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
    )
    return training_objects, training_parameters


def run_training(
    models_dir: str | PathLike[str],
    model_name: str,
    csv_path: str | PathLike[str] | None = None,
    training_csv_path: str | PathLike[str] | None = None,
    validation_csv_path: str | PathLike[str] | None = None,
    loss_name: str = "diceloss",
    lr_scheduler_name: str = "onecyclelr",
    epochs: int = 30,
    learning_rate: float = 1e-4,
    cpu_only: bool = False,
    validation_split: float = 0.15,
    validation_interval: int = 2,
    image_size: int = 1536,
    pad_images: bool = True,
    rgb_images: bool = True,
    frozen_epochs: int = 0,
    model_kwargs: dict[str, typing.Any] | None = None,
    loss_kwargs: dict[str, typing.Any] | None = None,
    include_background: bool = False,
    gpu_number: int | None = None,
    lr_scheduler_kwargs: dict[str, typing.Any] | None = None,
    training_batch_size: int = 2,
    validation_batch_size: int = 2,
) -> None:
    models_dir = Path(models_dir)

    if csv_path is None:
        if training_csv_path is None or validation_csv_path is None:
            raise ValueError(
                "Either 'csv_path' or 'training_csv_path' and 'validation_csv_path' must be given."
            )
        input_name = f"{Path(training_csv_path).stem}_{Path(validation_csv_path).stem}"
    else:
        input_name = Path(csv_path).stem

    model_save_path = (
        models_dir
        / f"{datetime.now().strftime('%y%m%d_%H%M%S')}_{input_name}_{model_name}.pth"
    )
    training_objects, training_parameters = setup_training(
        model_save_path=model_save_path,
        model_name=model_name,
        csv_path=csv_path,
        training_csv_path=training_csv_path,
        validation_csv_path=validation_csv_path,
        image_size=image_size,
        pad=pad_images,
        rgb=rgb_images,
        validation_split=validation_split,
        validation_interval=validation_interval,
        loss_name=loss_name,
        lr_scheduler_name=lr_scheduler_name,
        epochs=epochs,
        learning_rate=learning_rate,
        cpu_only=cpu_only,
        frozen_epochs=frozen_epochs,
        model_kwargs=model_kwargs,
        loss_kwargs=loss_kwargs,
        include_background=include_background,
        gpu_number=gpu_number,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        training_batch_size=training_batch_size,
        validation_batch_size=validation_batch_size,
    )
    train.run(
        training_objects,
        training_parameters=training_parameters,
    )


def submit_validation_for_mlflow_run(
    mlflow_run_id: str,
    model_save_path: str | PathLike[str],
    epoch: int,
    csv_path: str | PathLike[str],
    model_name: str,
    loss_name: str = "diceloss",
    lr_scheduler_name: str = "onecyclelr",
    epochs: int = 30,
    learning_rate: float = 1e-4,
    cpu_only: bool = False,
    image_size: int = 1536,
    frozen_epochs: int = 0,
    model_kwargs: dict[str, typing.Any] | None = None,
    loss_kwargs: dict[str, typing.Any] | None = None,
    lr_scheduler_kwargs: dict[str, typing.Any] | None = None,
    include_background: bool = False,
    gpu_number: int | None = None,
) -> None:
    training_objects, training_parameters = setup_training(
        csv_path=csv_path,
        model_save_path=model_save_path,
        model_name=model_name,
        loss_name=loss_name,
        epochs=epochs,
        learning_rate=learning_rate,
        lr_scheduler_name=lr_scheduler_name,
        cpu_only=cpu_only,
        image_size=image_size,
        frozen_epochs=frozen_epochs,
        model_kwargs=model_kwargs,
        loss_kwargs=loss_kwargs,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        include_background=include_background,
        gpu_number=gpu_number,
    )
    if mlflow_run_id is None:
        last_run = mlflow.last_active_run()
        if last_run is None:
            raise ValueError("Failed to get last MLFlow run")
        mlflow_run_id = last_run.info.run_id()

    with mlflow.start_run(mlflow_run_id):
        train.submit_validation_images_to_mflow(
            training_objects, training_parameters, epoch=epoch
        )
