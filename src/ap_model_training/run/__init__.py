import logging
import typing
from datetime import datetime
from pathlib import Path
from os import PathLike

import numpy as np

from monai.data.dataset import Dataset
from monai.utils import set_determinism

from ap_model_training import setup
from ap_model_training import files
from ap_model_training.run import train, validate
from ap_model_training import models
from ap_model_training import losses

import mlflow

if typing.TYPE_CHECKING:
    from os import PathLike
    from pandas import DataFrame

_logger = logging.getLogger(__package__)


def _load_csv(path: str | PathLike[str]) -> DataFrame:
    csv_path = Path(path).absolute()
    if not csv_path.is_file():
        raise FileNotFoundError(path)

    return files.paths_dataframe_from_csv(path)


def setup_training(
    output_path: str | PathLike[str],
    run_id: str,
    model_name: str,
    csv_path: str | PathLike[str] | None = None,
    training_csv_path: str | PathLike[str] | None = None,
    validation_csv_path: str | PathLike[str] | None = None,
    dataset_type: type[Dataset] = Dataset,
    dataset_kwargs: dict[str, typing.Any] | None = None,
    image_size: int = 1536,
    pad: bool = False,
    rgb: bool = True,
    validation_split: float = 0.15,
    validation_interval: int = 1,
    loss_name: str = "compound_loss",
    epochs: int = 100,
    learning_rate: float = 1e-5,
    lr_scheduler_name: str = "onecyclelr",
    cpu_only: bool = False,
    frozen_epochs: int = 0,
    model_kwargs: dict[str, typing.Any] | None = None,
    loss_kwargs: dict[str, typing.Any] | None = None,
    lr_scheduler_kwargs: dict[str, typing.Any] | None = None,
    include_background: bool = True,
    gpu_number: int | None = None,
    training_batch_size: int = 6,
    validation_batch_size: int = 1,
    num_training_workers: int | None = None,
    num_validation_workers: int | None = None,
    seed: int = 42,
) -> tuple[setup.TrainingObjects, setup.TrainingParameters]:
    # Fix determinism for consistent results
    set_determinism(seed=seed)

    if lr_scheduler_kwargs is None:
        lr_scheduler_kwargs = {}

    if model_kwargs is None:
        model_kwargs = {}

    if loss_kwargs is None:
        loss_kwargs = {}

    if dataset_kwargs is None:
        dataset_kwargs = {}

    if csv_path is None:
        if training_csv_path is None or validation_csv_path is None:
            raise ValueError(
                "Either 'csv_path' or 'training_csv_path' and 'validation_csv_path' must be given."
            )
        training_data = setup.create_dataset(
            _load_csv(training_csv_path),
            image_size=image_size,
            augmentations=True,
            dataset_type=dataset_type,
            pad=pad,
            rgb=rgb,
            **dataset_kwargs,
        )
        _logger.info("Training dataset loaded from %s", training_csv_path)
        validation_data = setup.create_dataset(
            _load_csv(validation_csv_path),
            image_size=image_size,
            augmentations=False,
            dataset_type=dataset_type,
            pad=pad,
            rgb=rgb,
            **dataset_kwargs,
        )
        _logger.info("Validation dataset loaded from %s", validation_csv_path)
    else:
        training_data, validation_data = setup.create_datasets(
            _load_csv(csv_path),
            image_size=image_size,
            validation_split=validation_split,
            dataset_type=dataset_type,
            pad=pad,
            rgb=rgb,
            **dataset_kwargs,
        )
        _logger.info("Datasets loaded from %s", csv_path)

    num_classes = 5
    foreground_labels = (1, 2, 3)  # before background is added

    device = setup.get_device(cpu_only, gpu=gpu_number)

    label_names = ("padding", "background", "gis", "lamella", "crack", "vacuum")

    training_parameters = setup.TrainingParameters(
        output_path=output_path,
        run_id=run_id,
        model_name=model_name,
        num_classes=num_classes,
        num_channels=3 if rgb else 1,
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
        seed=seed,
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
        ).detach()

    model_creator = models.model_creation_functions[model_name]

    loss_function = losses.loss_creation_functions[loss_name](**loss_kwargs)

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
        training_batch_size=training_batch_size,
        validation_batch_size=validation_batch_size,
        num_training_workers=num_training_workers,
        num_validation_workers=num_validation_workers,
        learning_rate=training_parameters.learning_rate,
        lr_scheduler_name=lr_scheduler_name,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
    )
    return training_objects, training_parameters


def run_training(
    output_path: str | PathLike[str],
    csv: str | PathLike[str] | tuple[str | PathLike[str], str | PathLike[str]],
    cpu_only: bool = False,
    gpu_number: int | None = None,
    validation_interval: int = 1,
    epochs: int = 100,
    frozen_epochs: int = 25,
    model_name: str = "smp_fpn",
    loss_name: str = "compound_loss",
    lr_scheduler_name: str = "onecyclelr",
    learning_rate: float = 1e-5,
    validation_split: float = 0.15,
    image_size: int = 1536,
    pad_images: bool = False,
    rgb_images: bool = True,
    dataset_type: type[Dataset] = Dataset,
    dataset_kwargs: dict[str, typing.Any] | None = None,
    model_kwargs: dict[str, typing.Any] | None = None,
    loss_kwargs: dict[str, typing.Any] | None = None,
    include_background: bool = True,
    lr_scheduler_kwargs: dict[str, typing.Any] | None = None,
    training_batch_size: int = 6,
    validation_batch_size: int = 1,
    num_training_workers: int | None = None,
    num_validation_workers: int | None = None,
    log_mlflow: bool = False,
    submit_training_images: bool = False,
    seed: int = 42,
) -> None:
    output_path = Path(output_path)

    if isinstance(csv, tuple):
        csv_path = None
        training_csv_path = Path(csv[0])
        validation_csv_path = Path(csv[1])
        input_name = f"{training_csv_path.stem}_{validation_csv_path.stem}"
    else:
        csv_path = Path(csv)
        training_csv_path = None
        validation_csv_path = None
        input_name = csv_path.stem

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    timestamp_subdirectory = output_path / timestamp
    try:
        timestamp_subdirectory.mkdir()
    except OSError:
        _logger.error(
            "Failed to create timestamp subdirectory '%s'",
            str(timestamp_subdirectory),
            exc_info=True,
        )
    training_objects, training_parameters = setup_training(
        output_path=timestamp_subdirectory,
        run_id=f"{timestamp}_{input_name}_{model_name}",
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
        num_training_workers=num_training_workers,
        num_validation_workers=num_validation_workers,
        dataset_type=dataset_type,
        dataset_kwargs=dataset_kwargs,
        seed=seed,
    )
    train.run(
        training_objects,
        training_parameters=training_parameters,
        submit_training_images=submit_training_images,
        log_mlflow=log_mlflow,
    )


def setup_evaluation(
    output_path: str | PathLike[str],
    weights_file: str | PathLike[str],
    run_id: str,
    model_name: str,
    csv_path: str | PathLike[str],
    dataset_type: type[Dataset] = Dataset,
    dataset_kwargs: dict[str, typing.Any] | None = None,
    image_size: int = 1536,
    pad: bool = False,
    rgb: bool = True,
    loss_name: str = "compound_loss",
    cpu_only: bool = False,
    model_kwargs: dict[str, typing.Any] | None = None,
    loss_kwargs: dict[str, typing.Any] | None = None,
    include_background: bool = True,
    gpu_number: int | None = None,
    batch_size: int = 1,
    num_workers: int | None = None,
    seed: int = 42,
) -> tuple[setup.EvaluationObjects, setup.EvaluationParameters]:
    # Fix determinism for consistent results
    set_determinism(seed=seed)

    if model_kwargs is None:
        model_kwargs = {}
    model_kwargs["weights_file"] = weights_file

    if loss_kwargs is None:
        loss_kwargs = {}

    if dataset_kwargs is None:
        dataset_kwargs = {}

    data = setup.create_dataset(
        _load_csv(csv_path),
        image_size=image_size,
        augmentations=False,
        dataset_type=dataset_type,
        pad=pad,
        rgb=rgb,
        **dataset_kwargs,
    )
    _logger.info("Dataset loaded from %s", csv_path)

    num_classes = 5
    foreground_labels = (1, 2, 3)  # before background is added

    device = setup.get_device(cpu_only, gpu=gpu_number)

    label_names = ("padding", "background", "gis", "lamella", "crack", "vacuum")

    evaluation_parameters = setup.EvaluationParameters(
        output_path=output_path,
        run_id=run_id,
        num_classes=num_classes,
        num_channels=3 if rgb else 1,
        label_names=label_names[2 - int(pad) - int(include_background) :],
        input_image_shape=(image_size, image_size),
        weights_file=weights_file,
        total_data=len(data),
        batch_size=batch_size,
        key_metrics=[
            "epoch_weighted_average_iou",
            "epoch_weighted_average_dice",
        ],
        foreground_labels=foreground_labels,
        loss_weights=loss_kwargs.get("weights", None),
        include_background=include_background,
        seed=seed,
    )

    model_kwargs.update(
        {
            "label_count": evaluation_parameters.num_classes,
            # + 1,  # for background (always included in model)
            "input_image_size": evaluation_parameters.input_image_shape,
        }
    )
    loss_kwargs.update(
        {
            "num_classes": evaluation_parameters.num_classes
            + 1
            - int(include_background),
            "include_background": evaluation_parameters.include_background,
        }
    )
    if loss_kwargs.get("weights") is not None:
        loss_kwargs["weights"] = losses.weights_to_tensor(
            loss_kwargs["weights"], device=device
        ).detach()

    model_creator = models.model_creation_functions[model_name]

    loss_function = losses.loss_creation_functions[loss_name](**loss_kwargs)

    _logger.info("Starting evaluation...")

    model = model_creator(**model_kwargs)
    evaluation_objects = setup.setup_evaulation_objects(
        device,
        data=data,
        num_classes=num_classes,
        model=model,
        loss_function=loss_function,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return evaluation_objects, evaluation_parameters


def run_evaluation(
    output_path: str | PathLike[str],
    weights_file: str | PathLike[str],
    model_name: str,
    csv: str | PathLike[str],
    dataset_type: type[Dataset] = Dataset,
    dataset_kwargs: dict[str, typing.Any] | None = None,
    image_size: int = 1536,
    pad_images: bool = False,
    rgb_images: bool = True,
    loss_name: str = "compound_loss",
    cpu_only: bool = False,
    model_kwargs: dict[str, typing.Any] | None = None,
    loss_kwargs: dict[str, typing.Any] | None = None,
    include_background: bool = True,
    gpu_number: int | None = None,
    batch_size: int = 1,
    num_workers: int | None = None,
    log_mlflow: bool = False,
    submit_training_images: bool = False,
    seed: int = 42,
) -> None:
    weights_file = Path(weights_file)
    output_path = Path(output_path)
    csv_path = Path(csv)
    input_name = csv_path.stem

    timestamp_subdirectory = output_path / f"{weights_file.stem}_{input_name}"
    try:
        timestamp_subdirectory.mkdir()
    except OSError:
        _logger.error(
            "Failed to create timestamp subdirectory '%s'",
            str(timestamp_subdirectory),
            exc_info=True,
        )

    evaulation_objects, evaluation_parameters = setup_evaluation(
        output_path=output_path,
        weights_file=weights_file,
        run_id=f"{model_name}_{weights_file.stem}_{input_name}",
        model_name=model_name,
        csv_path=csv_path,
        image_size=image_size,
        pad=pad_images,
        rgb=rgb_images,
        loss_name=loss_name,
        cpu_only=cpu_only,
        model_kwargs=model_kwargs,
        loss_kwargs=loss_kwargs,
        include_background=include_background,
        gpu_number=gpu_number,
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_type=dataset_type,
        dataset_kwargs=dataset_kwargs,
        seed=seed,
    )
    validate.evaluate(
        evaluation_objects=evaulation_objects,
        evaluation_parameters=evaluation_parameters,
        log_mlflow=log_mlflow,
    )


def submit_validation_for_mlflow_run(
    mlflow_run_id: str,
    output_path: str | PathLike[str],
    run_id: str,
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
    dataset_type: type[Dataset] = Dataset,
    dataset_kwargs: dict[str, typing.Any] | None = None,
    model_kwargs: dict[str, typing.Any] | None = None,
    loss_kwargs: dict[str, typing.Any] | None = None,
    lr_scheduler_kwargs: dict[str, typing.Any] | None = None,
    include_background: bool = False,
    gpu_number: int | None = None,
    seed: int = 42,
) -> None:
    training_objects, training_parameters = setup_training(
        output_path=output_path,
        run_id=run_id,
        csv_path=csv_path,
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
        dataset_type=dataset_type,
        dataset_kwargs=dataset_kwargs,
        seed=seed,
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
