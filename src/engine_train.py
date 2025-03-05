from __future__ import annotations
import logging
import typing
from dataclasses import dataclass, field, InitVar
from pathlib import Path

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from ignite import metrics as ignite_metrics

from monai.utils.misc import first
from monai import data, transforms, losses, optimizers, engines
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, MeanIoU
from monai.visualize import plot_2d_or_3d_image
from monai.handlers import (
    CheckpointSaver,
    EarlyStopHandler,
    LrScheduleHandler,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    from_engine,
)
from monai.inferers import SimpleInferer, SlidingWindowInferer


from utils import MONAI_KEYS, TENSORBOARD_LOG_DIR as _TENSORBOARD_LOG_DIR

if typing.TYPE_CHECKING:
    from os import PathLike
    from setup import TrainingObjects


_logger = logging.getLogger("adaptive_milling_training")


def setup_validation_handlers(training_objects: TrainingObjects) -> list[object]:
    return [
        # apply “EarlyStop” logic based on the validation metrics
        EarlyStopHandler(
            trainer=None,
            patience=5,
            score_function=lambda x: np.mean(
                [x.state.metrics[k] for k in training_objects.key_val_metrics.keys()]
            ),
        ),
        # use the logger "train_log" defined at the beginning of this program
        StatsHandler(name="train_log", output_transform=lambda x: None),
        TensorBoardStatsHandler(
            log_dir=_TENSORBOARD_LOG_DIR, output_transform=lambda x: None
        ),
        TensorBoardImageHandler(
            log_dir=_TENSORBOARD_LOG_DIR,
            batch_transform=from_engine([MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL]),
            output_transform=from_engine([MONAI_KEYS.PRED]),
        ),
        CheckpointSaver(
            save_dir=_TENSORBOARD_LOG_DIR,
            save_dict={"net": training_objects.model},
            save_key_metric=True,
        ),
    ]


def setup_training_handlers(training_objects: TrainingObjects) -> list[object]:
    return [
        # apply “EarlyStop” logic based on the loss value, use “-” negative value because smaller loss is better
        EarlyStopHandler(
            trainer=None,
            patience=20,
            score_function=lambda x: -x.state.output[0][MONAI_KEYS.LOSS],
            epoch_level=False,
        ),
        LrScheduleHandler(lr_scheduler=training_objects.lr_scheduler, print_lr=True),
        ValidationHandler(validator=None, interval=2, epoch_level=True),
        # use the logger "train_log" defined at the beginning of this program
        StatsHandler(
            name="train_log",
            tag_name="train_loss",
            output_transform=from_engine([MONAI_KEYS.LOSS], first=True),
        ),
        TensorBoardStatsHandler(
            log_dir=_TENSORBOARD_LOG_DIR,
            tag_name="train_loss",
            output_transform=from_engine([MONAI_KEYS.LOSS], first=True),
        ),
        CheckpointSaver(
            save_dir=_TENSORBOARD_LOG_DIR,
            save_dict={
                "net": training_objects.model,
                "opt": training_objects.optimizer,
            },
            save_interval=2,
            epoch_level=True,
        ),
    ]


def setup_training_engines(
    training_objects: TrainingObjects,
    model_path: str | PathLike[str],
    epochs: int = 10,
) -> tuple[engines.Trainer, engines.Evaluator]:
    model_path = Path(model_path)

    val_handlers = setup_validation_handlers(training_objects=training_objects)

    val_epoch_length = (
        training_objects.num_validation_data // training_objects.validation_batch_size
    )
    evaluator = engines.SupervisedEvaluator(
        training_objects.device,
        val_data_loader=training_objects.validation_dataloader,
        network=training_objects.model,
        postprocessing=training_objects.post_val_transform,
        key_val_metric=training_objects.key_train_metrics,
        additional_metrics=training_objects.additional_val_metrics,
        val_handlers=val_handlers,
        inferer=SlidingWindowInferer((96, 96), sw_batch_size=4),
        decollate=True,
        epoch_length=val_epoch_length,
        amp=False,
    )

    train_handlers = setup_training_handlers(training_objects=training_objects)

    train_epoch_length = (
        training_objects.num_training_data // training_objects.training_batch_size
    )

    trainer = engines.SupervisedTrainer(
        training_objects.device,
        max_epochs=epochs,
        train_data_loader=training_objects.training_dataloader,
        network=training_objects.model,
        optimizer=training_objects.optimizer,
        loss_function=training_objects.loss_function,
        postprocessing=training_objects.post_train_transform,
        key_train_metric=training_objects.key_train_metrics,
        additional_metrics=training_objects.additional_train_metrics,
        inferer=SimpleInferer(),
        train_handlers=train_handlers,
        decollate=True,
        epoch_length=train_epoch_length,
        amp=False,
    )

    # set initialized trainer for "early stop" handlers
    val_handlers[0].set_trainer(trainer)
    train_handlers[0].set_trainer(trainer)
    train_handlers[2].set_validator(evaluator)

    return trainer, evaluator
