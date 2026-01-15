from __future__ import annotations
import gc
import typing

import numpy as np
import torch

if typing.TYPE_CHECKING:
    from ap_model_training.setup import TrainingParameters

__all__ = ["get_model_artifact_path", "clear_memory", "get_metrics_to_log"]


def get_model_artifact_path(epoch: int) -> str:
    return f"epoch_{epoch}_model"


def clear_memory() -> None:
    """Helps avoid the memory usage gradually growing (especially GPU) but can slow down training"""
    with torch.no_grad():
        gc.collect()
        torch.cuda.empty_cache()


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
