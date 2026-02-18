from __future__ import annotations
import gc
import typing

import numpy as np
import torch

if typing.TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "get_model_artifact_path",
    "clear_memory",
    "get_mean_of_key_metrics",
    "get_metrics_to_log",
]


def get_model_artifact_path(epoch: int) -> str:
    return f"epoch_{epoch}_model"


def clear_memory() -> None:
    """Helps avoid the memory usage gradually growing (especially GPU) but can slow down training"""
    with torch.no_grad():
        gc.collect()
        torch.cuda.empty_cache()

def get_mean_of_key_metrics(
    metrics_dict: dict[str, float], key_metrics: Iterable[str]
) -> float:
    value = np.mean(tuple(metrics_dict[k] for k in key_metrics)).item()
    metrics_dict["mean_of_key_metrics"] = value
    return value


def get_metrics_to_log(
    stage: typing.Literal["train", "val", "eval"],
    metrics: dict[str, float],
    label_names: Iterable[str],
) -> dict[str, float]:
    metrics_to_log: dict[str, float] = {}
    for metric_name, values in metrics.items():
        if metric_name == "epoch":
            # Don't log epoch to MLFlow or you'll get a graph of y=x.
            continue
        metric_key = f"{stage}_{metric_name}"  # Prepend stage to log train and val metrics separately
        if isinstance(values, np.ndarray):
            # If it's a numpy array, assume it's per-class and log those separately (and the mean)
            for label_name, v in zip(label_names, values):
                metrics_to_log[f"{metric_key}_{label_name}"] = v
            metrics_to_log[f"{metric_key}_mean"] = values.mean()
        else:
            metrics_to_log[metric_key] = values
    return metrics_to_log
