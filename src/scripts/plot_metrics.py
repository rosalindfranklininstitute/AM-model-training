from __future__ import annotations
import logging
import typing
from pathlib import Path
from string import ascii_lowercase

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

if typing.TYPE_CHECKING:
    from os import PathLike

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

CLASS_LABELS = ["Background", "GIS", "Lamella", "Crack", "Vacuum"]


def __create_cmap() -> ListedColormap:
    _tab10 = plt.get_cmap("tab10")
    return ListedColormap([_tab10(_) for _ in range(len(CLASS_LABELS))])


LABEL_CMAP = __create_cmap()


def get_indexes_upto_epoch(df: pd.DataFrame, max_epoch: int | None = None) -> list[int]:
    if max_epoch is None:
        return list(range(len(df)))
    return df.index[df["epoch"] <= max_epoch].tolist()


def plot_metrics(
    train_metrics_path: str | PathLike[str],
    val_metrics_path: str | PathLike[str],
    output_image_path: str | PathLike[str],
    marker_epoch: int | None = None,
    max_epoch: int | None = None,
    log_loss: bool = True,
    metrics_to_plot: list[
        typing.Literal["IoU", "Dice", "Accuracy", "Precision", "Recall", "F1"]
    ] = ["IoU", "Dice", "Accuracy", "Precision", "Recall", "F1"],
) -> None:
    plt.rcParams["font.family"] = "Nimbus Sans"
    plt.rcParams["font.size"] = 5
    plt.rcParams["xtick.labelsize"] = 5
    plt.rcParams["ytick.labelsize"] = 5

    train_metrics = pd.read_csv(train_metrics_path)
    val_metrics = pd.read_csv(val_metrics_path)

    total_epochs = int(max(train_metrics["epoch"]))

    if max_epoch is None or max_epoch > total_epochs:
        max_epoch = total_epochs

    train_indexes = get_indexes_upto_epoch(train_metrics, max_epoch=max_epoch)
    val_indexes = get_indexes_upto_epoch(val_metrics, max_epoch=max_epoch)

    train_epochs = train_metrics["epoch"][train_indexes].tolist()
    val_epochs = val_metrics["epoch"][val_indexes].tolist()

    train_loss = train_metrics["loss"][train_indexes].tolist()
    val_loss = val_metrics["loss"][val_indexes].tolist()

    total_rows = 1 + len(metrics_to_plot)

    fig_width = 180 / 25.4
    fig_height = total_rows * 1.5
    fig, axes = plt.subplots(total_rows, 2, figsize=(fig_width, fig_height))

    for i, ax in enumerate(axes.flat):
        ax.text(
            -0.1,
            1.1,
            ascii_lowercase[i],
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="left",
        )
        ax.grid(True, linestyle="--", color="lightgrey", alpha=0.5)
        if marker_epoch is not None:
            ax.axvline(
                x=marker_epoch, color="gray", linestyle="--", linewidth=0.8, zorder=0
            )

    loss_max = max(train_loss + val_loss)

    if log_loss:
        loss_ylims = (1e-2, loss_max * 1.5)
    else:
        loss_ylims = (-0.05, loss_max * 1.05)

    ax_train_loss = axes[0, 0]
    ax_train_loss.plot(
        train_epochs,
        train_loss,
        label="Train Loss",
        color="k",
        marker=".",
        markersize=0.5,
        linewidth=0.4,
    )
    if log_loss:
        ax_train_loss.set_yscale("log")
    ax_train_loss.set_ylim(*loss_ylims)
    ax_train_loss.set_xlabel("Epochs")
    ax_train_loss.set_ylabel("Loss")
    ax_train_loss.set_title("Training Loss")
    ax_train_loss.grid(True, linestyle="--", color="lightgrey", alpha=0.5)

    ax_val_loss = axes[0, 1]
    ax_val_loss.plot(
        val_epochs,
        val_loss,
        label="Val Loss",
        color="k",
        marker=".",
        markersize=0.5,
        linewidth=0.4,
    )
    if log_loss:
        ax_val_loss.set_yscale("log")
    ax_val_loss.set_ylim(*loss_ylims)
    ax_val_loss.set_xlabel("Epochs")
    ax_val_loss.set_ylabel("Loss")
    ax_val_loss.set_title("Validation Loss")
    ax_val_loss.grid(True, linestyle="--", color="lightgrey", alpha=0.5)

    metric_ylims = (-0.05, 1.05)
    for i, metric_name in enumerate(metrics_to_plot, start=1):
        ax_train = axes[i, 0]
        for i, class_name in enumerate(CLASS_LABELS):
            ax_train.plot(
                train_epochs,
                train_metrics[f"{class_name.lower()}_{metric_name.lower()}"][
                    train_indexes
                ],
                color=LABEL_CMAP(i),
                label=class_name,
                marker=".",
                markersize=0.5,
                linewidth=0.4,
            )
        try:
            mean_metric_name = f"mean_{metric_name.lower()}"
            if mean_metric_name in train_metrics:
                ax_train.plot(
                    train_epochs,
                    train_metrics[f"mean_{metric_name.lower()}"][train_indexes],
                    label="Mean",
                    color="k",
                    linestyle="--",
                    marker=".",
                    markersize=0.5,
                    linewidth=0.4,
                )

            weighted_average_metric_name = f"weighted_average_{metric_name.lower()}"
            if weighted_average_metric_name in train_metrics:
                ax_train.plot(
                    train_epochs,
                    train_metrics[f"weighted_average_{metric_name.lower()}"][
                        train_indexes
                    ],
                    label="Weighted Average",
                    color="k",
                    linestyle=":",
                    marker=".",
                    markersize=0.5,
                    linewidth=0.4,
                )
        except Exception:
            _logger.error(
                "An exception occurred while plotting the mean and weighted average for train %s",
                metric_name,
                exc_info=True,
            )
        ax_train.set_ylim(*metric_ylims)
        ax_train.set_xlabel("Epochs")
        ax_train.set_ylabel(metric_name)
        ax_train.set_title(f"Training {metric_name} per Class")
        ax_train.legend()

    for i, metric_name in enumerate(metrics_to_plot, start=1):
        ax_val = axes[i, 1]
        for i, class_name in enumerate(CLASS_LABELS):
            ax_val.plot(
                val_epochs,
                val_metrics[f"{class_name.lower()}_{metric_name.lower()}"][val_indexes],
                color=LABEL_CMAP(i),
                label=class_name,
                marker=".",
                markersize=0.5,
                linewidth=0.4,
            )
        try:
            mean_metric_name = f"mean_{metric_name.lower()}"
            if mean_metric_name in val_metrics:
                ax_val.plot(
                    val_epochs,
                    val_metrics[f"mean_{metric_name.lower()}"][val_indexes],
                    label="Mean",
                    color="k",
                    linestyle="--",
                    marker=".",
                    markersize=0.5,
                    linewidth=0.4,
                )

            weighted_average_metric_name = f"weighted_average_{metric_name.lower()}"
            if weighted_average_metric_name in val_metrics:
                ax_val.plot(
                    val_epochs,
                    val_metrics[f"weighted_average_{metric_name.lower()}"][val_indexes],
                    label="Weighted Average",
                    color="k",
                    linestyle=":",
                    marker=".",
                    markersize=0.5,
                    linewidth=0.4,
                )
        except Exception:
            _logger.error(
                "An exception occurred while plotting the mean and weighted average for val %s",
                metric_name,
                exc_info=True,
            )
        ax_val.set_ylim(*metric_ylims)
        ax_val.set_xlabel("Epochs")
        ax_val.set_ylabel(metric_name)
        ax_val.set_title(f"Validation {metric_name} per Class")
        ax_val.legend()

    fig.tight_layout()
    fig.savefig(output_image_path, dpi=600)
    _logger.info("Plot saved to %s", output_image_path)


if __name__ == "__main__":
    # Add relevant paths here
    training_metrics_path = Path()
    validation_metrics_path = Path()
    output_image_path = training_metrics_path.parent / "training_metrics_plot.png"
    plot_metrics(
        training_metrics_path,
        validation_metrics_path,
        output_image_path=output_image_path,
    )
