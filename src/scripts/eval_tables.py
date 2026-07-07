from __future__ import annotations
from pathlib import Path
import json
import typing

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from ap_model_training.main import evaluate
from ap_model_training.utils import get_weights_path_of_best_epoch

if typing.TYPE_CHECKING:
    from collections.abc import Mapping
    from os import PathLike

    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure


def _create_cmap(
    input_cmap: str | Colormap = "YlGn", range: tuple[float, float] = (0.05, 0.8)
) -> ListedColormap:
    cmap = mpl.colormaps.get_cmap(input_cmap)
    total_colours = 256
    return ListedColormap(
        cmap(
            np.logspace(
                np.log(range[0]),
                np.log(range[1]),
                total_colours,
                base=np.e,
                endpoint=True,
            )
        ),
        N=total_colours,
    )


DEFAULT_CMAP = _create_cmap()
TAB10 = mpl.colormaps.get_cmap("tab10")

LABEL_INDEXES = {
    "background": 0,
    "gis": 1,
    "lamella": 2,
    "crack": 3,
    "vacuum": 4,
}


def evaluate_all(
    output_path: str | PathLike[str],
    weights_paths: Mapping[str, str | PathLike[str]],
    data_csvs: Mapping[str, str | PathLike[str]],
    batch_size: int = 1,
    cpu_only: bool = False,
    gpu_number: int = 0,
    save_json: bool = True,
) -> dict[str, dict[str, dict[str, float | list[float] | list[list[float]]]]]:
    output_path = Path(output_path)
    outputs: dict[
        str, dict[str, dict[str, float | list[float] | list[list[float]]]]
    ] = {}

    for weights_name, weights_path in weights_paths.items():
        weights_path = Path(weights_path)

        outputs[weights_name] = {}
        for csv_name, csv_path in data_csvs.items():
            csv = Path(csv_path)

            eval_output_path = output_path / f"{weights_name}_{csv_name}"
            eval_output_path.mkdir()

            evaluate(
                output_path=eval_output_path,
                csv=csv_path,
                weights_path=weights_path,
                cpu_only=cpu_only,
                gpu_number=gpu_number,  # Sets which GPU will be used (if cpu_only=False)
                batch_size=batch_size,
            )
            metrics_path = tuple(
                (eval_output_path / f"evaluation_{weights_path.stem}_{csv.stem}").glob(
                    "*_metrics.json"
                )
            )[0]
            if metrics_path.is_file():
                with metrics_path.open() as f:
                    outputs[weights_name][csv_name] = json.load(f)

    if save_json:
        output_file = output_path / "all_metrics.json"

        with output_file.open("w+") as f:
            json.dump(outputs, f, indent=4)

    return outputs


def plot_coloured_tables(
    metric_name: str,
    label_names: list[
        typing.Literal["Background", "GIS", "Lamella", "Crack", "Vacuum"]
    ],
    all_metrics_dict: dict[
        str, dict[str, dict[str, float | list[float] | list[list[float]]]]
    ],
    decimal_places: int = 3,
    cmap: Colormap = DEFAULT_CMAP,
    max_figure_columns: int = 3,
) -> Figure:
    table_rows: tuple[str, ...] = tuple(all_metrics_dict.keys())
    table_columns = tuple(tuple(all_metrics_dict.values())[0].keys())

    n_table_rows = len(table_rows)
    n_table_cols = len(table_columns)
    n_plots = len(label_names)

    n_fig_cols = min(n_plots, max_figure_columns)
    n_fig_rows = int(np.ceil(n_plots / n_fig_cols))

    fig_width_per_col = 0.8
    fig_height_per_row = 0.25
    fig_width = n_table_cols * (n_fig_cols + 0.5) * fig_width_per_col + 1
    fig_height = n_table_rows * (n_fig_rows + 0.5) * fig_height_per_row + 1

    fig, axs = plt.subplots(
        n_fig_rows,
        n_fig_cols,
        sharex=True,
        sharey=True,
        figsize=(fig_width, fig_height),
        dpi=600,
    )

    values_arr = np.full(
        (n_table_rows, n_table_cols, 5), fill_value=np.nan, dtype=np.float64
    )
    for i, w in enumerate(all_metrics_dict.values()):
        for j, m in enumerate(w.values()):
            values_arr[i, j, :] = m[metric_name.lower()]

    rounded_values_arr = np.round(values_arr, decimals=decimal_places)
    formatter = f"%.{decimal_places}f"

    norm = plt.Normalize(0, 1)

    for i, ax in enumerate(axs.flat):
        if i >= n_plots:
            # Hide any plots that aren't used
            ax.axis("off")
            continue

        label_name = label_names[i]

        label_index = LABEL_INDEXES[label_name.lower()]

        ax.pcolormesh(
            np.arange(n_table_cols + 1),
            np.arange(n_table_rows + 1),
            values_arr[::-1, :, label_index],  # Ensure rows are top down
            cmap=cmap,
            norm=norm,
            shading="flat",
            edgecolors="white",
            linewidth=0.5,
            antialiased=False,
        )

        # Set limits
        ax.set_xlim(0, n_table_cols)
        ax.set_ylim(0, n_table_rows)

        ax.set_xticks(
            np.arange(n_table_cols) + 0.5,
            labels=table_columns,
            rotation=30,
            ha="center",
            fontsize=10,
        )
        ax.set_yticks(
            np.arange(n_table_rows) + 0.5,
            labels=table_rows[::-1],  # Ensure rows are top down
            fontsize=10,
        )
        ax.tick_params(length=0)

        # Add values as text
        for j in range(n_table_rows):
            for k in range(n_table_cols):
                value = rounded_values_arr[j, k, label_index]
                text = "-" if np.isnan(value) else formatter % value
                ax.text(
                    k + 0.5,
                    n_table_rows - j - 0.5,
                    text,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black",
                )

        # Add title with label colour to each table
        table = ax.table(
            cellText=[[f"{label_name} - {metric_name}"]],
            cellColours=[[TAB10(label_index)]],
            cellLoc="center",
            loc="top",
        )
        table[0, 0].set_height(fig_height_per_row)

    fig.tight_layout()

    return fig


if __name__ == "__main__":
    # Set to True if the evaluation has already run.
    plot_only = False
    # List metrics to be plotted here. Don't worry about matching the case in the metrics file.
    metrics_to_plot = [
        "IoU",
        "Dice",
        # "Accuracy",
        # "Precision",
        # "Recall",
        # "F1",
    ]

    output_path = Path()  # The path where outputs will be saved

    # A dictionary containing all paths to training directories to evaluate,
    # with human friendly names (used for the table) as the keys:
    training_directories: dict[str, str | PathLike[str]] = {
        "Label text": "example/path/to/training_directory",
    }

    # A dictionary containing all paths to the data csv files to use for
    # evaluation, with human friendly names (used for the table) as the keys:
    eval_paths: dict[str, str | PathLike[str]] = {
        "Label text": "example/path/to/data.csv",
    }

    # OPTIONAL: Uncomment the next line to include an "all" csv
    # eval_paths["all"] = combine_csvs(*eval_paths.values(), output_path=output_path / "all.csv")

    if plot_only:
        all_metrics_path = output_path / "all_metrics.json"
        with all_metrics_path.open() as f:
            output_dict = dict(json.load(f))
    else:
        output_path.mkdir(exist_ok=True)

        weights_paths = {
            k: get_weights_path_of_best_epoch(v)
            for k, v in training_directories.items()
        }

        output_dict = evaluate_all(
            output_path=output_path,
            weights_paths=weights_paths,
            data_csvs=eval_paths,
            batch_size=6,
            cpu_only=False,
            gpu_number=0,
            save_json=True,
        )

    for metric_name in metrics_to_plot:
        fig = plot_coloured_tables(
            metric_name=metric_name,
            label_names=[
                # "Background",
                "Lamella",
                "GIS",
                "Crack",
                # "Vacuum",
            ],
            all_metrics_dict=output_dict,
            decimal_places=3,
        )
        fig.savefig(output_path / f"{metric_name}_table.png")
