from __future__ import annotations
import logging
import typing

from monai import data, transforms
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

from augmentations import get_transform_list

if typing.TYPE_CHECKING:
    import pandas as pd
    import numpy as np
    from numpy.typing import NDArray

__all__ = [
    "split_dataframe",
    "split_array",
    "create_datasets",
]

_logger = logging.getLogger("adaptive_milling_training")


def split_dataframe(dataframe: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    return dataframe["sem"], dataframe["labels"]


def split_array(array: NDArray[np.str_]) -> tuple[NDArray[np.str_], NDArray[np.str_]]:
    return array[:, 0], array[:, 1]


def create_datasets(
    x: NDArray[np.str_] | pd.DataFrame | pd.Series,
    y: NDArray[np.str_] | pd.DataFrame | pd.Series,
    image_size: int,
    label_count: int,
    validation_split: float = 0.2,
    *,
    label_changes: list[tuple[int, int]] = [],
    **transform_kwargs: typing.Any,
) -> tuple[data.ArrayDataset, data.ArrayDataset]:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=validation_split
    )
    return (
        data.ArrayDataset(
            img=x_train,
            img_transform=transforms.Compose(
                get_transform_list(
                    image_size,
                    labels=False,
                    training=True,
                    **transform_kwargs,
                )
            ),
            seg=y_train,
            seg_transform=transforms.Compose(
                get_transform_list(
                    image_size,
                    labels=True,
                    training=True,
                    label_count=label_count,
                    label_changes=label_changes,
                    **transform_kwargs,
                )
            ),
        ),
        data.ArrayDataset(
            img=x_test,
            img_transform=transforms.Compose(
                get_transform_list(
                    image_size,
                    labels=False,
                    training=False,
                    **transform_kwargs,
                )
            ),
            seg=y_test,
            seg_transform=transforms.Compose(
                get_transform_list(
                    image_size,
                    labels=True,
                    training=False,
                    label_count=label_count,
                    label_changes=label_changes,
                    **transform_kwargs,
                )
            ),
        ),
    )
