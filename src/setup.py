from __future__ import annotations
import logging
import typing
from pathlib import Path

from sklearn.model_selection import train_test_split
from monai import data, transforms


if typing.TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "dataframe_to_datasets",
    "get_transform",
    "get_transform_list",
]


def dataframe_to_datasets(
    dataframe: pd.Dataframe,
    validation_split: float = 0.2,
) -> tuple[data.ArrayDataset, data.ArrayDataset]:
    x_train, x_test, y_train, y_test = train_test_split(
        dataframe["sem"], dataframe["labels"], test_size=validation_split
    )

    return (
        data.ArrayDataset(
            img=x_train,
            img_transform=get_transform(image=True, training=True),
            seg=y_train,
            seg_transform=get_transform(image=False, training=True),
        ),
        data.ArrayDataset(
            img=x_test,
            img_transform=get_transform(image=True, training=False),
            seg=y_test,
            seg_transform=get_transform(image=False, training=False),
        ),
    )


def get_transform_list(
    image: bool = True, training: bool = False
) -> list[transforms.Transform]:
    post_load: list[transforms.Transform] = []

    post_scale: list[transforms.Transform] = []

    if training:
        if image:
            # Not applicable to segmentations or labels
            post_load.append(transforms.RandAdjustContrast(gamma=(0.5, 2)))
            post_scale.append(
                transforms.RandGaussianNoise(),
            )
        post_scale.extend(
            [  # The SEM image is not perpendicular to the FIB so the Y-axis cannot be flipped
                transforms.RandFlip(spatial_axis=1),
                transforms.RandScaleCrop(
                    0.3, 1, random_size=True, random_center=True
                ),  # reduce sensitivity to scale, since we don't have metadata to work with]
            ]
        )

    transforms_list: list[transforms.Transform] = [
        transforms.LoadImage(image_only=True, ensure_channel_first=True),
        *post_load,
        transforms.ScaleIntensity(),
        *post_scale,
        transforms.Resize(512, size_mode="longest"),  # Keep pixels square
    ]

    return transforms_list


def get_transform(image: bool = True, training: bool = False) -> transforms.Compose:
    return transforms.Compose(
        get_transform_list(
            image=image,
            training=training,
        )
    )
