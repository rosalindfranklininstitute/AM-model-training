from __future__ import annotations
import logging
import typing

import torch
from monai import data, transforms


if typing.TYPE_CHECKING:
    from numpy.typing import NDArray
    from collections.abc import Sequence

__all__ = [
    "get_transform_list",
]

_logger = logging.getLogger("adaptive_milling_training")


class TransformLabels(transforms.AsDiscrete):
    """A transform to change/swap/combine label values as part of the transforms, and optionally split them into separate channels via AsDiscrete"""

    def __init__(
        self, num_labels: int, *label_changes: tuple[int, int], to_onehot: bool = False
    ) -> None:
        super().__init__(to_onehot=num_labels if to_onehot else None)

        self._label_changes = label_changes

    def __call__(
        self,
        img: NDArray[typing.Any] | torch.Tensor,
        argmax: bool | None = None,
        to_onehot: int | None = None,
        threshold: float | None = None,
        rounding: str | None = None,
    ) -> NDArray[typing.Any] | torch.Tensor:
        for old_label, new_label in self._label_changes:
            img[img == old_label] = new_label

        return super().__call__(
            img,
            argmax=argmax,
            to_onehot=to_onehot,
            threshold=threshold,
            rounding=rounding,
        )


class RandScaleSquareCrop(transforms.RandScaleCrop):
    def __init__(
        self,
        roi_scale: float,
        max_roi_scale: float | None = None,
        random_center: bool = True,
        random_size: bool = False,
        lazy: bool = False,
    ) -> None:
        super().__init__(
            roi_scale=roi_scale,
            max_roi_scale=max_roi_scale,
            random_center=random_center,
            random_size=random_size,
            lazy=lazy,
        )

    def randomize(self, img_size: Sequence[int]) -> None:
        super().randomize(img_size)
        size = tuple(self._size)  # type: ignore[arg-type]
        img_size = tuple(img_size)
        square_dims = min(min(size), min(img_size))
        self._size = tuple([square_dims] * len(size))
        self._slices = data.utils.get_random_patch(img_size, self._size, self.R)


def get_transform_list(
    image_size: int,
    labels: bool = False,
    training: bool = False,
    *,
    label_count: int | None = None,
    label_changes: list[tuple[int, int]] = [],
) -> list[transforms.Transform]:
    post_load: list[transforms.Transform] = []

    post_scale: list[transforms.Transform] = []

    if labels:
        if label_count is None:
            raise ValueError("label_count must be specified if labels=True")
        post_load.append(TransformLabels(label_count, *label_changes, to_onehot=True))
    elif label_changes:
        raise ValueError("label_changes argument is not valid for image data")

    if training:
        if not labels:
            # Not applicable to segmentations or labels
            post_load.append(transforms.RandAdjustContrast(gamma=(0.5, 2)))
            post_scale.append(
                transforms.RandGaussianNoise(mean=0.5, prob=0.1),
            )
        post_scale.extend(
            [  # The SEM image is not perpendicular to the FIB so the Y-axis cannot be flipped
                transforms.RandFlip(spatial_axis=1),
                # reduce sensitivity to scale, since we don't have metadata to work with:
                transforms.RandScaleCrop(0.7, random_center=True, random_size=True),
                # RandScaleSquareCrop(
                #     0.3, 1, random_size=True, random_center=True
                # ),  # reduce sensitivity to scale, since we don't have metadata to work with
            ]
        )
    # else:
    #     post_scale.append(RandScaleSquareCrop(1, random_size=False, random_center=True))

    transforms_list: list[transforms.Transform] = [
        transforms.LoadImage(
            reader=data.image_reader.PILReader,
            image_only=True,
            ensure_channel_first=True,
        ),
        transforms.EnsureType(),
        *post_load,
        transforms.ScaleIntensity(),
        *post_scale,
        transforms.Resize(image_size, size_mode="longest"),  # Keep pixels square
        transforms.SpatialPad((image_size, image_size), mode="constant", value=0),
    ]

    return transforms_list
