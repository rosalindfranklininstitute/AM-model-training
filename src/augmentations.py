from __future__ import annotations
import logging
import typing

import torch
from monai import data, transforms
from utils import MONAI_KEYS

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray
    from collections.abc import Sequence

__all__ = [
    "get_transform_list",
]

_logger = logging.getLogger("adaptive_milling_training")


class ArgMax(transforms.Transform):
    def __init__(self, dim: int = 1):
        self._dim = dim

    def __call__(
        self, img: NDArray[typing.Any] | torch.Tensor
    ) -> NDArray[typing.Any] | torch.Tensor:
        return torch.argmax(img, dim=self._dim, keepdim=True)


class ChangeLabels(transforms.Transform):
    """A transform to change/swap/combine label values as part of the transforms, and optionally split them into separate channels via AsDiscrete"""

    def __init__(self, *label_changes: tuple[int, int]) -> None:
        self._label_changes = label_changes

    def __call__(
        self,
        img: NDArray[typing.Any] | torch.Tensor,
    ) -> NDArray[typing.Any] | torch.Tensor:
        for old_label, new_label in self._label_changes:
            img[img == old_label] = new_label
        img += 1  # Add 1 to everything to separate from background
        return img


class ChangeLabelsd(ChangeLabels, transforms.MapTransform):
    def __init__(
        self,
        keys: transforms.KeysCollection,
        *label_changes: tuple[int, int],
        allow_missing_keys: bool = False,
    ) -> None:
        transforms.MapTransform.__init__(
            self, keys, allow_missing_keys=allow_missing_keys
        )
        ChangeLabels.__init__(self, *label_changes)

    def __call__(self, d) -> dict:
        ()
        for key in self.key_iterator(d):
            d[key] = ChangeLabels.__call__(self, d[key])
        return d


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
    label_count: int,
    training: bool = False,
    *,
    label_changes: list[tuple[int, int]] = [],
) -> list[transforms.Transform]:
    post_load: list[transforms.Transform] = []
    post_scale: list[transforms.Transform] = []

    if training:
        # Not applicable to segmentations or labels
        post_load.append(
            transforms.RandAdjustContrastd([MONAI_KEYS.IMAGE], gamma=(0.5, 2))
        )
        post_scale.extend(
            [
                transforms.RandGaussianNoised([MONAI_KEYS.IMAGE], prob=0.5),
                transforms.RandGridDistortiond(
                    [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL]
                ),  # Small distortions might be good
                # The SEM image is not perpendicular to the FIB so the Y-axis cannot be flipped
                transforms.RandFlipd(
                    [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL], spatial_axis=1, prob=0.5
                ),
                # reduce sensitivity to scale, since we don't have metadata to work with:
                transforms.RandScaleCropd(
                    [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL],
                    0.7,
                    random_center=True,
                    random_size=True,
                ),
            ]
        )

    transforms_list: list[transforms.Transform] = [
        transforms.LoadImaged(
            [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL],
            reader=data.image_reader.PILReader,
            image_only=True,
            ensure_channel_first=True,
            reverse_indexing=False,  # Use PIL/NumPy indexing of (Y, X) rather than (X, Y)
        ),
        *post_load,
        # transforms.EnsureTyped([MONAI_KEYS.IMAGE]),
        # Ensure labels are properly formatted
        ChangeLabelsd(
            [MONAI_KEYS.LABEL], *label_changes
        ),  # Potentially swap or merge labels
        transforms.ScaleIntensityd([MONAI_KEYS.IMAGE]),
        *post_scale,
        transforms.Resized(
            [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL],
            image_size,
            size_mode="longest",
        ),  # Keep pixels square
        transforms.SpatialPadd(
            [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL],
            (image_size, image_size),
            mode="constant",
            value=0,
        ),
    ]

    return transforms_list

# def get_post_processing_transform_list()
