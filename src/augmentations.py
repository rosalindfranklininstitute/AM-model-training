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


class TransformLabelsd(TransformLabels, transforms.MapTransform):
    def __init__(
        self,
        keys: transforms.KeysCollection,
        num_labels: int,
        *label_changes: tuple[int, int],
        to_onehot: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        transforms.MapTransform.__init__(
            self, keys, allow_missing_keys=allow_missing_keys
        )
        TransformLabels.__init__(self, num_labels, *label_changes, to_onehot=to_onehot)

    def __call__(
        self,
        d,
        argmax: bool | None = None,
        to_onehot: int | None = None,
        threshold: float | None = None,
        rounding: str | None = None,
    ) -> dict:
        ()
        for key in self.key_iterator(d):
            d[key] = TransformLabels.__call__(
                self,
                d[key],
                argmax=argmax,
                to_onehot=to_onehot,
                threshold=threshold,
                rounding=rounding,
            )
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
    post_load: list[transforms.Transform] = [
        TransformLabelsd(
            [MONAI_KEYS.LABEL], label_count, *label_changes, to_onehot=True
        )
    ]

    post_scale: list[transforms.Transform] = []

    if training:
        # Not applicable to segmentations or labels
        post_load.append(
            transforms.RandAdjustContrastd([MONAI_KEYS.IMAGE], gamma=(0.5, 2))
        )
        post_scale.extend(
            [
                transforms.RandGaussianNoised([MONAI_KEYS.IMAGE], mean=0.5, prob=0.1),
                transforms.RandGridDistortiond(
                    [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL]
                ),  # Small distortions might be good
                # The SEM image is not perpendicular to the FIB so the Y-axis cannot be flipped
                transforms.RandFlipd(
                    [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL], spatial_axis=1
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
        ),
        transforms.EnsureTyped([MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL]),
        *post_load,
        transforms.ScaleIntensityd([MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL]),
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
