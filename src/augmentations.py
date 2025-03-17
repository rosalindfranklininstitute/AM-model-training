from __future__ import annotations
import logging
import typing

import torch
from numpy import deg2rad
from monai import data, transforms
from utils import MONAI_KEYS

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray
    from collections.abc import Sequence, Callable, Mapping, Hashable

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
        background_mask: NDArray[typing.Any] | torch.Tensor | bool | None = None,
    ) -> NDArray[typing.Any] | torch.Tensor:
        if background_mask is None:
            background_mask = True
        for old_label, new_label in self._label_changes:
            img[img == old_label and background_mask] = new_label
        img[background_mask] += 1  # Add 1 to everything to separate from background
        return img


class ChangeLabelsd(ChangeLabels, transforms.MapTransform):
    def __init__(
        self,
        keys: transforms.KeysCollection,
        *label_changes: tuple[int, int],
        background_mask_key: str | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        transforms.MapTransform.__init__(
            self, keys, allow_missing_keys=allow_missing_keys
        )
        ChangeLabels.__init__(self, *label_changes)
        self.background_mask_key = background_mask_key

    def __call__(self, d) -> dict:
        for key in self.key_iterator(d):
            d[key] = ChangeLabels.__call__(
                self, d[key], d.get(self.background_mask_key)
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


class ClassesToIndicesd(transforms.ClassesToIndicesd):
    def __init__(
        self,
        keys: transforms.KeysCollection,
        indices_postfix: str = "_cls_indices",
        num_classes: int | None = None,
        image_key: str | None = None,
        image_threshold: float = 0.0,
        output_shape: Sequence[int] | None = None,
        max_samples_per_class: int | None = None,
        allow_missing_keys: bool = False,
        ignore_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__(
            keys,
            indices_postfix,
            num_classes,
            image_key,
            image_threshold,
            output_shape,
            max_samples_per_class,
            allow_missing_keys,
        )
        self.ignore_indices = tuple(ignore_indices)
        self.converter = self._drop_indices_wrapper(self.converter)

    def _drop_indices_wrapper(
        self,
        fn: Callable[
            [
                NDArray[typing.Any] | torch.Tensor,
                NDArray[typing.Any] | torch.Tensor | None,
                Sequence[int] | None,
            ],
            list[NDArray[typing.Any] | torch.Tensor],
        ],
    ) -> Callable[
        [
            NDArray[typing.Any] | torch.Tensor,
            NDArray[typing.Any] | torch.Tensor | None,
            Sequence[int] | None,
        ],
        list[NDArray[typing.Any] | torch.Tensor],
    ]:
        def wrapped_converter(
            label: NDArray[typing.Any] | torch.Tensor,
            image: NDArray[typing.Any] | torch.Tensor | None = None,
            output_shape: Sequence[int] | None = None,
        ):
            return [
                i
                for j, i in enumerate(fn(label, image, output_shape))
                if j not in self.ignore_indices
            ]

        return wrapped_converter


class LabelToMaskd(transforms.LabelToMaskd):
    def __init__(  # pytype: disable=annotation-type-mismatch
        self,
        keys: transforms.KeysCollection,
        select_labels: Sequence[int] | int,
        merge_channels: bool = False,
        allow_missing_keys: bool = False,
        mask_postfix: str = "_mask",
    ) -> None:  # pytype: disable=annotation-type-mismatch
        super().__init__(keys, select_labels, merge_channels, allow_missing_keys)
        self.mask_postfix = mask_postfix

    def __call__(
        self, data: Mapping[Hashable, NDArray[typing.Any] | torch.Tensor]
    ) -> dict[Hashable, NDArray[typing.Any] | torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[str(key) + self.mask_postfix] = self.converter(d[key])

        return d


def get_transform_list(
    image_size: int,
    label_count: int,
    training: bool = False,
    foreground_labels: Sequence[int] | int | None = None,
    *,
    label_changes: list[tuple[int, int]] = [],
    use_numpy_indexing: bool = True,
) -> list[transforms.Transform]:
    post_load: list[transforms.Transform] = []
    post_scale: list[transforms.Transform] = []

    if training:
        mask_postfix = "_mask"
        label_mask_key = f"{MONAI_KEYS.LABEL}{mask_postfix}"
        # Not applicable to segmentations or labels
        if foreground_labels is None:
            raise ValueError(
                "'foreground_labels' must be defined for training datasets"
            )

        post_load.extend(
            [
                LabelToMaskd(
                    MONAI_KEYS.LABEL,
                    select_labels=foreground_labels,
                    mask_postfix=mask_postfix,
                ),
            ]
        )
        post_scale.extend(
            [
                transforms.Resized(
                    [
                        MONAI_KEYS.IMAGE,
                        MONAI_KEYS.LABEL,
                        label_mask_key,
                    ],
                    image_size,
                    size_mode="longest",
                ),  # Keep pixels square
                transforms.RandAffined(
                    [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL, label_mask_key],
                    scale_range=(0.0, 0.5),
                    prob=0.8,
                    padding_mode="zeros",
                ),
                # transforms.FgBgToIndicesd(label_mask_key),
                transforms.RandCropByPosNegLabeld(
                    [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL],
                    image_key=MONAI_KEYS.IMAGE,
                    label_key=label_mask_key,
                    spatial_size=(image_size, image_size),
                    allow_smaller=True,
                    num_samples=3,
                ),
                transforms.RandAffined(
                    [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL],
                    rotate_range=deg2rad(2),
                    prob=0.5,
                    padding_mode="zeros",
                ),
                transforms.RandAdjustContrastd([MONAI_KEYS.IMAGE], gamma=(0.5, 2)),
                transforms.RandGaussianNoised([MONAI_KEYS.IMAGE], prob=0.5),
                transforms.RandFlipd(
                    [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL],
                    spatial_axis=int(use_numpy_indexing),
                    prob=0.5,
                ),
                transforms.DeleteItemsd(
                    [label_mask_key]
                ),  # Delete newly created items to avoid errors resizing the data loader
                # transforms.RandGridDistortiond(
                #     [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL]
                # ),  # Small distortions might be good
                # The SEM image is not perpendicular to the FIB so the Y-axis cannot be flipped
                # reduce sensitivity to scale, since we don't have metadata to work with:
                # transforms.RandScaleCropd(
                #     [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL],
                #     0.7,
                #     random_center=True,
                #     random_size=True,
                # ),
            ]
        )

    transforms_list: list[transforms.Transform] = [
        transforms.LoadImaged(
            [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL],
            reader=data.image_reader.PILReader,
            image_only=True,
            ensure_channel_first=True,
            reverse_indexing=not use_numpy_indexing,  # If True, use PIL/NumPy indexing of (Y, X) rather than (X, Y)
        ),
        *post_load,
        # transforms.EnsureTyped([MONAI_KEYS.IMAGE]),
        # Ensure labels are properly formatted
        transforms.ScaleIntensityd([MONAI_KEYS.IMAGE]),
        ChangeLabelsd(
            [MONAI_KEYS.LABEL], *label_changes
        ),  # Potentially swap or merge labels
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
