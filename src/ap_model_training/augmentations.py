from __future__ import annotations
import logging
import typing

import torch
import numpy as np
from torchvision.transforms.v2 import (
    functional as functional_transforms,
    InterpolationMode,
)
from torchvision.transforms import RandomResizedCrop
from monai import data, transforms
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import RandAffined, Rand2DElasticd
from monai.transforms.intensity.dictionary import (
    RandGaussianSmoothd,
    RandGaussianSharpend,
)
from monai.utils.type_conversion import convert_to_tensor
from monai.data.meta_obj import get_track_meta

from ap_model_training.utils import MONAI_KEYS

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray
    from collections.abc import Mapping, Hashable, Collection

    KeysCollection = typing.Union[Collection[Hashable], Hashable]

__all__ = [
    "get_transform_list",
]

_logger = logging.getLogger("adaptive_milling_training")


# class ArgMax(transforms.Transform):
#     def __init__(self, dim: int = 1):
#         self._dim = dim

#     def __call__(
#         self, img: NDArray[typing.Any] | torch.Tensor
#     ) -> NDArray[typing.Any] | torch.Tensor:
#         return torch.argmax(img, dim=self._dim, keepdim=True)


# class ChangeLabels(transforms.Transform):
#     """A transform to change/swap/combine label values as part of the transforms, and optionally split them into separate channels via AsDiscrete"""

#     def __init__(self, *label_changes: tuple[int, int]) -> None:
#         self._label_changes = label_changes

#     def __call__(
#         self,
#         img: NDArray[typing.Any] | torch.Tensor,
#         background_mask: NDArray[typing.Any] | torch.Tensor | bool | None = None,
#     ) -> NDArray[typing.Any] | torch.Tensor:
#         if background_mask is None:
#             background_mask = True
#         for old_label, new_label in self._label_changes:
#             img[img == old_label and background_mask] = new_label
#         img[background_mask] += 1  # Add 1 to everything to separate from background
#         return img


# class ChangeLabelsd(transforms.transform.MapTransform):
#     def __init__(
#         self,
#         keys: KeysCollection,
#         *label_changes: tuple[int, int],
#         background_mask_key: str | None = None,
#         allow_missing_keys: bool = False,
#     ) -> None:
#         super().__init__(
#             keys, allow_missing_keys=allow_missing_keys
#         )
#         self._transform = ChangeLabels(*label_changes)
#         self.background_mask_key = background_mask_key

#     def __call__(self, d) -> dict:
#         for key in self.key_iterator(d):
#             d[key] = self._transform(
#                 self, d[key], d.get(self.background_mask_key)
#             )
#         return d


# class RandScaleSquareCrop(RandScaleCrop):
#     def __init__(
#         self,
#         roi_scale: float,
#         max_roi_scale: float | None = None,
#         random_center: bool = True,
#         random_size: bool = False,
#         lazy: bool = False,
#     ) -> None:
#         super().__init__(
#             roi_scale=roi_scale,
#             max_roi_scale=max_roi_scale,
#             random_center=random_center,
#             random_size=random_size,
#             lazy=lazy,
#         )

#     def randomize(self, img_size: Sequence[int]) -> None:
#         super().randomize(img_size)
#         size = tuple(self._size)  # type: ignore[arg-type]
#         img_size = tuple(img_size)
#         square_dims = min(min(size), min(img_size))
#         self._size = tuple([square_dims] * len(size))
#         self._slices = data.utils.get_random_patch(img_size, self._size, self.R)


# class ClassesToIndicesd(MonaiClassesToIndicesd):
#     def __init__(
#         self,
#         keys: KeysCollection,
#         indices_postfix: str = "_cls_indices",
#         num_classes: int | None = None,
#         image_key: str | None = None,
#         image_threshold: float = 0.0,
#         output_shape: Sequence[int] | None = None,
#         max_samples_per_class: int | None = None,
#         allow_missing_keys: bool = False,
#         ignore_indices: Iterable[int] | None = None,
#     ) -> None:
#         super().__init__(
#             keys,
#             indices_postfix,
#             num_classes,
#             image_key,
#             image_threshold,
#             output_shape,
#             max_samples_per_class,
#             allow_missing_keys,
#         )
#         self.ignore_indices = tuple(ignore_indices if ignore_indices is not None else [])
#         self.converter = self._drop_indices_wrapper(self.converter)

#     def _drop_indices_wrapper(
#         self,
#         fn: Callable[
#             [
#                 NDArray[typing.Any] | torch.Tensor,
#                 NDArray[typing.Any] | torch.Tensor | None,
#                 Sequence[int] | None,
#             ],
#             list[NDArray[typing.Any] | torch.Tensor],
#         ],
#     ) -> Callable[
#         [
#             NDArray[typing.Any] | torch.Tensor,
#             NDArray[typing.Any] | torch.Tensor | None,
#             Sequence[int] | None,
#         ],
#         list[NDArray[typing.Any] | torch.Tensor],
#     ]:
#         def wrapped_converter(
#             label: NDArray[typing.Any] | torch.Tensor,
#             image: NDArray[typing.Any] | torch.Tensor | None = None,
#             output_shape: Sequence[int] | None = None,
#         ):
#             return [
#                 i
#                 for j, i in enumerate(fn(label, image, output_shape))
#                 if j not in self.ignore_indices
#             ]

#         return wrapped_converter


# class LabelToMaskd(transforms.LabelToMaskd):
#     def __init__(  # pytype: disable=annotation-type-mismatch
#         self,
#         keys: KeysCollection,
#         select_labels: Sequence[int] | int,
#         merge_channels: bool = False,
#         allow_missing_keys: bool = False,
#         mask_postfix: str = "_mask",
#     ) -> None:  # pytype: disable=annotation-type-mismatch
#         super().__init__(keys, select_labels, merge_channels, allow_missing_keys)
#         self.mask_postfix = mask_postfix

#     def __call__(
#         self, data: Mapping[Hashable, NDArray[typing.Any] | torch.Tensor]
#     ) -> dict[Hashable, NDArray[typing.Any] | torch.Tensor]:
#         d = dict(data)
#         for key in self.key_iterator(d):
#             d[str(key) + self.mask_postfix] = self.converter(d[key])

#         return d


# def get_transform_list(
#     image_size: int,
#     training: bool = False,
#     foreground_labels: Sequence[int] | int | None = None,
#     *,
#     label_changes: list[tuple[int, int]] = [],
#     use_numpy_indexing: bool = True,
# ) -> list[transforms.Transform]:
#     post_load: list[transforms.Transform] = []
#     post_scale: list[transforms.Transform] = []

#     if training:
#         mask_postfix = "_mask"
#         label_mask_key = f"{MONAI_KEYS.LABEL}{mask_postfix}"
#         # Not applicable to segmentations or labels
#         if foreground_labels is None:
#             raise ValueError(
#                 "'foreground_labels' must be defined for training datasets"
#             )

#         post_load.extend(
#             [
#                 transforms.RandAdjustContrastd(
#                     MONAI_KEYS.IMAGE, gamma=(0.5, 2), prob=0.5, retain_stats=True
#                 ),
#                 transforms.RandHistogramShiftd(MONAI_KEYS.IMAGE),
#                 LabelToMaskd(
#                     MONAI_KEYS.LABEL,
#                     select_labels=foreground_labels,
#                     mask_postfix=mask_postfix,
#                 ),
#             ]
#         )
#         post_scale.extend(
#             [
#                 transforms.Resized(
#                     [
#                         MONAI_KEYS.IMAGE,
#                         MONAI_KEYS.LABEL,
#                         label_mask_key,
#                     ],
#                     image_size,
#                     size_mode="longest",
#                 ),  # Keep pixels square
#                 transforms.RandAffined(
#                     [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL, label_mask_key],
#                     scale_range=(0.0, 0.5),
#                     prob=0.5,
#                     padding_mode="zeros",
#                 ),
#                 # transforms.FgBgToIndicesd(label_mask_key),
#                 transforms.RandCropByPosNegLabeld(
#                     [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL],
#                     image_key=MONAI_KEYS.IMAGE,
#                     label_key=label_mask_key,
#                     spatial_size=(image_size, image_size),
#                     allow_smaller=True,
#                     num_samples=3,
#                 ),
#                 transforms.RandAffined(
#                     [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL],
#                     rotate_range=math.radians(2),
#                     prob=0.5,
#                     padding_mode="zeros",
#                 ),
#                 transforms.RandGaussianNoised([MONAI_KEYS.IMAGE], prob=0.25),
#                 transforms.RandFlipd(
#                     [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL],
#                     spatial_axis=int(use_numpy_indexing),
#                     prob=0.5,
#                 ),
#                 transforms.DeleteItemsd(
#                     [label_mask_key]
#                 ),  # Delete newly created items to avoid errors resizing the data loader
#                 # transforms.RandGridDistortiond(
#                 #     [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL]
#                 # ),  # Small distortions might be good
#                 # The SEM image is not perpendicular to the FIB so the Y-axis cannot be flipped
#                 # reduce sensitivity to scale, since we don't have metadata to work with:
#                 # transforms.RandScaleCropd(
#                 #     [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL],
#                 #     0.7,
#                 #     random_center=True,
#                 #     random_size=True,
#                 # ),
#             ]
#         )

#     transforms_list: list[transforms.Transform] = [
#         transforms.LoadImaged(
#             [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL],
#             reader=data.image_reader.PILReader,
#             image_only=True,
#             ensure_channel_first=True,
#             reverse_indexing=not use_numpy_indexing,  # If True, use PIL/NumPy indexing of (Y, X) rather than (X, Y)
#         ),
#         *post_load,
#         # transforms.EnsureTyped([MONAI_KEYS.IMAGE]),
#         # Ensure labels are properly formatted
#         transforms.NormalizeIntensityd([MONAI_KEYS.IMAGE]),
#         ChangeLabelsd(
#             [MONAI_KEYS.LABEL], *label_changes
#         ),  # Potentially swap or merge labels
#         *post_scale,
#         transforms.Resized(
#             [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL],
#             image_size,
#             size_mode="longest",
#         ),  # Keep pixels square
#         transforms.SpatialPadd(
#             [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL],
#             (image_size, image_size),
#             mode="constant",
#             value=0,
#         ),
#     ]

#     return transforms_list


def get_transform_list(
    image_size: int,
    augmentations: bool,
    rgb: bool = True,
    pad: bool = False,
) -> list[transforms.transform.MapTransform]:
    # preprocessing: normalise -> pad -> resize -> to_rgb
    load = LoadImaged(
        [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL],
        reader=data.image_reader.PILReader,
        image_only=True,
        ensure_channel_first=True,
        reverse_indexing=False,
        dtype=np.long,
    )
    preprocessing: list[transforms.transform.MapTransform] = [
        NormaliseTransformd([MONAI_KEYS.IMAGE], clamp=(-1, 1)),
    ]
    if pad:
        preprocessing.append(PadTransformd([MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL]))
    preprocessing.append(
        ResizeTransformd(
            [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL], image_size=image_size, pad=pad
        )
    )
    if rgb:
        preprocessing.append(ToRGBTransformd([MONAI_KEYS.IMAGE]))

    if not augmentations:
        return [load, *preprocessing]

    return [
        load,
        RandResizedCropd(
            [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL],
            size=(1024, 1536),
            scale=(0.7, 1.0),
            ratio=(1.5, 1.5),
            prob=0.3,
        ),
        RandGaussianSmoothd(
            [MONAI_KEYS.IMAGE], sigma_x=(3, 5), sigma_y=(3, 5), prob=0.3
        ),
        *preprocessing,
        # RandAffined(
        #     [MONAI_KEYS.IMAGE],
        #     scale_range=(0.95, 1.05),
        #     translate_range=(
        #         image_size * 0.05,
        #         image_size * 0.05,
        #     ),
        #     prob=0.3,
        # ),
        # RandGaussianSmoothd(
        #     [MONAI_KEYS.IMAGE], sigma_x=(3, 5), sigma_y=(3, 5), prob=0.3
        # ),
        # RandGaussianSharpend(  # TODO: inspect how equivalent this is.
        #     [MONAI_KEYS.IMAGE], prob=0.3
        # ),
        # Rand2DElasticd(  # TODO: inspect how equivalent this is.
        #     [MONAI_KEYS.IMAGE],
        #     spacing=(1, 1),
        #     magnitude_range=(0, 45),
        #     prob=0.2,
        # ),
    ]


class NormaliseTransform(transforms.transform.Transform):
    def __init__(self, clamp: tuple[int, int] = (-1, 1)) -> None:
        self._clamp_range = clamp

    def __call__(self, data: NDArray[typing.Any] | torch.Tensor) -> torch.Tensor:
        tensor = convert_to_tensor(
            data=data, dtype=torch.float32, track_meta=get_track_meta()
        )
        mean = tensor.mean()
        std = tensor.std(correction=1)
        tensor -= mean
        tensor /= 3 * std
        tensor.clamp_(*self._clamp_range)
        return tensor


class NormaliseTransformd(transforms.transform.MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        clamp: tuple[int, int] = (-1, 1),
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self._transform = NormaliseTransform(clamp=clamp)

    def __call__(
        self, data: Mapping[typing.Any, typing.Any]
    ) -> Mapping[typing.Any, typing.Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self._transform(d[key])
        return d


class PadTransformd(transforms.transform.MapTransform):
    def __call__(
        self, data: Mapping[typing.Any, typing.Any]
    ) -> Mapping[typing.Any, typing.Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self._transform(d[key])
        return d

    def _transform(self, image: torch.Tensor) -> torch.Tensor:
        # Calculate padding
        image_shape = (image.shape[-2], image.shape[-1])
        large_axis = np.argmax(image_shape)
        small_axis = 1 - large_axis
        axes_diff = image_shape[large_axis] - image_shape[small_axis]
        pad_size, remainder = divmod(axes_diff, 2)
        # Padding is [left, top, right, bottom]
        # Additional padding due to remainder will be added to the top or right
        padding = [0, pad_size + remainder, 0, pad_size]
        if large_axis == 0:
            padding = padding[::-1]
        return functional_transforms.pad(image, padding, fill=0)


class ResizeTransformd(transforms.transform.MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        image_size: int,
        pad: bool,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self._image_size = image_size
        self._pad = pad

    def __call__(
        self, data: Mapping[typing.Any, typing.Any]
    ) -> Mapping[typing.Any, typing.Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self._transform(
                d[key], mask=key in (MONAI_KEYS.LABEL, MONAI_KEYS.PRED)
            )
        return d

    def _transform(self, image: torch.Tensor, mask: bool) -> torch.Tensor:
        if self._pad:
            target_shape = (self._image_size, self._image_size)
        else:
            target_shape = ResizeTransformd._get_resize_shape(image, self._image_size)
        return functional_transforms.resize(
            image,
            list(target_shape),
            interpolation=InterpolationMode.NEAREST_EXACT
            if mask
            else InterpolationMode.BICUBIC,
            antialias=True,
        )

    @staticmethod
    def _get_resize_shape(image, image_size: int) -> tuple[int, int]:
        image_shape_array = np.asarray(image.shape[-2:])
        axis_multiplier = np.min(image_size / image_shape_array)
        shape = np.round(image_shape_array * axis_multiplier).astype(np.uint32)
        return (int(shape[0]), int(shape[1]))


class ToRGBTransformd(transforms.transform.MapTransform):
    def __call__(
        self, data: Mapping[typing.Any, typing.Any]
    ) -> Mapping[typing.Any, typing.Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self._transform(d[key])
        return d

    def _transform(self, image: torch.Tensor) -> torch.Tensor:
        return functional_transforms.grayscale_to_rgb(image)

    @staticmethod
    def _get_resize_shape(image, image_size: int) -> tuple[int, int]:
        image_shape_array = np.asarray(image.shape[-2:])
        axis_multiplier = np.min(image_size / image_shape_array)
        shape = np.round(image_shape_array * axis_multiplier).astype(np.uint32)
        return (int(shape[0]), int(shape[1]))


class RandResizedCropd(
    transforms.transform.RandomizableTransform, transforms.transform.MapTransform
):
    def __init__(
        self,
        keys: KeysCollection,
        size: tuple[float, float],
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (0.75, 1.3333333333333333),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
        prob: float = 0.3,
        allow_missing_keys: bool = False,
    ) -> None:
        transforms.transform.MapTransform.__init__(
            self, keys, allow_missing_keys=allow_missing_keys
        )
        transforms.transform.MapTransform.__init__(self, keys, allow_missing_keys)
        transforms.transform.RandomizableTransform.__init__(self, prob)
        self._transform = RandomResizedCrop(
            size=size,
            scale=scale,
            ratio=ratio,
            interpolation=interpolation,
            antialias=antialias,
        )

    def __call__(
        self, data: Mapping[typing.Any, typing.Any]
    ) -> Mapping[typing.Any, typing.Any]:
        self.randomize(None)
        d = dict(data)
        if self._do_transform:
            for key in self.key_iterator(d):
                d[key] = self._transform.forward(d[key])
        return d
