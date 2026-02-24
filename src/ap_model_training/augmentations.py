from __future__ import annotations
import typing
from random import Random

import torch
import numpy as np
from torchvision.transforms.v2 import (
    functional as functional_transforms,
    InterpolationMode,
    GaussianBlur,
)
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms.functional import resized_crop

from monai import data, transforms
from monai.transforms.io.dictionary import LoadImaged
from monai.utils.type_conversion import convert_to_tensor
from monai.data.meta_obj import get_track_meta

from ap_model_training.utils import MONAI_KEYS

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Hashable, Collection, Sequence

    from numpy.typing import NDArray

    KeysCollection = typing.Union[Collection[Hashable], Hashable]

__all__ = [
    "get_transform_list",
]


def get_transform_list(
    image_size: int,
    augmentations: bool,
    rgb: bool = True,
    pad: bool = False,
    dog: bool = False,
) -> list[transforms.transform.MapTransform]:
    # preprocessing: normalise -> pad -> resize -> to_rgb
    loading = [
        LoadImaged(
            [MONAI_KEYS.IMAGE],
            reader=data.image_reader.PILReader,
            image_only=True,
            ensure_channel_first=True,
            reverse_indexing=False,
            dtype=np.float32,
        ),
        LoadImaged(
            [MONAI_KEYS.LABEL],
            reader=data.image_reader.PILReader,
            image_only=True,
            ensure_channel_first=True,
            reverse_indexing=False,
            dtype=np.long,
        ),
    ]
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

    if dog:
        preprocessing.append(
            DoGChannelsd(
                [MONAI_KEYS.IMAGE],
                kernel_size=64,
                sigmas1=(image_size / 36, image_size / 16),
                sigmas2=(image_size / 128, image_size / 64),
            )
        )

    if not augmentations:
        return [*loading, *preprocessing]

    return [
        *loading,
        RandResizedCropd(
            [MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL],
            size=image_size,
            scale=(0.7, 1.0),
            ratio=(1.5, 1.5),
            prob=0.3,
            interpolation=InterpolationMode.BICUBIC,
            mask_interpolation=InterpolationMode.NEAREST_EXACT,
        ),
        RandGaussianBlurd([MONAI_KEYS.IMAGE], blur_limit=(3, 5), prob=0.3),
        *preprocessing,
    ]


class NormaliseTransform(transforms.transform.Transform):
    def __init__(self, clamp: tuple[int, int] = (-1, 1)) -> None:
        self._clamp_range = clamp

    @torch.no_grad()
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

    def _transform(self, data: NDArray[typing.Any] | torch.Tensor) -> torch.Tensor:
        image = convert_to_tensor(data=data, dtype=None, track_meta=get_track_meta())
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

    def _transform(
        self, data: NDArray[typing.Any] | torch.Tensor, mask: bool
    ) -> torch.Tensor:
        image = convert_to_tensor(data=data, dtype=None, track_meta=get_track_meta())
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


class RandGaussianBlurd(
    transforms.transform.RandomizableTransform, transforms.transform.MapTransform
):
    def __init__(
        self,
        keys: KeysCollection,
        blur_limit: tuple[int, int] | int = 0,
        sigma_limit: tuple[float, float] | float = (0.5, 3.0),
        prob: float = 0.5,
        per_image: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        transforms.transform.MapTransform.__init__(
            self, keys, allow_missing_keys=allow_missing_keys
        )
        transforms.transform.MapTransform.__init__(self, keys, allow_missing_keys)
        transforms.transform.RandomizableTransform.__init__(self, prob)
        self.blur_limit = typing.cast("tuple[int, int]", blur_limit)
        self.sigma_limit = typing.cast("tuple[float, float]", sigma_limit)
        self._per_image = per_image
        self.py_random = Random()

    def __call__(
        self, data: Mapping[typing.Any, typing.Any]
    ) -> Mapping[typing.Any, typing.Any]:
        d = dict(data)
        first_data = d[self.first_key(d)]
        if self._per_image:
            blur_list = [self.get_blur() for _ in range(first_data.shape[0])]
        else:
            blur = self.get_blur()
            if blur is None:
                # Skip everything if params is None
                return d
            blur_list = [blur]
        batch_size = d[self.first_key(d)].shape[0]
        for i in range(batch_size):
            if i == 0:
                blur = self.get_blur()
            # Per-image skipping and blur
            elif self._per_image and i > 0:
                self.randomize(None)
                if not self._do_transform:
                    continue
            blur = self.get_blur()

        for key in self.key_iterator(d):
            d[key] = convert_to_tensor(
                data=d[key], dtype=None, track_meta=get_track_meta()
            )
            if self._per_image:
                image_list: list[torch.Tensor] = []
                for i in range(first_data.shape[0]):
                    image = d[key][i, ...].unsqueeze(0)
                    if blur_list[i] is None:
                        image_list.append(image)
                    else:
                        image_list.append(
                            blur_list[i].transform(  # type: ignore
                                image,
                                params=blur_list[i].make_params(None),  # type: ignore
                            )
                        )
                tensor = torch.concatenate(image_list, dim=0)
            else:
                if blur_list[0] is None:
                    continue
                tensor = blur_list[0].transform(
                    d[key],
                    params=blur_list[0].make_params(None),  # type: ignore
                )
            d[key] = tensor
        return d

    def get_blur(self) -> GaussianBlur | None:
        self.randomize(None)
        if not self._do_transform:
            return None
        sigma = self.py_random.uniform(*self.sigma_limit)
        ksize = self.py_random.randint(*self.blur_limit)

        # Using the logic from Albumentations create_gaussian_kernel_1d
        # PIL's kernel creation approach
        size = int(sigma * 3.5) * 2 + 1 if ksize == 0 else ksize
        # Ensure odd size
        size = size + 1 if size % 2 == 0 else size

        return GaussianBlur(kernel_size=size, sigma=sigma)


class ToRGBTransformd(transforms.transform.MapTransform):
    def __call__(
        self, data: Mapping[typing.Any, typing.Any]
    ) -> Mapping[typing.Any, typing.Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self._transform(d[key])
        return d

    def _transform(self, data: NDArray[typing.Any] | torch.Tensor) -> torch.Tensor:
        image = convert_to_tensor(data=data, dtype=None, track_meta=get_track_meta())
        return functional_transforms.grayscale_to_rgb(image)


class DoGChannelsd(transforms.transform.MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        kernel_size: int | Sequence[int],
        sigmas1: tuple[float, float],
        sigmas2: tuple[float, float],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self._g1_0 = GaussianBlur(kernel_size=kernel_size, sigma=sigmas1[0])
        self._g1_1 = GaussianBlur(kernel_size=kernel_size, sigma=sigmas1[1])
        self._g2_0 = GaussianBlur(kernel_size=kernel_size, sigma=sigmas2[0])
        self._g2_1 = GaussianBlur(kernel_size=kernel_size, sigma=sigmas2[1])

    def __call__(
        self, data: Mapping[typing.Any, typing.Any]
    ) -> Mapping[typing.Any, typing.Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self._transform(d[key])
        return d

    @torch.no_grad()
    def _transform(self, data: NDArray[typing.Any] | torch.Tensor) -> torch.Tensor:
        image = convert_to_tensor(data=data, dtype=None, track_meta=get_track_meta())
        image[:, 0, ...] = self._g1_0.forward(image[:, 0, ...]) - self._g1_1.forward(
            image[:, 0, ...]
        )
        image[:, 2, ...] = self._g2_0.forward(image[:, 2, ...]) - self._g2_1.forward(
            image[:, 2, ...]
        )
        return image


class RandResizedCropd(
    transforms.transform.RandomizableTransform, transforms.transform.MapTransform
):
    def __init__(
        self,
        keys: KeysCollection,
        size: tuple[int, int] | int,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (0.75, 1.3333333333333333),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        mask_interpolation: InterpolationMode = InterpolationMode.NEAREST_EXACT,
        antialias: bool = True,
        prob: float = 0.3,
        per_image: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        transforms.transform.MapTransform.__init__(
            self, keys, allow_missing_keys=allow_missing_keys
        )
        transforms.transform.RandomizableTransform.__init__(self, prob)
        self._size = size
        self._scale = scale
        self._ratio = ratio
        self._interpolation = interpolation
        self._mask_interpolation = mask_interpolation
        self._antialias = antialias
        self._per_image = per_image

    def __call__(
        self, data: Mapping[typing.Any, typing.Any]
    ) -> Mapping[typing.Any, typing.Any]:
        d = dict(data)
        first_data = d[self.first_key(d)]
        if isinstance(self._size, int):
            size = RandResizedCropd._get_resize_shape(first_data, image_size=self._size)
        else:
            size = self._size
        params_list: list[tuple[int, int, int, int] | None]
        if self._per_image:
            params_list = [
                self.get_params(image=first_data[0, ...])
                for _ in range(first_data.shape[0])
            ]
        else:
            params = self.get_params(image=first_data)
            if params is None:
                # Skip everything if params is None
                return d
            params_list = [params]

        for key in self.key_iterator(d):
            data = convert_to_tensor(
                data=d[key], dtype=None, track_meta=get_track_meta()
            )
            if key in (MONAI_KEYS.LABEL, MONAI_KEYS.PRED):
                # Force NEAREST_EXACT for labels/predictions
                interpolation = self._mask_interpolation
            else:
                interpolation = self._interpolation
            if self._per_image:
                tensor = self._transform_per_image(
                    d[key],
                    size=size,
                    params_list=params_list,
                    interpolation=interpolation,
                )
            else:
                tensor = self._transform(
                    d[key],
                    size=size,
                    params=params_list[0],
                    interpolation=interpolation,
                )
            d[key] = tensor
        return d

    def _transform(
        self,
        data: torch.Tensor,
        size: tuple[int, int],
        params: tuple[int, int, int, int] | None,
        interpolation: InterpolationMode,
    ) -> torch.Tensor:
        if params is None:
            return data
        return resized_crop(
            data,
            *params,  # type: ignore
            size=size,  # type: ignore
            interpolation=interpolation,
            antialias=self._antialias,
        )

    def _transform_per_image(
        self,
        data: torch.Tensor,
        size: tuple[int, int],
        params_list: list[tuple[int, int, int, int] | None],
        interpolation: InterpolationMode,
    ) -> torch.Tensor:
        batch_size = data.shape[0]
        image_list: list[torch.Tensor] = []
        for i in range(batch_size):
            image = data[i, ...].unsqueeze(0)
            if params_list[i] is None:
                image_list.append(image)
            else:
                image_list.append(
                    resized_crop(
                        image,
                        *params_list[i],  # type: ignore
                        size=size,  # type: ignore
                        interpolation=interpolation,
                        antialias=self._antialias,
                    )
                )
        return torch.concatenate(image_list, dim=0)

    def get_params(self, image: torch.Tensor) -> tuple[int, int, int, int] | None:
        # Check if it should be done
        self.randomize(None)
        if not self._do_transform:
            return None
        # Get resize crop parameters
        return RandomResizedCrop.get_params(
            image,
            scale=self._scale,  # type: ignore
            ratio=self._ratio,  # type: ignore
        )

    @staticmethod
    def _get_resize_shape(image, image_size: int) -> tuple[int, int]:
        image_shape_array = np.asarray(image.shape[-2:])
        axis_multiplier = np.min(image_size / image_shape_array)
        shape = np.round(image_shape_array * axis_multiplier).astype(np.uint32)
        return (int(shape[0]), int(shape[1]))
