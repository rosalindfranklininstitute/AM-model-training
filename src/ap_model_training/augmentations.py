from __future__ import annotations
import logging
import typing
from pathlib import Path
from random import Random

import torch
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms.v2 import (
    functional as functional_transforms,
    InterpolationMode,
    GaussianBlur,
)
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms.functional import resized_crop

from monai import data, transforms
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import RandAffined, Rand2DElastic
from monai.transforms.intensity.dictionary import (
    RandGaussianSmoothd,
    RandGaussianSharpend,
)
from monai.utils import ensure_tuple
from monai.utils.type_conversion import convert_to_tensor
from monai.data.meta_obj import get_track_meta

from ap_model_training.utils import MONAI_KEYS

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Hashable, Collection, Sequence

    from numpy.typing import NDArray
    from monai.config import PathLike

    KeysCollection = typing.Union[Collection[Hashable], Hashable]

__all__ = [
    "get_transform_list",
]


_logger = logging.getLogger("adaptive_milling_training")

# Forces OpenCV to run in single-threaded mode within each worker process (avoids competing with worker threads)
cv2.setNumThreads(0)


class OpenCVReader(data.image_reader.NumpyReader):
    def __init__(self, rescale_input: bool = False, **kwargs) -> None:
        super().__init__()
        self.rescale_input = rescale_input
        self.kwargs = kwargs

    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        """
        Verify whether the specified `filename` is supported by the current reader.
        This method should return True if the reader is able to read the format suggested by the
        `filename`.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.

        """
        return True

    def read(
        self, data: Sequence[PathLike] | PathLike, **kwargs
    ) -> Sequence[typing.Any] | typing.Any:
        """
        Read image data from specified file or files.
        Note that it returns a data object or a sequence of data objects.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for actual `read` API of 3rd party libs.

        """
        img_ = []
        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        rescale_input: bool = kwargs_.pop("rescale_input", self.rescale_input)
        for name in filenames:
            if Path(name).is_file():
                arr = np.asarray(cv2.imread(str(name), cv2.IMREAD_UNCHANGED, **kwargs))
                if rescale_input:
                    if arr.dtype == np.uint8:
                        arr = arr.astype(np.float32) / 255.0
                    elif arr.dtype == np.uint16:
                        arr = arr.astype(np.float32) / 65535.0
                    else:
                        raise ValueError(f"Unsupported image dtype: {arr.dtype}")
                img_.append(arr)
        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img) -> tuple[torch.Tensor, dict]:  # type: ignore
        img, metadata = super().get_data(img=img)
        return torch.from_numpy(img), metadata


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
            reader=OpenCVReader,
            rescale_input=True,
            image_only=True,
            ensure_channel_first=True,
            reverse_indexing=False,
            dtype=np.float32,
        ),
        # NormalizeInputImagesd([MONAI_KEYS.IMAGE]),
        LoadImaged(
            [MONAI_KEYS.LABEL],
            reader=OpenCVReader,
            rescale_input=False,
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
            size=(1024, 1536),
            scale=(0.7, 1.0),
            ratio=(1.5, 1.5),
            prob=0.3,
            interpolation=InterpolationMode.BILINEAR,
            mask_interpolation=InterpolationMode.NEAREST_EXACT,
        ),
        # Ensure cropping doesn't introduce any NaNs (https://github.com/Project-MONAI/MONAI/discussions/2637):
        # SignalFillEmptyd([MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL]),
        RandGaussianBlurd([MONAI_KEYS.IMAGE], blur_limit=(3, 5), prob=0.3),
        # RandGaussianSmoothd(
        #     [MONAI_KEYS.IMAGE],
        #     sigma_x=(3, 5),
        #     sigma_y=(3, 5),
        #     sigma_z=(0, 0),
        #     prob=0.3,
        # ),
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
        # SignalFillEmptyd([MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL]),
    ]


class NormalizeInputImagesd(transforms.transform.MapTransform):
    def __call__(
        self, data: Mapping[typing.Any, typing.Any]
    ) -> Mapping[typing.Any, typing.Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = NormalizeInputImagesd._transform(d[key])
        return d

    @torch.no_grad()
    @staticmethod
    def _transform(data: NDArray[typing.Any] | torch.Tensor) -> torch.Tensor:
        img = convert_to_tensor(data=data, dtype=None, track_meta=get_track_meta())
        if img.dtype == torch.uint8:
            img = img.to(torch.float32) / 255.0
        elif img.dtype == torch.uint16:
            img = img.to(torch.float32) / 65535.0
        else:
            raise ValueError(f"Unsupported image dtype: {img.dtype}")
        return img


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

    @torch.no_grad()
    def __call__(
        self, data: Mapping[typing.Any, typing.Any]
    ) -> Mapping[typing.Any, typing.Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            image = convert_to_tensor(
                data=d[key], dtype=None, track_meta=get_track_meta()
            )
            d[key] = self._transform(image)
        return d


class PadTransformd(transforms.transform.MapTransform):
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

    @torch.no_grad()
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


    @torch.no_grad()
    def __call__(
        self, data: Mapping[typing.Any, typing.Any]
    ) -> Mapping[typing.Any, typing.Any]:
        d = dict(data)
        self.randomize(None)
        if self._do_transform or self._per_image:
            for i, key in enumerate(self.key_iterator(d)):
                if i == 0:
                    blur = self.get_blur()
                # Per-image skipping and blur
                elif self._per_image and i > 0:
                    self.randomize(None)
                    if not self._do_transform:
                        continue
                    blur = self.get_blur()

                image = convert_to_tensor(
                    data=d[key], dtype=None, track_meta=get_track_meta()
                )
                d[key] = blur.transform(image, params=blur.make_params(None))  # type: ignore
        return d

    def get_blur(self) -> GaussianBlur:
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

    @torch.no_grad()
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
        size: tuple[float, float],
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

    @torch.no_grad()
    def __call__(
        self, data: Mapping[typing.Any, typing.Any]
    ) -> Mapping[typing.Any, typing.Any]:
        d = dict(data)
        self.randomize(None)
        if self._do_transform or self._per_image:
            for i, key in enumerate(self.key_iterator(d)):
                image = convert_to_tensor(
                    data=d[key], dtype=None, track_meta=get_track_meta()
                )
                if i == 0:
                    params = self.get_params(image=image)
                # Per-image skipping and parameters
                elif self._per_image and i > 0:
                    self.randomize(None)
                    if not self._do_transform:
                        continue
                    params = self.get_params(image=image)

                if key in (MONAI_KEYS.LABEL, MONAI_KEYS.PRED):
                    # Force NEAREST_EXACT for labels/predictions
                    interpolation = self._mask_interpolation
                else:
                    interpolation = self._interpolation
                d[key] = resized_crop(
                    image,
                    *params,  # type: ignore
                    size=self._size,  # type: ignore
                    interpolation=interpolation,
                    antialias=self._antialias,
                )
        return d

    def get_params(self, image: torch.Tensor) -> tuple[int, int, int, int]:
        return RandomResizedCrop.get_params(
            image,
            scale=self._scale,  # type: ignore
            ratio=self._ratio,  # type: ignore
        )
