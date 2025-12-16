import logging
import typing

import numpy as np

import torch
from torchvision.transforms import v2, InterpolationMode

from albumentations.core.transforms_interface import DualTransform
import albumentations as A
from albumentations.pytorch import ToTensorV2

_logger = logging.getLogger(__name__)



def get_train_transform(
    image_size: int, normalise_first: bool, rgb: bool, pad: bool
) -> A.Compose:
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(1024, 1536), scale=(0.7, 1.0), ratio=(1.5, 1.5), p=0.3
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            PreprocessTransform(
                image_size=image_size, normalise_first=normalise_first, rgb=rgb, pad=pad
            ),
            ToTensorV2(),
        ]
    )


def get_val_transform(
    image_size: int, normalise_first: bool, rgb: bool, pad: bool
) -> A.Compose:
    return A.Compose(
        [
            PreprocessTransform(
                image_size=image_size, normalise_first=normalise_first, rgb=rgb, pad=pad
            ),
            ToTensorV2(),
        ]
    )



def tensor_to_numpy(tensor: torch.Tensor) -> np.typing.NDArray:
    tensor = tensor.permute(0, 2, 3, 1)
    tensor.squeeze_()
    return tensor.numpy(force=True)


class PreprocessTransform(DualTransform):
    def __init__(
        self,
        image_size: int,
        normalise_first: bool,
        rgb: bool,
        pad: bool,
        normalise_version: int = 2,
        resize_version: str = "pytorch",
    ):
        super().__init__(p=1)
        self._image_size = image_size
        self._normalise_first = normalise_first
        self._rgb = rgb
        self._pad = pad
        self._normalise_version = normalise_version
        self._resize_version = resize_version

    @property
    def targets(self) -> dict[str, typing.Callable[..., typing.Any]]:
        """Get mapping of target keys to their corresponding processing functions for ImageOnlyTransform.

        Returns:
            dict[str, Callable[..., Any]]: Dictionary mapping target keys to their processing functions.

        """
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
        }

    def apply(self, img: np.ndarray, **params):
        img_tensor = image_preprocess(
            img,
            image_size=self._image_size,
            normalise_first=self._normalise_first,
            rgb=self._rgb,
            pad=self._pad,
            normalise_version=self._normalise_version,
            resize_version=self._resize_version,
        )
        return tensor_to_numpy(img_tensor).astype(np.float32)

    def apply_to_mask(self, mask: np.ndarray, **params):
        mask_tensor = mask_preprocess(
            mask,
            image_size=self._image_size,
            pad=self._pad,
            resize_version=self._resize_version,
        )
        return tensor_to_numpy(mask_tensor).astype(np.uint8)


def _normalise_1(image: torch.Tensor) -> torch.Tensor:
    mean = image.mean()
    # correction=0 matches numpy's behaviour (without Bessel's
    # correction)
    std = image.std(correction=1)

    # Calculate in place:
    image -= mean
    image /= 3 * std

    image.clamp_(0, 1)
    return image


def _normalise_2(image: torch.Tensor) -> torch.Tensor:
    mean = image.mean()
    # correction=0 matches numpy's behaviour (without Bessel's
    # correction)
    std = image.std(correction=1)

    # Calculate in place:
    image -= mean
    image /= 3 * std

    image.clamp_(-1, 1)
    return image


def pad_tensor(image: torch.Tensor) -> torch.Tensor:
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

    return v2.functional.pad(image, padding, fill=0)


def resize_pytorch(
    image: torch.Tensor, image_size: int, pad: bool, mask: bool
) -> torch.Tensor:
    if pad:
        target_shape = (image_size, image_size)
    else:
        target_shape = _get_resize_shape(image, image_size)
    return v2.functional.resize(
        image,
        target_shape,
        interpolation=InterpolationMode.NEAREST_EXACT
        if mask
        else InterpolationMode.BICUBIC,
        antialias=True,
    )


def resize_cv2(
    image: torch.Tensor, image_size: int, pad: bool, mask: bool
) -> torch.Tensor:
    if pad:
        target_shape = (image_size, image_size)
    else:
        target_shape = _get_resize_shape(image, image_size)
    return torch.from_numpy(
        cv2.resize(
            image.numpy().squeeze(),
            target_shape,
            interpolation=cv2.INTER_NEAREST_EXACT if mask else cv2.INTER_CUBIC,
        )[np.newaxis, np.newaxis, ...]
    )


def image_preprocess(
    image: np.typing.NDArray,
    image_size: int,
    normalise_first: bool,
    rgb: bool,
    pad: bool,
    normalise_version: int = 2,
    resize_version: str = "pytorch",
    device=None,
) -> torch.Tensor:
    normalise_function = _normalise_1 if normalise_version == 1 else _normalise_2
    resize_function = resize_cv2 if resize_version == "cv2" else resize_pytorch
    with torch.no_grad():
        # the following functions expect channel and batch axes
        image_tensor = torch.from_numpy(
            image[np.newaxis, np.newaxis, ...].astype(np.float32)
        )
        if device is not None:
            image_tensor = image_tensor.to(device)

        if normalise_first:
            image_tensor = normalise_function(image_tensor)

        if pad:
            # Pad to square
            image_tensor = pad_tensor(image_tensor)

        if not normalise_first:
            image_tensor = normalise_function(image_tensor)

        # Resize to input dimensions. Unfortunately albumentations uses cv2
        # which doesn't match the behaviour of pytorch, so numpy has to be
        # used.img_gray = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        image_tensor = resize_function(image_tensor, image_size, pad=pad, mask=False)

        if rgb:
            image_tensor = v2.functional.grayscale_to_rgb(image_tensor)
        return image_tensor


def mask_preprocess(
    mask: np.typing.NDArray,
    image_size: int,
    pad: bool,
    resize_version: str = "pytorch",
    device=None,
) -> torch.Tensor:
    resize_function = resize_cv2 if resize_version == "cv2" else resize_pytorch
    with torch.no_grad():
        # the following functions expect channel and batch axes
        mask_tensor = torch.from_numpy(
            mask[np.newaxis, np.newaxis, ...].astype(np.uint8)
        )
        if device is not None:
            mask_tensor = mask_tensor.to(device)

        if pad:
            # Pad to square
            mask_tensor = pad_tensor(mask_tensor)

        # Resize to input dimensions. Unfortunately albumentations uses cv2
        # which doesn't match the behaviour of pytorch, so numpy has to be
        # used.
        return resize_function(mask_tensor, image_size, pad=pad, mask=True).to(
            torch.long
        )


def _get_resize_shape(image, image_size: int) -> tuple[int, int]:
    image_shape_array = np.asarray(image.shape[-2:])
    axis_multiplier = np.min(image_size / image_shape_array)
    shape = np.round(image_shape_array * axis_multiplier).astype(np.uint32)
    return (int(shape[0]), int(shape[1]))


if __name__ == "__main__":
    import sys
    import cv2
    import matplotlib.pyplot as plt

    LABEL_CMAP = plt.get_cmap("tab10")

    image_path = sys.argv[1]
    mask_path = sys.argv[2]

    # Preprocessing parameters
    image_size = 1536
    device = "cpu"
    normalise_version = 2
    resize_version = "pytorch"
    normalise_first = True
    rgb = False
    pad = False

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    image_tensor = image_preprocess(
        image,
        image_size=image_size,
        normalise_first=normalise_first,
        rgb=rgb,
        pad=pad,
        normalise_version=normalise_version,
        resize_version=resize_version,
        device=device,
    )

    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
    mask_tensor = mask_preprocess(
        mask,
        image_size=image_size,
        pad=pad,
        resize_version=resize_version,
        device=device,
    )

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(image, cmap="Greys_r")
    axs[0, 0].set_title(f"Image: {image.shape}")

    axs[0, 1].imshow(tensor_to_numpy(image_tensor), cmap="Greys_r")
    axs[0, 1].set_title(f"Image (preprocessed): {image_tensor.shape}")

    axs[1, 0].imshow(
        mask,
        cmap=LABEL_CMAP,
        vmin=0,
        vmax=len(LABEL_CMAP.colors),
        interpolation="none",
    )
    axs[1, 0].set_title(f"Mask: {mask.shape}")

    axs[1, 1].imshow(
        tensor_to_numpy(mask_tensor),
        cmap=LABEL_CMAP,
        vmin=0,
        vmax=len(LABEL_CMAP.colors),
        interpolation="none",
    )
    axs[1, 1].set_title(f"Mask (preprocessed): {mask_tensor.shape}")

    plt.show(block=True)
