from __future__ import annotations
import logging
import typing
from pathlib import Path

import torch
import numpy as np
import cv2
from monai import data
from monai.utils import ensure_tuple

if typing.TYPE_CHECKING:
    from collections.abc import Hashable, Collection, Sequence
    from os import PathLike

    KeysCollection = typing.Union[Collection[Hashable], Hashable]

__all__ = ["OpenCVReader"]

_logger = logging.getLogger(__name__)

# Forces OpenCV to run in single-threaded mode within each worker process (avoids competing with worker threads)
cv2.setNumThreads(0)


class OpenCVReader(data.image_reader.NumpyReader):
    def __init__(self, rescale_input: bool = False, **kwargs) -> None:
        super().__init__()
        self.rescale_input = rescale_input
        self.kwargs = kwargs

    def verify_suffix(
        self, filename: Sequence[str | PathLike[str]] | str | PathLike[str]
    ) -> bool:
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
        self, data: Sequence[str | PathLike[str]] | str | PathLike[str], **kwargs
    ) -> Sequence[typing.Any] | typing.Any:
        """
        Read image data from specified file or files.
        Note that it returns a data object or a sequence of data objects.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for actual `read` API of 3rd party libs.

        """
        img_ = []
        filenames: Sequence[str | PathLike[str]] = ensure_tuple(data)
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
        return torch.as_tensor(img), metadata
