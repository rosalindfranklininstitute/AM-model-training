from __future__ import annotations
import typing

from torch import Tensor
from monai import losses

if typing.TYPE_CHECKING:
    from torch import DeviceObjType
    from collections.abc import Callable, Sequence


def weights_to_tensor(weights: Sequence[float], device: DeviceObjType) -> Tensor:
    return Tensor(weights).to(device)


def diceloss(weights: Tensor, **kwargs) -> losses.DiceLoss:
    return losses.DiceLoss(
        include_background=True,
        weight=weights,
    )


def diceceloss(weights: Tensor, **kwargs) -> losses.DiceCELoss:
    return losses.DiceCELoss(
        include_background=True,
        weight=weights,
    )


def softdicecldiceloss(**kwargs) -> losses.SoftDiceclDiceLoss:
    return losses.SoftDiceclDiceLoss()


def naclloss(num_classes: int, **kwargs) -> losses.NACLLoss:
    return losses.NACLLoss(num_classes, dim=2, kernel_size=5)


loss_creation_functions: dict[str, Callable[..., losses._Loss]] = {
    "diceloss": diceloss,
    "diceceloss": diceceloss,
    "softdicecldiceloss": softdicecldiceloss,
    "naclloss": naclloss,
}
