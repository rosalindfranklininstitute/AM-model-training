from __future__ import annotations
import warnings
import typing

import torch
from monai import losses
from monai.networks.utils import one_hot

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray
    from numpy import float32 as np_float32
    from torch import Tensor
    from torch._prims_common import DeviceLikeType
    from collections.abc import Callable, Sequence


def weights_to_tensor(weights: Sequence[float], device: DeviceLikeType) -> Tensor:
    return torch.Tensor(weights).to(device)


def loss_wrapper(
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    include_background: bool = True,
    to_onehot_y: bool = False,
    sigmoid: bool = False,
    softmax: bool = False,
    other_act: Callable | None = None,
) -> Callable[[Tensor, Tensor], Tensor]:
    def wrapped_loss(input: Tensor, target: Tensor, *args, **kwargs):
        # Adapted from the beginning of DiceLoss.forward to apply it to other loss functions that are missing it
        if sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if other_act is not None:
            input = other_act(input)

        if to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not include_background:
            if n_pred_ch == 1:
                warnings.warn(
                    "single channel prediction, `include_background=False` ignored."
                )
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        return loss_fn(input, target, *args, **kwargs)

    return wrapped_loss


def diceloss(include_background: bool, weights: Tensor, **kwargs) -> losses.DiceLoss:
    if not include_background:
        weights = weights[1:]
    return losses.DiceLoss(
        include_background=include_background,
        to_onehot_y=True,
        softmax=True,
        weight=weights,
    )


def diceceloss(
    include_background: bool, weights: Tensor, **kwargs
) -> losses.DiceCELoss:
    return losses.DiceCELoss(
        include_background=include_background,
        to_onehot_y=True,
        softmax=True,
        # DiceCELoss doesn't trim the first weight value:
        weight=weights,
        lambda_dice=0.5,
        lambda_ce=0.5,
    )


def dicefocalloss(
    include_background: bool, weights: Tensor, **kwargs
) -> losses.DiceFocalLoss:
    if not include_background:
        weights = weights[1:]
    return losses.DiceFocalLoss(
        include_background=include_background,
        to_onehot_y=True,
        softmax=True,
        # DiceCELoss doesn't trim the first weight value:
        weight=weights,
        lambda_dice=0.5,
        lambda_focal=0.5,
    )


def softdicecldiceloss(**kwargs) -> losses.SoftDiceclDiceLoss:
    return losses.SoftDiceclDiceLoss()


def naclloss(include_background: bool, num_classes: int, **kwargs) -> losses.NACLLoss:
    return losses.NACLLoss(num_classes + int(include_background), dim=2, kernel_size=5)


def generalized_wasserstein_dice_loss(
    dist_matrix: NDArray[np_float32], weighting_mode: str = "default", **kwargs
) -> losses.GeneralizedWassersteinDiceLoss:
    return losses.GeneralizedWassersteinDiceLoss(
        dist_matrix=dist_matrix,
        weighting_mode=weighting_mode,
    )


def generalizeddicefocalloss(include_background: bool, weights: Tensor, **kwargs):
    if not include_background:
        weights = weights[1:]
    return losses.GeneralizedDiceFocalLoss(
        include_background=include_background,
        to_onehot_y=True,
        softmax=True,
        lambda_focal=0.5,
        lambda_gdl=0.5,
        weight=weights,
    )


def generalizeddiceloss(include_background: bool, **kwargs):
    return losses.GeneralizedDiceLoss(
        include_background=include_background,
        to_onehot_y=True,
        softmax=True,
    )


loss_creation_functions: dict[str, Callable] = {
    "diceloss": diceloss,
    "diceceloss": diceceloss,
    "dicefocalloss": dicefocalloss,
    "softdicecldiceloss": softdicecldiceloss,
    "naclloss": naclloss,
    "generalized_wasserstein_dice_loss": generalized_wasserstein_dice_loss,
    "generalizeddicefocalloss": generalizeddicefocalloss,
    "generalizeddiceloss": generalizeddiceloss,
}
