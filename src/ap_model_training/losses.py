from __future__ import annotations
import warnings
import typing

import torch
from monai import losses
from monai.networks.utils import one_hot
import segmentation_models_pytorch as smp

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray
    from numpy import float32 as np_float32
    from torch import Tensor
    from torch._prims_common import DeviceLikeType
    from collections.abc import Callable, Sequence


def weights_to_tensor(weights: Sequence[float], device: DeviceLikeType) -> Tensor:
    return torch.tensor(weights, device=device)


def log_exp_softmax_activation(tensor: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(tensor, dim=1).exp()


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
        # DiceLoss doesn't trim the first weight value:
        weights = weights[1:]
    return losses.DiceLoss(
        include_background=include_background,
        to_onehot_y=True,
        other_act=log_exp_softmax_activation,
        weight=weights,
    )


def diceceloss(
    include_background: bool, weights: Tensor, **kwargs
) -> losses.DiceCELoss:
    return losses.DiceCELoss(
        include_background=include_background,
        to_onehot_y=True,
        other_act=log_exp_softmax_activation,
        weight=weights,
        smooth_nr=1e-7,
        smooth_dr=1e-7,
        lambda_dice=0.5,
        lambda_ce=0.5,
    )


def dicefocalloss(
    include_background: bool, weights: Tensor, **kwargs
) -> losses.DiceFocalLoss:
    if not include_background:
        # DiceFocalLoss doesn't trim the first weight value:
        weights = weights[1:]
    return losses.DiceFocalLoss(
        other_act=log_exp_softmax_activation,
        include_background=include_background,
        to_onehot_y=True,
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


def generalizeddicefocalloss(
    include_background: bool, weights: Tensor, **kwargs
) -> losses.GeneralizedDiceFocalLoss:
    if not include_background:
        weights = weights[1:]
    return losses.GeneralizedDiceFocalLoss(
        include_background=include_background,
        to_onehot_y=True,
        other_act=log_exp_softmax_activation,
        lambda_focal=0.5,
        lambda_gdl=0.5,
        weight=weights,
    )


def generalizeddiceloss(
    include_background: bool, **kwargs
) -> losses.GeneralizedDiceLoss:
    return losses.GeneralizedDiceLoss(
        include_background=include_background,
        to_onehot_y=True,
        other_act=log_exp_softmax_activation,
    )


def crossentropyloss(
    include_background: bool, weights: Tensor, **kwargs
) -> torch.nn.CrossEntropyLoss:
    if not include_background:
        # CrossEntropyLoss doesn't trim the first weight value:
        weights = weights[1:]
    return torch.nn.CrossEntropyLoss(
        weight=weights, ignore_index=-100 if include_background else 0
    )


class CustomDiceCELoss(torch.nn.Module):
    def __init__(self, weight: Tensor, alpha: float = 0.5) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.weight: Tensor
        self.alpha = alpha
        # Define individual loss components
        self._ce_loss = torch.nn.CrossEntropyLoss(weight=self.weight)
        self._dice_loss = smp.losses.DiceLoss(
            mode="multiclass", from_logits=True
        )  # Using multiclass mode without 'reduction'

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) > 3:
            # Squeeze channels
            with torch.no_grad():
                target = target.view(target.shape[0], *target.shape[2:])

        weighted_ce_loss = self._ce_loss(input, target)
        dice_loss_per_class = self._dice_loss(input, target)
        weighted_dice_loss = (dice_loss_per_class * self.weight).mean()
        return self.alpha * weighted_ce_loss + (1 - self.alpha) * weighted_dice_loss


def compound_loss(weights: Tensor, alpha: float = 0.5, **kwargs) -> CustomDiceCELoss:
    return CustomDiceCELoss(weight=weights, alpha=alpha)


loss_creation_functions: dict[str, Callable[..., torch.nn.Module]] = {
    "diceloss": diceloss,
    "diceceloss": diceceloss,
    "dicefocalloss": dicefocalloss,
    "softdicecldiceloss": softdicecldiceloss,
    "naclloss": naclloss,
    "generalized_wasserstein_dice_loss": generalized_wasserstein_dice_loss,
    "generalizeddicefocalloss": generalizeddicefocalloss,
    "generalizeddiceloss": generalizeddiceloss,
    "crossentropyloss": crossentropyloss,
    "compound_loss": compound_loss,
}
