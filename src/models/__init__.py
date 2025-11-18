from __future__ import annotations
import typing

import segmentation_models_pytorch as smp
from monai.networks import nets

if typing.TYPE_CHECKING:
    from collections.abc import Callable
    import torch


def efficientnet_b4_flexibleunet(
    label_count: int = 5, pretrained: bool = False, **kwargs: typing.Any
) -> nets.FlexibleUNet:
    return nets.FlexibleUNet(
        in_channels=1,
        out_channels=label_count,
        backbone="efficientnet-b4",
        pretrained=pretrained,
        spatial_dims=2,
    )


def smp_efficientnet_b4_unet(
    label_count: int = 5,
    encoder_weights: typing.Literal["imagenet", "advprop"] | None = None,
    **kwargs: typing.Any,
) -> smp.UNet:
    return smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=encoder_weights,
        classes=label_count,
        activation=None,
        decoder_attention_type="scse",
        in_channels=1,
    )


def smp_efficientnet_b4_unetplusplus(
    label_count: int = 5,
    encoder_weights: typing.Literal["imagenet", "advprop"] | None = None,
    **kwargs: typing.Any,
) -> smp.UNetPlusPlus:
    return smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=encoder_weights,
        classes=label_count,
        activation=None,
        decoder_attention_type="scse",
        in_channels=1,
    )


def unet(
    label_count: int = 5,
    **kwargs: typing.Any,
) -> nets.UNet:
    return nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=label_count,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        act="PReLU",
    )


def dynunet(
    label_count: int = 5,
    **kwargs: typing.Any,
) -> nets.DynUNet:
    return nets.DynUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=label_count,
        kernel_size=(3, 3, 5, 5),
        upsample_kernel_size=(3, 3, 5, 5),
        strides=(2, 2, 2, 2),
        deep_supervision=False,
    )


def segresnet(
    label_count: int = 5,
    **kwargs: typing.Any,
) -> nets.SegResNet:
    return nets.SegResNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=label_count,
    )


def segresnetds(
    label_count: int = 5,
    **kwargs: typing.Any,
) -> nets.SegResNetDS:
    return nets.SegResNetDS(
        spatial_dims=2,
        in_channels=1,
        out_channels=label_count,
    )


def segresnetds2(
    label_count: int = 5,
    **kwargs: typing.Any,
) -> nets.SegResNetDS2:
    return nets.SegResNetDS2(
        spatial_dims=2,
        in_channels=1,
        out_channels=label_count,
    )


def segresnetvae(
    label_count: int = 5,
    input_image_size=tuple[int, int],
    **kwargs: typing.Any,
) -> nets.SegResNetVAE:
    return nets.SegResNetVAE(
        input_image_size,
        spatial_dims=2,
        in_channels=1,
        out_channels=label_count,
        vae_estimate_std=True,
    )


def fpn(
    label_count: int = 5,
    **kwargs: typing.Any,
) -> smp.FPN:
    return smp.FPN(
        encoder_name="efficientnet-b6",
        encoder_weights=None,
        in_channels=1,
        classes=label_count,
        activation=None,
        decoder_attention_type="scse",
    )


model_creation_functions: dict[str, Callable[..., torch.nn.Module]] = {
    "efficientnet_b4_flexibleunet": efficientnet_b4_flexibleunet,
    "unet": unet,
    "dynunet": dynunet,
    "segresnet": segresnet,
    "segresnetds": segresnetds,
    "segresnetds2": segresnetds2,
    "segresnetvae": segresnetvae,
    "smp_efficientnet_b4_unet": smp_efficientnet_b4_unet,
    "smp_efficientnet_b4_unetplusplus": smp_efficientnet_b4_unetplusplus,
    "fpn": fpn,
}
