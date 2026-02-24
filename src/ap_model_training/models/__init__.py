from __future__ import annotations
import typing

import torch

import segmentation_models_pytorch as smp
from monai.networks import nets

if typing.TYPE_CHECKING:
    from os import PathLike
    from collections.abc import Callable


def retrain_wrapper(
    model_creation_function: Callable[..., torch.nn.Module],
) -> Callable[..., torch.nn.Module]:
    def wrap(
        *, weights_file: str | PathLike[str] | None = None, **kwargs: typing.Any
    ) -> torch.nn.Module:
        if weights_file is not None:
            kwargs["pretrained"] = False

        model = model_creation_function(**kwargs)

        if weights_file is not None:
            state = torch.load(
                weights_file, map_location=torch.device("cpu"), weights_only=True
            )
            model.load_state_dict(state)

        return model

    return wrap


def efficientnet_b4_flexibleunet(
    *,
    num_channels: int = 3,
    label_count: int = 5,
    pretrained: bool = False,
    **kwargs: typing.Any,
) -> nets.FlexibleUNet:
    return nets.FlexibleUNet(
        in_channels=num_channels,
        out_channels=label_count,
        backbone="efficientnet-b4",
        pretrained=pretrained,
        spatial_dims=2,
    )


def smp_efficientnet_b4_unet(
    *,
    num_channels: int = 3,
    label_count: int = 5,
    encoder_weights: typing.Literal["imagenet", "advprop"] | None = None,
    **kwargs: typing.Any,
) -> smp.Unet:
    return smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=encoder_weights,
        classes=label_count,
        activation=None,
        decoder_attention_type="scse",
        in_channels=num_channels,
    )


def smp_efficientnet_b4_unetplusplus(
    *,
    num_channels: int = 3,
    label_count: int = 5,
    encoder_weights: typing.Literal["imagenet", "advprop"] | None = None,
    **kwargs: typing.Any,
) -> smp.UnetPlusPlus:
    return smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=encoder_weights,
        classes=label_count,
        activation=None,
        decoder_attention_type="scse",
        in_channels=num_channels,
    )


def unet(
    *,
    num_channels: int = 3,
    label_count: int = 5,
    **kwargs: typing.Any,
) -> nets.UNet:
    return nets.UNet(
        spatial_dims=2,
        in_channels=num_channels,
        out_channels=label_count,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        act="PReLU",
    )


def dynunet(
    *,
    num_channels: int = 3,
    label_count: int = 5,
    **kwargs: typing.Any,
) -> nets.DynUNet:
    return nets.DynUNet(
        spatial_dims=2,
        in_channels=num_channels,
        out_channels=label_count,
        kernel_size=(3, 3, 5, 5),
        upsample_kernel_size=(3, 3, 5, 5),
        strides=(2, 2, 2, 2),
        deep_supervision=False,
    )


def segresnet(
    *,
    num_channels: int = 3,
    label_count: int = 5,
    **kwargs: typing.Any,
) -> nets.SegResNet:
    return nets.SegResNet(
        spatial_dims=2,
        in_channels=num_channels,
        out_channels=label_count,
    )


def segresnetds(
    *,
    num_channels: int = 3,
    label_count: int = 5,
    **kwargs: typing.Any,
) -> nets.SegResNetDS:
    return nets.SegResNetDS(
        spatial_dims=2,
        in_channels=num_channels,
        out_channels=label_count,
    )


def segresnetds2(
    *,
    num_channels: int = 3,
    label_count: int = 5,
    **kwargs: typing.Any,
) -> nets.SegResNetDS2:
    return nets.SegResNetDS2(
        spatial_dims=2,
        in_channels=num_channels,
        out_channels=label_count,
    )


def segresnetvae(
    *,
    num_channels: int = 3,
    label_count: int = 5,
    input_image_size=tuple[int, int],
    **kwargs: typing.Any,
) -> nets.SegResNetVAE:
    return nets.SegResNetVAE(
        input_image_size=input_image_size,  # type: ignore
        spatial_dims=2,
        in_channels=num_channels,
        out_channels=label_count,
        vae_estimate_std=True,
    )


def smp_fpn(
    *,
    num_channels: int = 3,
    label_count: int = 5,
    pretrained: bool = False,
    **kwargs: typing.Any,
) -> smp.FPN:
    return smp.FPN(
        encoder_name="efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=num_channels,
        classes=label_count,
        activation=None,
        decoder_attention_type="scse",  # type: ignore
    )


def smp_manet(
    *,
    num_channels: int = 3,
    label_count: int = 5,
    pretrained: bool = False,
    **kwargs: typing.Any,
) -> smp.MAnet:
    return smp.MAnet(
        encoder_name="efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=num_channels,
        classes=label_count,
        activation=None,
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
    "smp_fpn": smp_fpn,
    "smp_manet": smp_manet,
}

model_creation_functions = {
    k: retrain_wrapper(v) for k, v in model_creation_functions.items()
}
