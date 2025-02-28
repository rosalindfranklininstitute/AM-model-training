from __future__ import annotations
import typing

from monai.networks import nets

if typing.TYPE_CHECKING:
    from collections.abc import Callable
    from torch.nn import Module


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
    )


model_creation_functions: dict[str, Callable[..., Module]] = {
    "unet": unet,
    "dynunet": dynunet,
    "segresnet": segresnet,
    "segresnetds": segresnetds,
    "segresnetds2": segresnetds2,
    "segresnetvae": segresnetvae,
}
