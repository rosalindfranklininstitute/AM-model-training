import typing

from torch.optim import lr_scheduler as torch_lr_scheduler, Optimizer
from monai.optimizers import lr_scheduler as monai_lr_scheduler


def onecyclelr(
    optimizer: Optimizer,
    max_lr: float,
    epochs: int,
    steps_per_epoch: int,
    pct_start: float = 0.3,
    **kwargs: typing.Any,
) -> torch_lr_scheduler.OneCycleLR:
    return torch_lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=pct_start,
    )


def cycliclr(
    optimizer: Optimizer,
    base_lr: float,
    max_lr: float,
    epochs: int,
    step_size_up: int,
    step_size_down: int | None = None,
    **kwargs: typing.Any,
) -> torch_lr_scheduler.CyclicLR:
    return torch_lr_scheduler.CyclicLR(
        optimizer,
        base_lr=base_lr,
        max_lr=max_lr,
        epochs=epochs,
        step_size_up=step_size_up,
        step_size_down=step_size_down,
    )


def warmupcosineschedule(
    optimizer: Optimizer,
    warmup_steps: int,
    epochs: int,
    steps_per_epoch: int,
    cycles: float = 0.5,
    **kwargs: typing.Any,
) -> monai_lr_scheduler.WarmupCosineSchedule:
    t_total = epochs * steps_per_epoch
    return monai_lr_scheduler.WarmupCosineSchedule(
        optimizer, warmup_steps=warmup_steps, t_total=t_total, cycles=cycles
    )


lr_scheduler_creation_functions: dict[str, torch_lr_scheduler.LRScheduler] = {
    "onecyclelr": onecyclelr,
    "cycliclr": cycliclr,
    "warmupcosineschedule": warmupcosineschedule,
}
