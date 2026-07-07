from __future__ import annotations
import logging
import typing

from monai import optimizers

if typing.TYPE_CHECKING:
    from matplotlib.axes import Axes
    from ap_model_training.setup.train import TrainingObjects

_logger = logging.getLogger(__name__)


def find_learning_rate(
    ax: Axes,
    training_objects: TrainingObjects,
    lower_learning_rate: float = 1e-7,
    upper_learning_rate: float = 1e-2,
    iterations: int = 20,
) -> None:
    lr_finder = optimizers.LearningRateFinder(
        model=training_objects.model,
        optimizer=training_objects.optimizer,
        criterion=training_objects.loss_function,
        device=training_objects.device,
    )
    lr_finder.range_test(
        training_objects.training.dataloader,
        training_objects.validation.dataloader,
        start_lr=lower_learning_rate,
        end_lr=upper_learning_rate,
        num_iter=iterations,
    )
    # for grad, loss in zip(*lr_finder.get_lrs_and_losses())
    #     print(f"Gradient, loss: {grad}, {loss}")
    sg, sg_loss = lr_finder.get_steepest_gradient()
    msg = f"Steepest gradient: {sg:2e}, loss: {sg_loss:2e}"
    print(msg)
    _ = lr_finder.plot(ax=ax)
    ax.set_title(msg)
