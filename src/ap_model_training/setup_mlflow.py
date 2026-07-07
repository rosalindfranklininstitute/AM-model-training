from __future__ import annotations

import monai
from monai.handlers import MLFlowHandler


def setup_mlflow(
    training_engine, evaluation_engine, tracking_uri: str | None = None
) -> None:
    train_mlflow_handler = MLFlowHandler(
        tracking_uri=tracking_uri,
        experiment_name="ConfigWorkflowExperiment",
        run_name="Training",
        tag_name="train_loss",
        iteration_log=True,
        epoch_log=True,
        output_transform=monai.handlers.from_engine(["loss"], first=True),
        close_on_complete=True,
    )
    val_mlflow_handler = MLFlowHandler(
        tracking_uri=tracking_uri,
        experiment_name="ConfigWorkflowExperiment",
        run_name="ConfigWorkflowrun1",
        iteration_log=False,
    )

    train_mlflow_handler.attach(training_engine)
    val_mlflow_handler.attach(evaluation_engine)
