from __future__ import annotations

import torch
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from am_model_training.metrics import get_metrics_from_confusion_matrix


def test_get_metrics_from_confusion_matrix() -> None:
    num_classes = 5
    f1_metric = MulticlassF1Score(num_classes=num_classes, average="none")
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes, average="none")
    precision_metric = MulticlassPrecision(num_classes=num_classes, average="none")
    recall_metric = MulticlassRecall(num_classes=num_classes, average="none")
    confusion_matrix_metric = MulticlassConfusionMatrix(num_classes=num_classes)

    # Test epochs and steps are both correct
    for _ in range(3):
        # Ignore the last class in both cases to see how 0s are handled
        prediction = torch.randint(
            0, num_classes - 1, size=(6, 1, 10, 20), dtype=torch.long
        )
        target = torch.randint(0, num_classes - 1, size=(6, 1, 10, 20), dtype=torch.long)

        # Step check
        confusion_matrix = confusion_matrix_metric.forward(prediction, target)

        expected_f1 = f1_metric.forward(prediction, target)
        expected_accuracy = accuracy_metric.forward(prediction, target)
        expected_precision = precision_metric.forward(prediction, target)
        expected_recall = recall_metric.forward(prediction, target)

        f1, accuracy, precision, recall = get_metrics_from_confusion_matrix(
            confusion_matrix
        )

        torch.testing.assert_close(accuracy, expected_accuracy)
        torch.testing.assert_close(precision, expected_precision)
        torch.testing.assert_close(recall, expected_recall)
        torch.testing.assert_close(f1, expected_f1)

    # Epoch check
    confusion_matrix = confusion_matrix_metric.compute()

    expected_accuracy = accuracy_metric.compute()
    expected_precision = precision_metric.compute()
    expected_recall = recall_metric.compute()
    expected_f1 = f1_metric.compute()

    f1, accuracy, precision, recall = get_metrics_from_confusion_matrix(
        confusion_matrix
    )
    torch.testing.assert_close(f1, expected_f1)
    torch.testing.assert_close(accuracy, expected_accuracy)
    torch.testing.assert_close(precision, expected_precision)
    torch.testing.assert_close(recall, expected_recall)
