from __future__ import annotations
import logging
import typing
from dataclasses import dataclass, asdict


from am_model_training.setup.abstract import (
    _AbstractInferenceObjects,
    _AbstractMetricsObjects,
)


_logger = logging.getLogger(__name__)


@dataclass
class StageObjects(_AbstractInferenceObjects, _AbstractMetricsObjects):
    def asdict(self) -> dict[str, typing.Any]:
        return asdict(self)
