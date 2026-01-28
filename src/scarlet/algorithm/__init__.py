from enum import StrEnum

from .common import (
    CommonClientArgs,
    CommonClientTrainer,
    CommonMetricType,
    CommonPipeline,
    CommonServerArgs,
    CommonServerHandler,
    distill,
    evaulate,
    predict,
    train,
)
from .dsfl import DSFLClientTrainer, DSFLServerHandler
from .scarlet import SCARLETClientTrainer, SCARLETServerHandler


class AlgorithmName(StrEnum):
    DSFL = "DSFL"
    SCARLET = "SCARLET"


__all__ = [
    "AlgorithmName",
    "DSFLServerHandler",
    "DSFLClientTrainer",
    "SCARLETServerHandler",
    "SCARLETClientTrainer",
    "CommonServerHandler",
    "CommonClientTrainer",
    "CommonServerArgs",
    "CommonClientArgs",
    "CommonPipeline",
    "CommonMetricType",
    "evaulate",
    "distill",
    "train",
    "predict",
]
