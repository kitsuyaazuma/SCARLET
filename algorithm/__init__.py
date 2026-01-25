from enum import StrEnum

from algorithm.common import (
    CommonClientArgs,
    CommonClientTrainer,
    CommonMetricType,
    CommonServerArgs,
    CommonServerHandler,
    distill,
    evaulate,
    predict,
    train,
)
from algorithm.dsfl import DSFLClientTrainer, DSFLServerHandler
from algorithm.scarlet import SCARLETClientTrainer, SCARLETServerHandler


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
    "CommonMetricType",
    "evaulate",
    "distill",
    "train",
    "predict",
]
