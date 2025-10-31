from enum import StrEnum

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
]
