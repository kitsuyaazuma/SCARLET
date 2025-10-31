from enum import StrEnum

import torch
from blazefl.core import ModelSelector
from torch import nn

from models.resnet import resnet20


class CommonModelName(StrEnum):
    RESNET20 = "RESNET20"


class CommonModelSelector(ModelSelector[CommonModelName]):
    def __init__(self, num_classes: int, seed: int) -> None:
        self.num_classes = num_classes
        self.seed = seed

    def select_model(self, model_name: CommonModelName) -> nn.Module:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            match model_name:
                case CommonModelName.RESNET20:
                    return resnet20(num_classes=self.num_classes)
