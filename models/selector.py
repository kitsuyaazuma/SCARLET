from enum import StrEnum

import torch
from blazefl.core import ModelSelector
from torch import nn
from torchvision.models import resnet18

from models.resnet import resnet20


class CommonModelName(StrEnum):
    RESNET20 = "resnet20"
    RESNET18 = "resnet18"


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
                case CommonModelName.RESNET18:
                    return resnet18(num_classes=self.num_classes)
