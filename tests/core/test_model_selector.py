from enum import StrEnum

import torch

from scarlet.core.model_selector import ModelSelector


class DummyModelName(StrEnum):
    LINEAR = "linear"
    CONV = "conv"


class DummyModelSelector(ModelSelector[DummyModelName]):
    def select_model(self, model_name: DummyModelName) -> torch.nn.Module:
        match model_name:
            case DummyModelName.LINEAR:
                return torch.nn.Linear(10, 5)
            case DummyModelName.CONV:
                return torch.nn.Conv2d(1, 3, 3)


def test_model_selector_implementation() -> None:
    selector = DummyModelSelector()

    model_linear = selector.select_model(DummyModelName.LINEAR)
    assert isinstance(model_linear, torch.nn.Linear)
    assert model_linear.in_features == 10
    assert model_linear.out_features == 5

    model_conv = selector.select_model(DummyModelName.CONV)
    assert isinstance(model_conv, torch.nn.Conv2d)
    assert model_conv.in_channels == 1
    assert model_conv.out_channels == 3
