from model.resnet import (
    resnet20,
    resnet32,
)
from torchvision.models.resnet import resnet18


def get_model(name: str, num_classes: int = 10):
    match name:
        case "resnet20":
            return resnet20(num_classes=num_classes)
        case "resnet32":
            return resnet32(num_classes=num_classes)
        case "resnet18":
            return resnet18(num_classes=num_classes)
        case _:
            raise ValueError(f"Invalid model name: {name}")
